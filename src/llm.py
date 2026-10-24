"""Groq HTTP adapter with bounded retries and sanitized, actionable failures."""
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import httpx

BASE_URL = 'https://api.groq.com/openai/v1'


class LLMError(RuntimeError):
    def __init__(self, code, message, attempts=0):
        super().__init__(message)
        self.code, self.attempts = code, attempts


def retry_delay(value, attempt):
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            delay = 2 ** attempt
    return max(0, delay) if math.isfinite(delay) else 2 ** attempt


def token_upper_estimate(messages):
    # Conservative UTF-8-byte proxy, NOT the provider's exact tokenizer count.
    return sum(len(m['content'].encode('utf-8')) + 128 for m in messages) + 128


class GroqClient:
    def __init__(self, key, model='openai/gpt-oss-20b', timeout=20, attempts=3,
                 client=None, sleep=time.sleep):
        if not key or not key.strip():
            raise LLMError('missing_key', 'Set GROQ_API_KEY in local .env or environment; never share it in chat.')
        if not 1 <= attempts <= 5 or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError('Require attempts 1..5 and timeout 0..60 seconds')
        self.model, self.attempts, self.timeout = model, attempts, timeout
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.headers = {'Authorization': 'Bearer ' + key}
        self.sleep = sleep
        self.calls = 0
        self.context_limit = None

    def close(self):
        self.client.close()

    def _request(self, method, path, payload=None):
        for attempt in range(self.attempts):
            self.calls += 1
            try:
                response = self.client.request(method, BASE_URL + path, headers=self.headers,
                                               json=payload, timeout=self.timeout)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt + 1 == self.attempts:
                    raise LLMError('network_timeout', 'Groq network/timeout failure; retry later.', attempt + 1) from None
                self.sleep(min(2 ** attempt, 10))
                continue
            status = response.status_code
            if status in (429, 500, 502, 503, 504):
                delay = retry_delay(response.headers.get('retry-after'), attempt)
                if attempt + 1 == self.attempts or delay > 10:
                    raise LLMError('rate_limit' if status == 429 else 'provider_unavailable',
                                   'Groq is rate limited or unavailable; retry later (no model fallback).', attempt + 1)
                self.sleep(delay)
                continue
            if status >= 300:
                errors = {400: ('invalid_request', 'Invalid request or context limit; reduce context/output budget.'),
                          401: ('authentication', 'Groq rejected credentials; check local GROQ_API_KEY.'),
                          403: ('permission', 'Groq denied access; check model/account permissions.'),
                          404: ('model_unavailable', 'Model or endpoint unavailable; run the models command and select an active ID.'),
                          413: ('token_limit', 'Request too large; reduce context/output budget.'),
                          422: ('invalid_request', 'Unsupported request settings; check model compatibility.')}
                code, message = errors.get(status, ('http_error', f'Groq HTTP {status}; inspect provider documentation.'))
                # Never include raw error bodies, request headers or credentials.
                raise LLMError(code, message, attempt + 1)
            try:
                data = response.json()
            except ValueError:
                raise LLMError('invalid_response', 'Groq returned invalid JSON.', attempt + 1) from None
            if not isinstance(data, dict) or 'error' in data:
                raise LLMError('invalid_response', 'Groq returned an invalid response envelope.', attempt + 1)
            return data

    def models(self):
        data = self._request('GET', '/models').get('data')
        if not isinstance(data, list) or any(not isinstance(m, dict) or not isinstance(m.get('id'), str) for m in data):
            raise LLMError('invalid_response', 'Groq models response is invalid.')
        return data

    def verify_model(self):
        match = next((m for m in self.models() if m.get('id') == self.model and m.get('active', True)), None)
        if match is None:
            raise LLMError('model_unavailable', 'Selected model is not active; run models and set GROQ_MODEL.')
        limit = match.get('context_window')
        if type(limit) is not int or limit < 1024:
            raise LLMError('invalid_response', 'Live model context window is unavailable; check provider metadata.')
        self.context_limit = limit
        return {'id': self.model, 'context_window': limit}

    def complete(self, messages, max_tokens, context_limit):
        if self.context_limit is None:
            raise LLMError('model_unverified', 'Verify the active model before generation.')
        if token_upper_estimate(messages) + max_tokens + 256 > min(context_limit, self.context_limit):
            raise LLMError('token_limit', 'Prompt and output reserve exceed the configured context budget.')
        payload = {'model': self.model, 'messages': messages, 'temperature': 0,
                   'max_completion_tokens': max_tokens,
                   'response_format': {'type': 'json_object'}, 'stream': False}
        if self.model in ('openai/gpt-oss-20b', 'openai/gpt-oss-120b'):
            payload.update(reasoning_effort='low', include_reasoning=False)
        data = self._request('POST', '/chat/completions', payload)
        try:
            choice = data['choices'][0]
            if choice['finish_reason'] != 'stop':
                raise LLMError('incomplete_output', 'Generation was truncated or refused; no answer released.')
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError()
            return content
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError('invalid_response', 'Groq returned no usable completed text.') from None
