import json
import httpx
import pytest
from src.llm import GroqClient, LLMError, retry_delay, token_upper_estimate


def make_client(handler, **kwargs):
    return GroqClient('unit-test-secret', client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_live_model_validation_and_payload():
    def handler(request):
        assert request.headers['authorization'] == 'Bearer unit-test-secret'
        if request.method == 'GET':
            return httpx.Response(200, json={'data': [{'id': 'openai/gpt-oss-20b', 'active': True, 'context_window': 8192}]})
        payload = json.loads(request.content)
        assert payload['max_completion_tokens'] == 128
        assert payload['response_format'] == {'type': 'json_object'}
        assert payload['reasoning_effort'] == 'low'
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    client = make_client(handler)
    assert client.verify_model()['context_window'] == 8192
    assert client.complete([{'role': 'user', 'content': 'JSON please'}], 128, 8192) == '{}'
    assert client.calls == 2


@pytest.mark.parametrize('status,code', [(400,'invalid_request'), (401,'authentication'), (403,'permission'),
    (404,'model_unavailable'), (413,'token_limit'), (422,'invalid_request'), (302,'http_error')])
def test_permanent_errors_not_retried_or_leaked(status, code):
    client = make_client(lambda r: httpx.Response(status, text='unit-test-secret PRIVATE PROMPT'))
    with pytest.raises(LLMError) as error:
        client.models()
    assert error.value.code == code
    assert client.calls == 1
    assert 'unit-test-secret' not in str(error.value) and 'PRIVATE' not in str(error.value)


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_transient_retry_then_success(status):
    attempts, sleeps = [], []
    def handler(request):
        attempts.append(1)
        return httpx.Response(status, headers={'retry-after': '0.1'}) if len(attempts) == 1 else httpx.Response(200, json={'data': []})
    client = make_client(handler, sleep=sleeps.append)
    assert client.models() == [] and client.calls == 2 and sleeps == [.1]


def test_rate_limit_long_wait_and_exhaustion():
    client = make_client(lambda r: httpx.Response(429, headers={'retry-after': '120'}))
    with pytest.raises(LLMError, match='rate limited'):
        client.models()
    assert client.calls == 1
    sleeps = []
    client = make_client(lambda r: httpx.Response(503), sleep=sleeps.append)
    with pytest.raises(LLMError):
        client.models()
    assert client.calls == 3 and sleeps == [1, 2]


def test_timeout_bounded_and_sanitized():
    def handler(request):
        raise httpx.ReadTimeout('unit-test-secret', request=request)
    client = make_client(handler, sleep=lambda _: None)
    with pytest.raises(LLMError) as error:
        client.models()
    assert error.value.code == 'network_timeout' and client.calls == 3
    assert 'unit-test-secret' not in str(error.value)


@pytest.mark.parametrize('payload', [{'error': {'message': 'private'}}, [], {'data': None}])
def test_malformed_models(payload):
    client = make_client(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(LLMError):
        client.models()


def test_model_not_active_and_missing_key():
    with pytest.raises(LLMError, match='GROQ_API_KEY'):
        GroqClient('')
    client = make_client(lambda r: httpx.Response(200, json={'data': []}))
    with pytest.raises(LLMError) as error:
        client.verify_model()
    assert error.value.code == 'model_unavailable'


@pytest.mark.parametrize('finish,content', [('length','{}'), ('content_filter','{}'), ('stop',None)])
def test_incomplete_response_rejected(finish, content):
    client = make_client(lambda r: httpx.Response(200, json={'choices': [{'finish_reason':finish,'message':{'content':content}}]}))
    client.context_limit = 8192
    with pytest.raises(LLMError):
        client.complete([], 128, 8192)


def test_token_budget_before_network_and_retry_values():
    client = make_client(lambda r: pytest.fail('Must not send oversized request'))
    client.context_limit = 2048
    with pytest.raises(LLMError) as error:
        client.complete([{'role':'user','content':'hello '*1000}], 128, 2048)
    assert error.value.code == 'token_limit' and client.calls == 0
    assert token_upper_estimate([{'content':'\u65e5'}]) > token_upper_estimate([{'content':'a'}])
    assert retry_delay('nan', 1) == 2
    assert retry_delay('bad', 2) == 4
