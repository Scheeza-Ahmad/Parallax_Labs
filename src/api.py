"""HTTP interface for the existing retrieval and grounded-answer pipeline."""
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.nlp import make_filter
from src.rag import answer
from src.llm import LLMError

request_log = logging.getLogger('parallax.api')
if not request_log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    request_log.addHandler(handler)
request_log.setLevel(logging.INFO)
request_log.propagate = False


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    max_distance: float = Field(default=.55, ge=0, le=2)
    topic: int | None = None
    entity: str | None = None
    min_topic_weight: float | None = None


class Service:
    def __init__(self, db='vector_store', collection='wikipedia_minilm_v1', offline=False):
        from src.embeddings import Encoder
        from src.vector_store import client_at, open_collection
        self.encoder = Encoder(offline=offline)
        self.collection, _ = open_collection(client_at(db), collection)

    def retrieve(self, question, top_k, where=None):
        from src.vector_store import search
        return search(self.collection, self.encoder, question, top_k, where=where)


def create_app(service=None, llm=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.service is None:
            app.state.service = Service(db=os.getenv('RAG_DB', 'vector_store'),
                                        collection=os.getenv('RAG_COLLECTION', 'wikipedia_minilm_v1'),
                                        offline=os.getenv('RAG_OFFLINE', 'false').lower() == 'true')
        yield
        if app.state.llm is not None and hasattr(app.state.llm, 'close'):
            app.state.llm.close()

    app = FastAPI(title='Parallax Labs RAG API', version='0.1.0', lifespan=lifespan,
                  description='Historical Simple English Wikipedia retrieval and source-checked answers. '
                              'CUDA is unavailable in the verified local environment.')
    app.state.service, app.state.llm = service, llm

    @app.middleware('http')
    async def request_logging(request: Request, call_next):
        started = perf_counter()
        request_id = uuid4().hex
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers['X-Request-ID'] = request_id
            return response
        except Exception:
            request_log.error(json.dumps({'request_id': request_id, 'event': 'unhandled_exception'}))
            return JSONResponse(status_code=500, content={'error': {'code': 'internal_error',
                'message': 'Internal server error', 'request_id': request_id}},
                headers={'X-Request-ID': request_id})
        finally:
            request_log.info(json.dumps({'request_id': request_id, 'method': request.method,
                'path': request.url.path, 'status': status,
                'duration_ms': round((perf_counter() - started) * 1000, 3)}))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={'error': {'code': 'validation_error',
            'message': 'Request body or query parameters are invalid'}})

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={'error': {
            'code': 'bad_request' if exc.status_code == 400 else
                    'validation_error' if exc.status_code == 422 else
                    'service_unavailable' if exc.status_code == 503 else 'upstream_error',
            'message': exc.detail}})

    def get_service():
        if app.state.service is None:
            raise HTTPException(503, 'Retrieval service is unavailable')
        return app.state.service

    @app.get('/health')
    def health(svc=Depends(get_service)):
        return {'status': 'ok', 'stored_chunks': svc.collection.count()}

    @app.post('/retrieve')
    def retrieve(query: Query, svc=Depends(get_service)):
        try:
            where = make_filter(query.topic, query.entity, query.min_topic_weight)
            if not query.question.strip():
                raise HTTPException(400, 'Question must not be blank')
            return svc.retrieve(query.question, query.top_k, where=where)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/ask')
    def ask(query: Query, svc=Depends(get_service)):
        if app.state.llm is None:
            raise HTTPException(503, 'Groq generation is not configured')
        try:
            where = make_filter(query.topic, query.entity, query.min_topic_weight)
            if not query.question.strip():
                raise HTTPException(400, 'Question must not be blank')
            if app.state.llm.context_limit is None:
                app.state.llm.verify_model()
            retrieve_fn = lambda question, k: svc.retrieve(question, k, where=where)
            result = answer(query.question, retrieve_fn, app.state.llm, query.top_k, query.max_distance)
            if result['status'] == 'error':
                raise HTTPException(502, result.get('error', {}).get('message', 'Generation failed'))
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc

    return app


def build_app():
    load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    key = os.getenv('GROQ_API_KEY')
    llm = None
    if key:
        from src.llm import GroqClient
        llm = GroqClient(key, os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'))
    return create_app(llm=llm)


app = build_app()
