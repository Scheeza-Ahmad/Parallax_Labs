from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from src.api import create_app
from src.api import Query


class FakeCollection:
    def count(self):
        return 7


class FakeService:
    collection = FakeCollection()

    def retrieve(self, question, top_k, where=None):
        return {'query': question, 'hits': [], 'where': where}


def test_health_retrieve_and_validation():
    app = create_app(service=FakeService())
    routes = {route.path: route.endpoint for route in app.routes}
    assert routes['/health'](svc=FakeService()) == {'status': 'ok', 'stored_chunks': 7}
    assert routes['/retrieve'](Query(question='What is an atom?', entity='PERSON'), svc=FakeService())['where'] == {'has_person': True}
    with pytest.raises(ValueError):
        Query(question='')
    with pytest.raises(HTTPException) as exc:
        routes['/ask'](Query(question='What is an atom?'), svc=FakeService())
    assert exc.value.status_code == 503
    assert '/ask' in app.openapi()['paths']


def test_http_endpoints_and_errors():
    with TestClient(create_app(service=FakeService())) as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['stored_chunks'] == 7
        assert response.headers['x-request-id']
        response = client.post('/retrieve', json={'question': 'Alan Turing', 'entity': 'PERSON'})
        assert response.status_code == 200
        assert response.json()['where'] == {'has_person': True}
        assert client.post('/retrieve', json={'question': '   '}).status_code == 400
        assert client.post('/retrieve', json={'question': ''}).status_code == 422
        assert client.post('/retrieve', json={'question': 'x', 'top_k': 0}).status_code == 422
        assert client.post('/ask', json={'question': 'Alan Turing'}).status_code == 503
        assert client.get('/openapi.json').status_code == 200


def test_http_internal_error_is_sanitized():
    class BrokenService(FakeService):
        def retrieve(self, question, top_k, where=None):
            raise RuntimeError('private-internal-detail')

    with TestClient(create_app(service=BrokenService())) as client:
        response = client.post('/retrieve', json={'question': 'atom'})
        assert response.status_code == 500
        assert response.json()['error']['code'] == 'internal_error'
        assert 'private-internal-detail' not in response.text
