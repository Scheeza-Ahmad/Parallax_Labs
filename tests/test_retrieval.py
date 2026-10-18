import json
import numpy as np
import pytest
from src.chunking import Chunk
from src.vector_store import client_at, open_collection, add_chunks, missing_chunks, search
from scripts.retrieval import load_corpus, latency_summary


class UnitEncoder:
    """Synthetic vectors only for storage/error unit tests; not a semantic evaluation."""
    identity = {'model': 'unit-test', 'dimension': 2}

    def encode(self, texts):
        return np.array([[1., 0.] for _ in texts])


@pytest.fixture
def store(tmp_path):
    client = client_at(tmp_path / 'db')
    signature = {'encoder': UnitEncoder.identity, 'corpus': 'test'}
    collection, _ = open_collection(client, 'unit_collection', signature, True)
    return client, collection, signature


def test_persistent_idempotent_and_ranked(store):
    client, collection, signature = store
    chunks = [Chunk('a', 'Ocean water', {'doc_id': '1', 'language_confidence': 0.9999978173055841}),
              Chunk('b', 'Dry land', {'doc_id': '2'})]
    add_chunks(collection, chunks, [[1, 0], [0, 1]], 2)
    assert missing_chunks(collection, chunks) == []
    collection.modify(metadata=dict(collection.metadata, status='ready'))
    reopened, _ = open_collection(client, 'unit_collection', signature)
    result = search(reopened, UnitEncoder(), 'water', 20)
    assert result['returned'] == 2 and result['hits'][0]['chunk_id'] == 'a'
    assert result['hits'][0]['cosine_distance'] == pytest.approx(0)
    assert result['timing_ms']['total_retrieval'] >= result['timing_ms']['database_search']
    assert collection.count() == 2
    with pytest.raises(ValueError, match='differs'):
        missing_chunks(collection, [Chunk('a', 'Changed', {'doc_id': '1'})])


def test_incompatible_collection_preserved(store):
    client, collection, signature = store
    with pytest.raises(ValueError, match='Incompatible'):
        open_collection(client, 'unit_collection', dict(signature, encoder={'dimension': 3}), True)
    assert collection.metadata['status'] == 'building'
    with pytest.raises(ValueError, match='does not exist'):
        open_collection(client, 'missing')


@pytest.mark.parametrize('query,k', [('', 1), ('  ', 1), ('water', 0), ('water', -1), ('water', True)])
def test_invalid_search(store, query, k):
    with pytest.raises(ValueError):
        search(store[1], UnitEncoder(), query, k)


def test_empty_incomplete_and_wrong_encoder(store):
    collection = store[1]
    with pytest.raises(ValueError, match='incomplete'):
        search(collection, UnitEncoder(), 'water')
    collection.modify(metadata=dict(collection.metadata, status='ready'))
    with pytest.raises(ValueError, match='empty'):
        search(collection, UnitEncoder(), 'water')
    other = UnitEncoder()
    other.identity = {'model': 'wrong', 'dimension': 2}
    with pytest.raises(ValueError, match='incompatible'):
        search(collection, other, 'water')


@pytest.mark.parametrize('chunks,vectors', [([], []), ([Chunk('a', '', {'doc_id': '1'})], [[1, 0]]),
    ([Chunk('a', 'text', {'nested': {}})], [[1, 0]]),
    ([Chunk('a', 'text', {'doc_id': '1'})], [[1, 0, 0]]),
    ([Chunk('a', 'text', {'doc_id': '1'})], [[float('nan'), 0]])])
def test_bad_ingestion(store, chunks, vectors):
    with pytest.raises(ValueError):
        add_chunks(store[1], chunks, vectors, 2)
    assert store[1].count() == 0


def test_input_and_summary(tmp_path):
    path = tmp_path / 'input.jsonl'
    path.write_text('')
    with pytest.raises(ValueError, match='empty'):
        load_corpus(path)
    path.write_text('\n'.join(json.dumps({'doc_id': '1'}) for _ in range(2)))
    with pytest.raises(ValueError, match='Duplicate'):
        load_corpus(path)
    assert latency_summary(list(range(1, 21))) == {'mean': 10.5, 'median': 10.5, 'p95': 19}


def test_failed_write_is_actionable():
    class FailedCollection:
        def add(self, **kwargs):
            raise OSError('disk full')
    with pytest.raises(RuntimeError, match='partial data retained'):
        add_chunks(FailedCollection(), [Chunk('a', 'text', {'doc_id': '1'})], [[1, 0]], 2)


def test_long_query_rejected_before_encoding():
    from src.embeddings import Encoder
    encoder = Encoder.__new__(Encoder)
    encoder.max_seq_length = 3
    encoder.tokenizer = lambda text, **kwargs: {'input_ids': list(range(4))}
    with pytest.raises(ValueError, match='exceeds model limit'):
        encoder.encode(['long query'])
