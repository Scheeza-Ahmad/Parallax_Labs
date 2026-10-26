"""Persistent, signature-checked Chroma storage; never deletes collections."""
import json
import math
from time import perf_counter
import numpy as np
from src.chunking import validate_metadata


def client_at(path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(str(path), settings=Settings(anonymized_telemetry=False))


def open_collection(client, name, signature=None, create=False):
    from chromadb.errors import NotFoundError
    try:
        collection = client.get_collection(name, embedding_function=None)
    except NotFoundError:
        if not create or signature is None:
            raise ValueError(f'Collection {name!r} does not exist; run ingest first') from None
        collection = client.create_collection(name, embedding_function=None,
            metadata={'signature': json.dumps(signature, sort_keys=True), 'status': 'building'},
            configuration={'hnsw': {'space': 'cosine'}})
    try:
        stored = json.loads((collection.metadata or {})['signature'])
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError('Unmanaged collection: select a new --collection name') from None
    if signature is not None and stored != signature:
        raise ValueError('Incompatible collection (model, dimension, corpus or chunk settings). '
                         'Use a new --collection name; existing data was preserved.')
    if collection.configuration.get('hnsw', {}).get('space') != 'cosine':
        raise ValueError('Incompatible distance metric; use a new --collection name')
    return collection, stored


def add_chunks(collection, chunks, vectors, dimension):
    if not chunks:
        raise ValueError('Cannot ingest an empty chunk batch')
    if len({c.id for c in chunks}) != len(chunks):
        raise ValueError('Duplicate IDs in batch')
    for chunk in chunks:
        validate_metadata(chunk.metadata)
        if not chunk.text.strip():
            raise ValueError('Empty chunk text')
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(chunks), dimension) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding dimensions or nonfinite values')
    try:
        collection.add(ids=[c.id for c in chunks], documents=[c.text for c in chunks],
                       metadatas=[c.metadata for c in chunks], embeddings=vectors)
    except Exception as exc:
        raise RuntimeError('Chroma ingestion failed; partial data retained. Rerun identical '
                           'input to resume, or select a new collection.') from exc


def missing_chunks(collection, chunks):
    existing = collection.get(ids=[c.id for c in chunks], include=['documents', 'metadatas'])
    found = {i: (t, m) for i, t, m in zip(existing['ids'], existing['documents'], existing['metadatas'])}
    for chunk in chunks:
        if chunk.id in found:
            text, metadata = found[chunk.id]
            # Chroma's JSON round trip can shift floats by one ULP.
            same = metadata.keys() == chunk.metadata.keys() and all(
                math.isclose(v, metadata[k], rel_tol=1e-14, abs_tol=0.0)
                if type(v) is float and type(metadata[k]) is float else v == metadata[k]
                for k, v in chunk.metadata.items())
            if text != chunk.text or not same:
                raise ValueError('Existing chunk content differs; use a new collection')
    return [c for c in chunks if c.id not in found]


def search(collection, encoder, query, top_k=5, where=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Query must not be blank')
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    signature = json.loads(collection.metadata['signature'])
    if signature['encoder'] != encoder.identity:
        raise ValueError('Query encoder is incompatible with collection')
    if collection.metadata.get('status') != 'ready':
        raise ValueError('Collection is incomplete; rerun ingestion before searching')
    if not collection.count():
        raise ValueError('Collection is empty; ingest documents first')
    k = min(top_k, collection.count())
    started = perf_counter()
    vector = encoder.encode([query])
    embedded = perf_counter()
    result = collection.query(query_embeddings=vector, n_results=k, **({'where': where} if where else {}),
                              include=['documents', 'metadatas', 'distances'])
    finished = perf_counter()
    hits = [dict(rank=i + 1, chunk_id=cid, text=text, metadata=meta,
                 cosine_distance=distance)
            for i, (cid, text, meta, distance) in enumerate(zip(result['ids'][0],
                result['documents'][0], result['metadatas'][0], result['distances'][0]))]
    return dict(query=query, requested_top_k=top_k, returned=len(hits), hits=hits,
                timing_ms={'query_embedding': (embedded-started)*1000,
                           'database_search': (finished-embedded)*1000,
                           'total_retrieval': (finished-started)*1000})
