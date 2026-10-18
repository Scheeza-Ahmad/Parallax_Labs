"""Index cleaned JSONL, search persistent Chroma, and benchmark retrieval latency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import statistics
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import chunk_document
from src.embeddings import Encoder, MODEL, REVISION
from src.vector_store import client_at, open_collection, add_chunks, missing_chunks, search


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


def load_corpus(path, limit=None):
    rows, ids = [], set()
    with path.open(encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('doc_id'), str):
                raise ValueError(f'Invalid schema on line {line_number}')
            if row['doc_id'] in ids:
                raise ValueError(f'Duplicate document ID on line {line_number}')
            ids.add(row['doc_id'])
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError('Input corpus is empty')
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return rows, digest


def ingest(args, encoder):
    rows, digest = load_corpus(args.input, args.limit)
    started = perf_counter()
    chunks = []
    for row in rows:
        parts = chunk_document(row, encoder.tokenizer, args.chunk_size, args.overlap,
                               encoder.max_seq_length)
        if not parts:
            raise ValueError(f'Empty document {row["doc_id"]}; cleaned corpus must be nonempty')
        chunks.extend(parts)
    chunk_seconds = perf_counter() - started
    print(f'Prepared {len(rows)} documents / {len(chunks)} chunks in {chunk_seconds:.2f}s', flush=True)
    signature = {'encoder': encoder.identity, 'corpus_sha256': digest,
                 'documents': len(rows), 'chunks': len(chunks), 'strategy': 'token-window-v1',
                 'chunk_size': args.chunk_size, 'overlap': args.overlap}
    client = client_at(args.db)
    collection, _ = open_collection(client, args.collection, signature, create=True)
    encoded_count = skipped = 0
    encoding_seconds = write_seconds = 0.0
    for start in range(0, len(chunks), args.write_batch):
        batch = chunks[start:start + args.write_batch]
        pending = missing_chunks(collection, batch)
        skipped += len(batch) - len(pending)
        if pending:
            before = perf_counter()
            vectors = encoder.encode([c.text for c in pending])
            after = perf_counter()
            encoding_seconds += after - before
            add_chunks(collection, pending, vectors, encoder.identity['dimension'])
            write_seconds += perf_counter() - after
            encoded_count += len(pending)
        print(f'Checked {min(start + len(batch), len(chunks))}/{len(chunks)} chunks; '
              f'encoded={encoded_count}, skipped={skipped}, encoding_seconds={encoding_seconds:.2f}',
              flush=True)
    if collection.count() != len(chunks):
        raise ValueError('Stored count differs from expected; use a new collection')
    # Read back all expected IDs/text/metadata, including the final newly written batch.
    for start in range(0, len(chunks), args.write_batch):
        if missing_chunks(collection, chunks[start:start + args.write_batch]):
            raise ValueError('Read-back validation found missing chunks; rerun ingestion')
    collection.modify(metadata=dict(collection.metadata, status='ready'))
    report = dict(encoder=encoder.details, signature=signature, input=str(args.input),
                  sample_only=args.limit is not None, database=str(args.db), collection=args.collection,
                  encoded_chunks=encoded_count, skipped_existing_chunks=skipped,
                  stored_chunks=collection.count(), chunking_seconds=chunk_seconds,
                  encoding_seconds=encoding_seconds, database_write_seconds=write_seconds,
                  encoding_chunks_per_second=encoded_count/encoding_seconds if encoding_seconds else None,
                  readback_passed=True)
    save_report(args.report or Path('reports/index.json'), report)
    print(json.dumps(report, indent=2))


def latency_summary(values):
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'p95': sorted(values)[math.ceil(.95 * len(values)) - 1]}


def benchmark(args, collection, encoder):
    queries = json.loads(args.queries.read_text(encoding='utf-8'))
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError('Queries must be a nonempty JSON list of nonblank strings')
    warm_started = perf_counter()
    for query in queries:
        search(collection, encoder, query, args.top_k)
    warmup_seconds = perf_counter() - warm_started
    results = []
    for repeat in range(args.repeats):
        for query in queries:
            result = search(collection, encoder, query, args.top_k)
            result['repeat'] = repeat + 1
            for hit in result['hits']:
                hit['snippet'] = hit.pop('text')[:350]
            results.append(result)
    summary = {key: latency_summary([r['timing_ms'][key] for r in results])
               for key in ('query_embedding', 'database_search', 'total_retrieval')}
    report = dict(encoder=encoder.details, collection=args.collection,
                  collection_signature=json.loads(collection.metadata['signature']),
                  stored_chunks=collection.count(), query_count=len(queries), repeats=args.repeats,
                  measured_searches=len(results), warmup_searches=len(queries),
                  warmup_seconds=warmup_seconds, summary_ms=summary,
                  measurement='perf_counter; encoding includes token validation and numpy conversion; '
                  'database search includes Chroma query and result deserialization; total is their sum. '
                  'Excludes process/import/model/client startup, count/compatibility checks, warm-up, '
                  'hit formatting, console and report I/O. p95 uses nearest rank.',
                  relevance='Unlabelled exploratory queries; no precision or recall claimed.', results=results)
    save_report(args.report or Path('reports/retrieval_benchmark.json'), report)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ingest', 'search', 'benchmark'])
    parser.add_argument('--db', type=Path, default=Path('vector_store'))
    parser.add_argument('--collection', default='wikipedia_minilm_v1')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--input', type=Path, default=Path('data/processed/clean_corpus.jsonl'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--chunk-size', type=int, default=220)
    parser.add_argument('--overlap', type=int, default=32)
    parser.add_argument('--write-batch', type=int, default=256)
    parser.add_argument('--query')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--queries', type=Path, default=Path('benchmark_queries.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if any(v < 1 for v in (args.batch_size, args.threads, args.write_batch, args.top_k, args.repeats)):
        parser.error('Batch sizes, threads, top-k and repeats must be positive')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.command == 'search' and (not args.query or not args.query.strip()):
        parser.error('--query must not be blank')
    try:
        print(f'Loading encoder on {args.device}; CPU runs do not satisfy the Week 1 CUDA check.',
              file=sys.stderr, flush=True)
        encoder = Encoder(args.model, args.revision, args.device, args.batch_size,
                          offline=args.offline, threads=args.threads)
        if args.command == 'ingest':
            ingest(args, encoder)
        else:
            collection, _ = open_collection(client_at(args.db), args.collection)
            if args.command == 'benchmark':
                benchmark(args, collection, encoder)
            else:
                result = search(collection, encoder, args.query, args.top_k)
                result['encoder'] = encoder.details
                result['stored_chunks'] = collection.count()
                if args.report:
                    save_report(args.report, result)
                print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
