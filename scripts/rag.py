"""Week 3 CLI: models, grounded answer, and end-to-end benchmark."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from src.llm import GroqClient, LLMError
from src.rag import answer
from scripts.retrieval import latency_summary


def main():
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['models', 'ask', 'benchmark'])
    parser.add_argument('--query')
    parser.add_argument('--model', default=os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Offline embedding model only; live LLM still needs network')
    parser.add_argument('--db', type=Path, default=Path('vector_store'))
    parser.add_argument('--collection', default='wikipedia_minilm_v1')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-distance', type=float, default=.55)
    parser.add_argument('--context-tokens', type=int, default=int(os.getenv('RAG_CONTEXT_TOKENS', '8192')))
    parser.add_argument('--output-tokens', type=int, default=int(os.getenv('RAG_MAX_OUTPUT_TOKENS', '1024')))
    parser.add_argument('--queries', type=Path, default=Path('rag_queries.json'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.command == 'ask' and (not args.query or not args.query.strip()):
        parser.error('--query must not be blank')
    if not 1 <= args.repeats <= 5:
        parser.error('--repeats must be 1..5')
    llm = None
    try:
        started = perf_counter()
        model_info = None
        if not args.dry_run or args.command == 'models':
            llm = GroqClient(os.getenv('GROQ_API_KEY'), args.model,
                             timeout=float(os.getenv('GROQ_TIMEOUT_SECONDS', '20')),
                             attempts=int(os.getenv('GROQ_MAX_ATTEMPTS', '3')))
            if args.command == 'models':
                print(json.dumps([{'id': m.get('id'), 'active': m.get('active'),
                                   'context_window': m.get('context_window')}
                                  for m in llm.models()], indent=2))
                return 0
            model_info = llm.verify_model()
            args.context_tokens = min(args.context_tokens, model_info['context_window'])
        discovery_seconds = perf_counter() - started
        from src.embeddings import Encoder
        from src.vector_store import client_at, open_collection, search
        startup = perf_counter()
        encoder = Encoder(offline=args.offline)
        collection, _ = open_collection(client_at(args.db), args.collection)
        startup_seconds = perf_counter() - startup
        retrieve = lambda q, k: search(collection, encoder, q, k)
        queries = [args.query] if args.command == 'ask' else json.loads(args.queries.read_text(encoding='utf-8'))
        if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError('Queries must be a nonempty list of nonblank strings')
        results = []
        for repeat in range(args.repeats if args.command == 'benchmark' else 1):
            for query in queries:
                result = answer(query, retrieve, llm, args.top_k, args.max_distance,
                                args.context_tokens, args.output_tokens, args.dry_run)
                result['repeat'] = repeat + 1
                results.append(result)
                print(f'{result["status"]}: {ascii(query)}', file=sys.stderr, flush=True)
        report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                  'mode': 'dry_run' if args.dry_run else 'live', 'model': args.model,
                  'live_model_metadata': model_info, 'model_discovery_seconds': discovery_seconds,
                  'embedding_and_database_startup_seconds': startup_seconds,
                  'encoder': encoder.details, 'collection': args.collection, 'stored_chunks': collection.count(),
                  'query_count': len(queries), 'repeats': args.repeats if args.command == 'benchmark' else 1,
                  'measurement': 'No generation warm-up. End-to-end includes retrieval, prompt packing, generation, '
                  'verification and retry waits; excludes model discovery/startup and report/console I/O. '
                  'All outcomes retained; successful answers summarized separately. Dry-run is not generation evidence.',
                  'summary_all_ms': latency_summary([r['timing']['end_to_end_ms'] for r in results]),
                  'summary_answered_ms': latency_summary([r['timing']['end_to_end_ms'] for r in results
                                                          if r['status'] == 'answered'])
                    if any(r['status'] == 'answered' for r in results) else None,
                  'results': results}
        path = args.report or Path('data/rag_runs') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        # ASCII JSON survives legacy Windows console encodings; report remains UTF-8.
        print(json.dumps(report, ensure_ascii=True, indent=2))
        print(f'Report: {path}', file=sys.stderr)
        return 1 if any(r['status'] in ('error', 'withheld') for r in results) else 0
    except (LLMError, ValueError, OSError) as exc:
        # Application errors are sanitized. Do not print raw transport errors or .env.
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        if llm:
            llm.close()


if __name__ == '__main__':
    raise SystemExit(main())
