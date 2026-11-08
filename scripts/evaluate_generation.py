"""Automated end-to-end status, grounding and latency evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from src.rag import answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('evaluation/generation_cases.json'))
    parser.add_argument('--report', type=Path, default=Path('reports/week5_generation.json'))
    parser.add_argument('--dry-run', action='store_true', help='No provider calls or generation quality score')
    parser.add_argument('--db', type=Path, default=Path('vector_store'))
    parser.add_argument('--collection', default='wikipedia_minilm_v1')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases or any(c.get('expected') not in
            ('answered', 'off_topic', 'insufficient_context') for c in cases):
        raise ValueError('Expected nonempty case list with valid expected statuses')
    load_dotenv(ROOT / '.env', override=False)
    llm = None
    if not args.dry_run:
        from src.llm import GroqClient
        llm = GroqClient(os.getenv('GROQ_API_KEY'), os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'))
        llm.verify_model()
    try:
        from src.embeddings import Encoder
        from src.vector_store import client_at, open_collection, search
        encoder = Encoder(offline=True)
        collection, _ = open_collection(client_at(args.db), args.collection)
        results = []
        for case in cases:
            retrieve = lambda q, k: search(collection, encoder, q, k)
            outcome = answer(case['question'], retrieve, llm=llm, dry_run=args.dry_run)
            results.append({'question': case['question'], 'expected': case['expected'],
                'observed': outcome['status'], 'status_match': None if args.dry_run else outcome['status'] == case['expected'],
                'grounding': outcome['grounding'], 'timing': outcome['timing'],
                'api_attempts': outcome['api_attempts'], 'error': outcome.get('error')})
        live = not args.dry_run
        report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'mode': 'live' if live else 'dry_run', 'cases': len(results),
            'status_accuracy': sum(r['status_match'] for r in results)/len(results) if live else None,
            'all_answered_grounded': all(r['grounding']['exact_quotes'] and r['grounding']['llm_verifier']
                for r in results if r['observed'] == 'answered') if live else None,
            'mean_end_to_end_ms': sum(r['timing']['end_to_end_ms'] for r in results)/len(results),
            'limitation': 'Expected statuses are Codex-authored; status agreement and automated grounding checks do not establish factual correctness.' +
                (' Dry run contains no generated answers.' if args.dry_run else ''),
            'results': results}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
        return 1 if live and any(r['observed'] in ('error','withheld') for r in results) else 0
    finally:
        if llm:
            llm.close()


if __name__ == '__main__':
    raise SystemExit(main())
