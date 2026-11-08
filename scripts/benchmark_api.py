"""Benchmark a running local FastAPI service without calling paid generation."""
import argparse
import json
import math
from pathlib import Path
import statistics
from time import perf_counter

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--query', default='What is an atom made of?')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--report', type=Path, default=Path('reports/api_benchmark.json'))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error('--repeats must be 1..100')
    timings = []
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False) as client:
        health = client.get('/health')
        health.raise_for_status()
        for _ in range(args.repeats):
            start = perf_counter()
            response = client.post('/retrieve', json={'question': args.query, 'top_k': 5})
            response.raise_for_status()
            assert response.json()['hits']
            timings.append((perf_counter() - start) * 1000)
    ordered = sorted(timings)
    report = {'endpoint': '/retrieve', 'base_url': args.base_url, 'question': args.query,
              'repeats': args.repeats, 'mean_ms': statistics.mean(timings),
              'median_ms': statistics.median(timings),
              'p95_ms': ordered[math.ceil(len(ordered)*.95)-1],
              'scope': 'Local HTTP retrieval, includes API and query embedding; excludes startup and Groq generation.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
