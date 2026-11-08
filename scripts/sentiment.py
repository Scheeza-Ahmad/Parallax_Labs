"""Score the whole cleaned corpus and save an exploratory summary."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sentiment import analyze
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main() -> None:
    """Score every cleaned article and save an exploratory count summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/clean_corpus.jsonl'))
    parser.add_argument('--report', type=Path, default=Path('reports/sentiment.json'))
    args = parser.parse_args()
    counts = Counter()
    analyzer = SentimentIntensityAnalyzer()
    with args.input.open(encoding='utf-8') as file:
        for line in file:
            counts[analyze(json.loads(line).get('text'), analyzer)['label']] += 1
    report = {'method': 'VADER, first 10000 characters per article', 'documents': sum(counts.values()),
              'counts': dict(counts), 'limitation': 'Lexicon sentiment is exploratory; neutral encyclopedia text and long articles can be misclassified. No labelled accuracy measured.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
