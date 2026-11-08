"""Run retrieval evaluation with explicit label provenance."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--db', type=Path, default=Path('vector_store'))
    parser.add_argument('--collection', default='wikipedia_minilm_v1')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--report', type=Path, default=Path('reports/rag_retrieval_evaluation.json'))
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding='utf-8'))
    if labels.get('provenance') not in ('human_reviewed', 'codex_manual') or not labels.get('reviewer'):
        raise ValueError('Labels require explicit manual provenance and reviewer')
    from src.embeddings import Encoder
    from src.vector_store import client_at, open_collection, search
    encoder = Encoder(offline=True)
    collection, _ = open_collection(client_at(args.db), args.collection)
    result = evaluate_retrieval(labels['cases'], lambda q, k: search(collection, encoder, q, k), args.k)
    result['label_provenance'] = {'reviewer': labels['reviewer'], 'type': labels['provenance'],
                                  'source': str(args.labels), 'limitation': labels.get('limitation')}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
