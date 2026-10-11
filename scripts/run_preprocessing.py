"""Clean, filter, deduplicate and read back a JSONL corpus before publishing it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import clean_text, detect_language


def run(input_path, output_path, report_path, minimum):
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.partial')
    counts = Counter()
    lengths, ids, hashes = [], set(), set()
    with input_path.open(encoding='utf-8') as source, temporary.open('x', encoding='utf-8') as output:
        for line in source:
            counts['input'] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts['malformed_json'] += 1
                continue
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ['id', 'url', 'title']):
                counts['invalid_schema'] += 1
                continue
            text = clean_text(row.get('text'))
            if not text:
                counts['empty_text'] += 1
                continue
            language, confidence = detect_language(text)
            if language != 'en':
                counts['uncertain_language' if language == 'und' else 'non_english'] += 1
                continue
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if row['id'] in ids or digest in hashes:
                counts['duplicate'] += 1
                continue
            ids.add(row['id'])
            hashes.add(digest)
            result = {'doc_id': row['id'], 'title': row['title'], 'text': text,
                      'source': 'Simple English Wikipedia', 'source_url': row['url'],
                      'language': language, 'language_confidence': confidence}
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            lengths.append(len(text))
            counts['accepted'] += 1
    checked_ids, checked_hashes = set(), set()
    with temporary.open(encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            for key in ['doc_id', 'title', 'text', 'source', 'source_url', 'language']:
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'Invalid output field: {key}')
            digest = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
            if row['doc_id'] in checked_ids or digest in checked_hashes or row['language'] != 'en':
                raise ValueError('Invalid or duplicate output')
            checked_ids.add(row['doc_id'])
            checked_hashes.add(digest)
    if len(checked_ids) != counts['accepted']:
        raise ValueError('Read-back count mismatch')
    report = {'input': str(input_path), 'output': str(output_path),
              'counts': {key: counts[key] for key in ['input', 'accepted', 'empty_text', 'duplicate', 'invalid_schema', 'malformed_json', 'uncertain_language', 'non_english']},
              'rejected': counts['input'] - counts['accepted'],
              'length_characters': {'min': min(lengths, default=0), 'max': max(lengths, default=0),
                                    'mean': statistics.mean(lengths) if lengths else 0,
                                    'median': statistics.median(lengths) if lengths else 0},
              'configuration': {'unicode': 'NFC', 'target_language': 'en', 'detector': 'langdetect',
                                'seed': 0, 'minimum_letters': 50, 'confidence_threshold': 0.9,
                                'detection_max_characters': 10000, 'deduplication': 'source ID or exact cleaned-text SHA256'},
              'validated_records': len(checked_ids), 'minimum_required': minimum,
              'passed': len(checked_ids) >= minimum}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise ValueError(f'Fewer than {minimum} usable documents; partial file retained for diagnosis')
    temporary.rename(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/raw/wikipedia-7000.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/clean_corpus.jsonl'))
    parser.add_argument('--report', type=Path, default=Path('reports/preprocessing.json'))
    parser.add_argument('--minimum', type=int, default=5000)
    args = parser.parse_args()
    if args.minimum < 1:
        parser.error('--minimum must be positive')
    run(args.input, args.output, args.report, args.minimum)
