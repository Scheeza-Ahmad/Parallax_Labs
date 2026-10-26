"""Week 4: analyze corpus, clone/enrich vectors, evaluate NER, filtered search."""
import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp import discover_topics, extract_entities, metadata_for, make_filter, evaluate_entities, fingerprint
from src.vector_store import client_at, open_collection, search
from scripts.retrieval import load_corpus, save_report


def analyze(args):
    import spacy
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    rows, digest = load_corpus(args.input)
    started = perf_counter()
    vectorizer, model, assignments, topics = discover_topics([r['text'] for r in rows], args.topics)
    print('Topics fitted; extracting entities from all documents.', flush=True)
    nlp = spacy.load('en_core_web_sm', enable=['tok2vec', 'ner'])
    records = [{'doc_id': row['doc_id'], **a, 'entities': []} for row, a in zip(rows, assignments)]
    def segments():
        for i, row in enumerate(rows):
            for start in range(0, len(row['text']), 50000):
                yield row['text'][start:start+50000], (i, start)
    from src.nlp import ENTITY_TYPES
    for doc, (i, start) in nlp.pipe(segments(), as_tuples=True, batch_size=8):
        records[i]['entities'].extend({'start': start+e.start_char, 'end': start+e.end_char,
                                      'label': e.label_, 'text': e.text}
                                     for e in doc.ents if e.label_ in ENTITY_TYPES)
        if (i + 1) % 500 == 0:
            print(f'NER {i+1}/{len(rows)}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = {'corpus_sha256': digest, 'records': records, 'topics': topics,
                'configuration': {'seed': 42, 'topics': args.topics, 'iterations': 10,
                                  'model': 'en_core_web_sm', 'model_version': nlp.meta['version']}}
    (args.output / 'metadata.json').write_text(json.dumps(artifact, ensure_ascii=False), encoding='utf-8')
    joblib.dump({'vectorizer': vectorizer, 'model': model}, args.output / 'topic_model.joblib')
    for topic in topics:
        members = sorted([(r, a) for r, a in zip(rows, assignments) if a['topic_id'] == topic['topic_id']],
                         key=lambda x: x[1]['topic_weight'], reverse=True)
        topic['documents'] = len(members)
        topic['examples'] = [{'doc_id': r['doc_id'], 'title': r.get('title'), 'excerpt': r['text'][:600],
                              'source_url': r.get('source_url'), 'topic_weight': a['topic_weight']} for r, a in members[:3]]
    save_report(args.report, {'documents': len(rows), 'corpus_sha256': digest, 'topics': topics,
                'unassigned': sum(a['topic_id'] == -1 for a in assignments),
                'entity_counts': {label: sum(e['label'] == label for r in records for e in r['entities'])
                                  for label in ('PERSON','ORG','GPE','LOC','DATE')},
                'elapsed_seconds': perf_counter()-started, 'configuration': artifact['configuration']})


def enrich(args):
    import numpy as np
    artifact = json.loads((args.output / 'metadata.json').read_text(encoding='utf-8'))
    with client_at(args.db) as client:
        source, signature = open_collection(client, args.source)
        if args.source == args.collection or source.metadata.get('status') != 'ready':
            raise ValueError('Use a distinct target and a ready source collection')
        if signature['corpus_sha256'] != artifact['corpus_sha256']:
            raise ValueError('NLP corpus does not match source collection')
        records = {r['doc_id']: r for r in artifact['records']}
        if len(records) != len(artifact['records']) or len(records) != signature['documents']:
            raise ValueError('NLP document coverage mismatch')
        enriched_signature = dict(signature, nlp_sha256=fingerprint(artifact))
        target, _ = open_collection(client, args.collection, enriched_signature, create=True)
        added = skipped = 0
        for offset in range(0, source.count(), 256):
            batch = source.get(limit=256, offset=offset, include=['documents','metadatas','embeddings'])
            metas = [dict(m, **metadata_for(records[m['doc_id']] )) for m in batch['metadatas']]
            existing = set(target.get(ids=batch['ids'], include=[])['ids'])
            indices = [i for i, cid in enumerate(batch['ids']) if cid not in existing]
            if indices:
                target.add(ids=[batch['ids'][i] for i in indices], documents=[batch['documents'][i] for i in indices],
                           metadatas=[metas[i] for i in indices], embeddings=batch['embeddings'][indices])
            check = target.get(ids=batch['ids'], include=['documents','metadatas','embeddings'])
            lookup = {cid: i for i, cid in enumerate(check['ids'])}
            for i, cid in enumerate(batch['ids']):
                j = lookup[cid]
                actual = check['metadatas'][j]
                same_metadata = actual.keys() == metas[i].keys() and all(
                    math.isclose(v, actual[k], rel_tol=1e-14, abs_tol=0.)
                    if type(v) is float and type(actual[k]) is float else v == actual[k]
                    for k, v in metas[i].items())
                if check['documents'][j] != batch['documents'][i] or not same_metadata or not np.allclose(check['embeddings'][j],batch['embeddings'][i],rtol=1e-6,atol=1e-7):
                    raise ValueError('Enriched read-back mismatch; use a new collection')
            added += len(indices); skipped += len(existing)
            if (offset // 256 + 1) % 16 == 0:
                print(f'Validated {offset + len(batch["ids"])}/{source.count()} enriched chunks', flush=True)
        if target.count() != source.count():
            raise ValueError('Enriched collection count mismatch')
        target.modify(metadata=dict(target.metadata, status='ready'))
        save_report(args.report, {'source': args.source, 'target': args.collection, 'chunks': target.count(),
                    'added': added, 'skipped': skipped, 'readback_passed': True, 'original_preserved': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['analyze','enrich','evaluate','search'])
    p.add_argument('--input', type=Path, default=Path('data/processed/clean_corpus.jsonl'))
    p.add_argument('--output', type=Path, default=Path('data/nlp'))
    p.add_argument('--topics', type=int, default=10)
    p.add_argument('--db', type=Path, default=Path('vector_store'))
    p.add_argument('--source', default='wikipedia_minilm_v1')
    p.add_argument('--collection', default='wikipedia_nlp_v2')
    p.add_argument('--report', type=Path, default=Path('reports/nlp.json'))
    p.add_argument('--labels', type=Path, default=Path('evaluation/ner_labels.json'))
    p.add_argument('--query'); p.add_argument('--topic', type=int)
    p.add_argument('--entity'); p.add_argument('--min-weight', type=float)
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    if args.command == 'analyze': analyze(args)
    elif args.command == 'enrich': enrich(args)
    elif args.command == 'evaluate':
        import spacy
        labels = json.loads(args.labels.read_text(encoding='utf-8'))
        nlp = spacy.load('en_core_web_sm', enable=['tok2vec','ner'])
        report = evaluate_entities(nlp, labels['examples'])
        report['model'] = {'name': 'en_core_web_sm', 'version': nlp.meta['version']}
        report['annotation_provenance'] = labels['annotation_provenance']
        save_report(args.report, report)
    else:
        from src.embeddings import Encoder
        encoder = Encoder(offline=True)
        collection, _ = open_collection(client_at(args.db), args.collection)
        where = make_filter(args.topic, args.entity, args.min_weight)
        result = search(collection, encoder, args.query, args.top_k, where=where)
        result['filter'] = where
        save_report(args.report, result)
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
