"""Deterministic corpus topics, named entities, and explicit metadata filters."""
import hashlib
import json
import re

ENTITY_TYPES = ('PERSON', 'ORG', 'GPE', 'LOC', 'DATE')


def discover_topics(texts: list[str], topics: int = 10, iterations: int = 10) -> tuple:
    """Fit deterministic LDA and return model, document assignments and terms."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    if len(texts) < 2 or not 2 <= topics <= len(texts):
        raise ValueError('Require at least two documents and 2..document-count topics')
    vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=.95,
                                 max_features=15000, token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b')
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        raise ValueError('Corpus has insufficient vocabulary for topic modeling') from exc
    model = LatentDirichletAllocation(n_components=topics, max_iter=iterations,
        learning_method='batch', random_state=42, n_jobs=1)
    weights = model.fit_transform(matrix)
    vocabulary = vectorizer.get_feature_names_out()
    descriptions = [{'topic_id': i, 'terms': vocabulary[row.argsort()[-12:][::-1]].tolist()}
                    for i, row in enumerate(model.components_)]
    assignments = []
    for text, counts, scores in zip(texts, np.asarray(matrix.sum(axis=1)).ravel(), weights):
        short = len(re.findall(r'\b\w+\b', text)) < 20
        unknown = counts == 0 or short
        assignments.append({'topic_id': -1 if unknown else int(scores.argmax()),
                            'topic_weight': 0.0 if unknown else float(scores.max()),
                            'topic_status': 'short' if short else 'out_of_vocabulary' if counts == 0 else 'assigned'})
    return vectorizer, model, assignments, descriptions


def extract_entities(nlp: object, text: str) -> list[dict[str, object]]:
    """Extract selected entity types across bounded text windows with offsets."""
    # Process every character in bounded windows; entity offsets remain document-relative.
    entities = []
    for start in range(0, len(text), 50000):
        doc = nlp(text[start:start + 50000])
        entities.extend({'start': start + e.start_char, 'end': start + e.end_char,
                         'label': e.label_, 'text': e.text}
                        for e in doc.ents if e.label_ in ENTITY_TYPES)
    return entities


def metadata_for(record: dict[str, object]) -> dict[str, object]:
    """Convert document-level topic and entities to filterable Chroma scalars."""
    labels = {e['label'] for e in record['entities']}
    return {**{k: record[k] for k in ('topic_id', 'topic_weight', 'topic_status')},
            **{'has_' + label.lower(): label in labels for label in ENTITY_TYPES},
            'entity_count': len(record['entities'])}


def make_filter(topic: int | None = None, entity: str | None = None,
                min_weight: float | None = None) -> dict | None:
    """Build a strict metadata filter without silently widening search scope."""
    clauses = []
    if topic is not None:
        if type(topic) is not int or topic < -1:
            raise ValueError('Topic must be an integer >= -1')
        clauses.append({'topic_id': topic})
    if entity is not None:
        if entity not in ENTITY_TYPES:
            raise ValueError('Unsupported entity type')
        clauses.append({'has_' + entity.lower(): True})
    if min_weight is not None:
        if not 0 <= min_weight <= 1:
            raise ValueError('Topic weight must be 0..1')
        clauses.append({'topic_weight': {'$gte': min_weight}})
    return {'$and': clauses} if len(clauses) > 1 else clauses[0] if clauses else None


def evaluate_entities(nlp: object, examples: list[dict]) -> dict[str, object]:
    """Score exact entity spans and types against provenance-labelled examples."""
    tp = fp = fn = exact = 0
    details = []
    for example in examples:
        text = example['text']
        gold = set()
        for e in example['entities']:
            if e['label'] not in ENTITY_TYPES or not 0 <= e['start'] < e['end'] <= len(text):
                raise ValueError('Invalid annotation span or label')
            if text[e['start']:e['end']] != e['text']:
                raise ValueError('Annotation text does not match offsets')
            gold.add((e['start'], e['end'], e['label']))
        predicted = extract_entities(nlp, text)
        pred = {(e['start'], e['end'], e['label']) for e in predicted}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
        exact += gold == pred
        details.append({'doc_id': example['doc_id'], 'text': text, 'gold': example['entities'],
                        'predicted': predicted, 'exact_match': gold == pred})
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {'examples': len(examples), 'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'exact_example_accuracy': exact / len(examples) if examples else 0., 'details': details}


def fingerprint(records: list[dict]) -> str:
    """Return a deterministic digest used to protect NLP artifact reuse."""
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
