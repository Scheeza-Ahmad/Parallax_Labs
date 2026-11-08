"""Retrieval precision/recall over explicitly labelled relevant document IDs."""


def evaluate_retrieval(cases, retrieve, k=5):
    if not cases:
        raise ValueError('At least one labelled query is required')
    results = []
    for case in cases:
        if not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError('Each case needs a question')
        relevant = case.get('relevant_doc_ids')
        if not isinstance(relevant, list) or not relevant or any(not isinstance(x, str) or not x for x in relevant):
            raise ValueError('Each case needs reviewed relevant_doc_ids')
        if len(set(relevant)) != len(relevant):
            raise ValueError('Duplicate reference IDs')
        hits = retrieve(case['question'], k)['hits']
        found = list(dict.fromkeys(h['metadata']['doc_id'] for h in hits))
        tp = len(set(found) & set(relevant))
        results.append({'question': case['question'], 'retrieved_doc_ids': found,
                        'relevant_doc_ids': relevant, 'precision_at_k': tp / k,
                        'recall_at_k': tp / len(relevant)})
    return {'queries': len(results), 'k': k,
            'mean_precision_at_k': sum(r['precision_at_k'] for r in results) / len(results),
            'mean_recall_at_k': sum(r['recall_at_k'] for r in results) / len(results),
            'results': results}
