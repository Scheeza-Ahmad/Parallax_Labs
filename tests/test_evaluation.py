import pytest
from src.evaluation import evaluate_retrieval
from src.sentiment import analyze


def test_retrieval_metrics():
    cases = [{'question': 'atom', 'relevant_doc_ids': ['a', 'b']}]
    retrieve = lambda q, k: {'hits': [{'metadata': {'doc_id': 'a'}}, {'metadata': {'doc_id': 'c'}}]}
    result = evaluate_retrieval(cases, retrieve, k=2)
    assert result['mean_precision_at_k'] == .5
    assert result['mean_recall_at_k'] == .5


def test_missing_labels_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval([{'question': 'atom', 'relevant_doc_ids': []}], lambda q, k: {}, 2)


def test_sentiment_edge_cases():
    assert analyze(None)['label'] == 'unscorable'
    assert analyze('This is wonderful!')['label'] == 'positive'
    assert analyze('This is terrible!')['label'] == 'negative'
