import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from src.chunking import chunk_document


@pytest.fixture
def tokenizer():
    raw = Tokenizer(models.WordLevel({'[UNK]': 0, 'one': 1, 'two': 2, 'three': 3}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]')


def record(text):
    return dict(doc_id='wiki:1', text=text, source='Wikipedia', source_url='https://example.org/1',
                title='Test', language='en', language_confidence=0.99)


@pytest.mark.parametrize('text', ['', '  \n'])
def test_empty(tokenizer, text):
    assert chunk_document(record(text), tokenizer, 4, 1) == []


def test_short_and_metadata(tokenizer):
    row = record('Café 日本語 🙂 one')
    chunks = chunk_document(row, tokenizer, 5, 1)
    assert len(chunks) == 1 and chunks[0].text == row['text']
    assert all(chunks[0].metadata[k] == v for k, v in row.items() if k != 'text')
    assert chunks == chunk_document(row, tokenizer, 5, 1)
    assert chunks[0].id != chunk_document(dict(row, doc_id='other'), tokenizer, 5, 1)[0].id


@pytest.mark.parametrize('overlap', [0, 1, 3])
def test_boundaries_coverage_overlap(tokenizer, overlap):
    text = '  one two three\n\nCafé 日本語 🙂 one two three end  '
    chunks = chunk_document(record(text), tokenizer, 4, overlap)
    assert len(chunks) > 1
    assert chunks[0].metadata['char_start'] == 0
    assert chunks[-1].metadata['char_end'] == len(text)
    covered = set()
    for i, chunk in enumerate(chunks):
        m = chunk.metadata
        assert chunk.text == text[m['char_start']:m['char_end']]
        assert m['token_count'] <= 4 and m['chunk_index'] == i
        covered.update(range(m['char_start'], m['char_end']))
        if i:
            assert chunks[i-1].metadata['token_end'] - m['token_start'] == overlap
    assert covered == set(range(len(text)))
    assert len({c.id for c in chunks}) == len(chunks)


@pytest.mark.parametrize('size,overlap', [(0, 0), (4, -1), (4, 4), (4, 5), (1.5, 0), (True, 0), (300, 1)])
def test_invalid_settings(tokenizer, size, overlap):
    with pytest.raises(ValueError):
        chunk_document(record('one two'), tokenizer, size, overlap)


@pytest.mark.parametrize('change', [{'text': None}, {'doc_id': ''}, {'extra': []}, {'extra': float('nan')}, {'chunk_index': 7}])
def test_invalid_record(tokenizer, change):
    with pytest.raises(ValueError):
        chunk_document(dict(record('one two'), **change), tokenizer)


def test_special_tokens_consume_model_budget():
    raw = Tokenizer(models.WordPiece({'[UNK]': 0, '[CLS]': 1, '[SEP]': 2,
                                     'one': 3, 'two': 4, 'three': 5}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    raw.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]')
    assert tokenizer.num_special_tokens_to_add(pair=False) == 2
    chunks = chunk_document(record('one two three one two three'), tokenizer, 3, 1, 5)
    assert all(len(tokenizer(c.text)['input_ids']) <= 5 for c in chunks)
    with pytest.raises(ValueError, match='special tokens'):
        chunk_document(record('one'), tokenizer, 4, 1, 5)
