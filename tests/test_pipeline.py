import json
import pytest
from scripts.run_preprocessing import run

ENGLISH = 'The ocean covers most of the surface of our planet. Many different animals live in the water and depend on this environment for food.'


def test_pipeline_counts_and_roundtrip(tmp_path):
    source = tmp_path / 'raw.jsonl'
    output = tmp_path / 'clean.jsonl'
    report = tmp_path / 'report.json'
    rows = [{'id': str(i), 'url': 'https://example.org/article', 'title': 'Ocean', 'text': text}
            for i, text in enumerate([ENGLISH, ENGLISH, None, 'Hello'])]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows) + 'bad json\n[]\n', encoding='utf-8')
    original = source.read_bytes()
    run(source, output, report, 1)
    result = json.loads(report.read_text())
    assert result['counts'] == {'input': 6, 'accepted': 1, 'empty_text': 1, 'duplicate': 1,
                                'invalid_schema': 1, 'malformed_json': 1, 'uncertain_language': 1, 'non_english': 0}
    assert json.loads(output.read_text())['text'] == ENGLISH
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        run(source, output, report, 1)


def test_minimum_is_enforced(tmp_path):
    source = tmp_path / 'raw.jsonl'
    source.write_text(json.dumps({'id': '1', 'url': 'https://example.org', 'title': 'Ocean', 'text': ENGLISH}) + '\n')
    output = tmp_path / 'clean.jsonl'
    with pytest.raises(ValueError, match='Fewer than 5000'):
        run(source, output, tmp_path / 'report.json', 5000)
    assert not output.exists()
