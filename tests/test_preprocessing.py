import pytest
from src.preprocessing import clean_text, normalize_unicode, normalize_whitespace, detect_language, is_target_language


def test_html_readability():
    assert clean_text('<p>Hello <b>world</b> &amp; friends.</p><p>Next.</p><script>bad()</script>') == 'Hello world & friends.\n\nNext.'


def test_malformed_html():
    assert clean_text('<p>Hello <b>world') == 'Hello world'


def test_unicode():
    assert normalize_unicode('Cafe\u0301') == 'Café'
    assert clean_text('Case: A + B = C!') == 'Case: A + B = C!'


def test_comparisons_and_inline_markup():
    assert clean_text('A < 5 and B > 2. <em>Keep</em> punctuation!') == 'A < 5 and B > 2. Keep punctuation!'


def test_cleaning_is_idempotent():
    text = clean_text('<p>Cafe\u0301 &amp; tea.</p><p>  Next\t paragraph! </p>')
    assert clean_text(text) == text


def test_whitespace():
    assert normalize_whitespace('  a\t b\r\n\r\n\r\n c  ') == 'a b\n\nc'


@pytest.mark.parametrize('value', [None, '', ' \t\n ', 12, {}, float('nan')])
def test_missing(value):
    assert clean_text(value) == ''


def test_language_filter():
    assert is_target_language('The ocean covers most of the surface of our planet. Many different animals live in the water and depend on this environment for food.')
    assert not is_target_language('La langue française est une langue parlée dans plusieurs pays du monde. Les enfants apprennent cette langue dans les écoles de leur région.')


def test_uncertain_short_language():
    assert detect_language('Hello!') == ('und', 0.0)
    assert not is_target_language('12345 ' * 30)
