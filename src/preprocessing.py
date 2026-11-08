"""Conservative text cleaning; deterministic English language filtering."""
import re
import unicodedata
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0


def strip_html(text: str) -> str:
    """Remove markup and script/style content while preserving block breaks."""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["p", "div", "br", "li", "h1", "h2", "h3", "tr"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    return soup.get_text()


def normalize_unicode(text: str) -> str:
    """Compose equivalent Unicode sequences into NFC form."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse inline spaces while retaining paragraph separation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_text(value: object) -> str:
    """Apply safe cleaning; missing or non-string values become empty text."""
    if not isinstance(value, str):
        return ""
    return normalize_whitespace(normalize_unicode(strip_html(value)))


def detect_language(text: str) -> tuple[str, float]:
    """Return deterministic language and confidence, or und for uncertain text."""
    # Short samples are unreliable even when a detector returns high confidence.
    if sum(c.isalpha() for c in text) < 50:
        return "und", 0.0
    try:
        result = detect_langs(text[:10000])[0]
        return (result.lang if result.prob >= 0.90 else "und", result.prob)
    except LangDetectException:
        return "und", 0.0


def is_target_language(text: str, target: str = "en") -> bool:
    """Decide whether the detected language matches the requested language."""
    return detect_language(text)[0] == target
