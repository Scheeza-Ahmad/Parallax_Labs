"""Exploratory corpus sentiment, with explicit neutral and empty cases."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def analyze(text, analyzer=None):
    if not isinstance(text, str) or not text.strip():
        return {'label': 'unscorable', 'compound': None}
    analyzer = analyzer or SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text[:10000])['compound']
    return {'label': 'positive' if score >= .05 else 'negative' if score <= -.05 else 'neutral',
            'compound': score}
