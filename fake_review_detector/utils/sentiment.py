"""
sentiment.py
-------------
Lightweight lexicon-based sentiment analyzer. Deliberately avoids
external downloads (e.g. NLTK's VADER corpus) so the project runs
fully offline. Swap in NLTK/VADER or a transformer model for
production-grade accuracy.
"""

import re

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "loved", "perfect",
    "awesome", "fantastic", "wonderful", "best", "happy", "satisfied",
    "reliable", "recommend", "comfortable", "beautiful", "impressive",
    "smooth", "fast", "durable", "affordable", "solid", "nice", "easy",
    "helpful", "quality", "worth", "pleased", "efficient", "sturdy",
}

NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "awful", "horrible", "poor", "hate",
    "hated", "disappointed", "disappointing", "broken", "defective",
    "useless", "slow", "cheap", "flimsy", "waste", "regret", "faulty",
    "junk", "scam", "problem", "issue", "annoying", "uncomfortable",
    "late", "damaged", "refund", "return", "fail", "failed", "failure",
}


def analyze_sentiment(text: str):
    """Returns ('Positive'|'Negative'|'Neutral', score) where score is
    positive_count - negative_count normalized to [-1, 1]."""
    if not text:
        return "Neutral", 0.0

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "Neutral", 0.0

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)

    if pos == 0 and neg == 0:
        return "Neutral", 0.0

    score = (pos - neg) / max(1, (pos + neg))

    if score > 0.15:
        return "Positive", round(score, 2)
    elif score < -0.15:
        return "Negative", round(score, 2)
    return "Neutral", round(score, 2)


def analyze_batch(texts):
    return [analyze_sentiment(t) for t in texts]
