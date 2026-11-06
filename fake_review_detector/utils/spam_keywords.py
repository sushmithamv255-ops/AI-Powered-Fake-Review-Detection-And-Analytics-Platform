"""
spam_keywords.py
------------------
Flags suspicious promotional / spammy language in a review and computes
a 0-100 spam score based on keyword hits, excessive punctuation, and
ALL-CAPS shouting.
"""

import re

SPAM_KEYWORDS = [
    "buy now", "click here", "free gift", "guaranteed", "act now",
    "limited time", "don't miss", "best deal", "discount code",
    "visit our website", "check out my", "follow me", "subscribe",
    "100% real", "money back", "as seen on", "promo code", "cheap price",
    "order now", "call now", "winner", "congratulations you",
]


def analyze_spam(text: str):
    if not text:
        return {"spam_score": 0, "matched_keywords": [], "flags": []}

    lower = text.lower()
    matched = [kw for kw in SPAM_KEYWORDS if kw in lower]

    exclamations = text.count("!")
    caps_words = re.findall(r"\b[A-Z]{3,}\b", text)
    repeated_chars = bool(re.search(r"(.)\1{3,}", text))  # e.g. "sooooo"

    score = 0
    score += len(matched) * 20
    score += min(exclamations * 5, 30)
    score += min(len(caps_words) * 5, 20)
    score += 10 if repeated_chars else 0
    score = min(score, 100)

    flags = []
    if matched:
        flags.append(f"{len(matched)} promotional keyword(s) found")
    if exclamations >= 3:
        flags.append("Excessive exclamation marks")
    if len(caps_words) >= 2:
        flags.append("Excessive ALL-CAPS words")
    if repeated_chars:
        flags.append("Repeated character stretching (e.g. 'soooo')")

    return {
        "spam_score": score,
        "matched_keywords": matched,
        "flags": flags,
    }


def analyze_batch(texts):
    return [analyze_spam(t) for t in texts]
