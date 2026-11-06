"""
reviewer_analysis.py
----------------------
Heuristic-based "fake reviewer" risk scoring. Looks at behavior patterns
across a dataset (not just single-review text) to flag reviewers who are
statistically unusual: posting many reviews, near-identical text, or
extreme ratings.

Expects a pandas DataFrame with at least a 'review_text' column, and
optionally 'reviewer_name' and 'rating'.
"""

import pandas as pd


def analyze_reviewers(df: pd.DataFrame):
    """Returns a DataFrame: reviewer_name, review_count, avg_rating,
    duplicate_ratio, risk_level."""
    if "reviewer_name" not in df.columns:
        return pd.DataFrame(columns=[
            "reviewer_name", "review_count", "avg_rating",
            "duplicate_ratio", "risk_level"
        ])

    work = df.copy()
    work["reviewer_name"] = work["reviewer_name"].fillna("unknown")
    work["review_text"] = work["review_text"].fillna("")

    grouped = []
    for name, sub in work.groupby("reviewer_name"):
        count = len(sub)
        avg_rating = sub["rating"].mean() if "rating" in sub.columns else None
        # crude duplicate ratio: fraction of reviews under 8 words (low effort)
        short_reviews = (sub["review_text"].str.split().str.len() < 8).sum()
        duplicate_ratio = round(short_reviews / count, 2) if count else 0

        risk_score = 0
        if count >= 5:
            risk_score += 2
        elif count >= 3:
            risk_score += 1
        if avg_rating is not None and (avg_rating >= 4.8 or avg_rating <= 1.2):
            risk_score += 1
        if duplicate_ratio >= 0.5:
            risk_score += 2
        elif duplicate_ratio >= 0.25:
            risk_score += 1

        if risk_score >= 4:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        grouped.append({
            "reviewer_name": name,
            "review_count": count,
            "avg_rating": round(avg_rating, 2) if avg_rating is not None else None,
            "duplicate_ratio": duplicate_ratio,
            "risk_level": risk_level,
        })

    result = pd.DataFrame(grouped).sort_values(
        by="review_count", ascending=False
    ).reset_index(drop=True)
    return result
