"""
duplicates.py
--------------
Finds exact and near-duplicate reviews in a dataset using TF-IDF
cosine similarity. Two reviews with similarity above SIMILARITY_THRESHOLD
are flagged as duplicates.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
SIMILARITY_THRESHOLD = 0.85
def find_duplicates(texts):
    """
    texts: list[str]
    Returns:
        duplicate_flags: list[bool] -- True if this review is a duplicate
                          of an earlier one in the list
        duplicate_pairs: list[(i, j, similarity)] for reporting
        duplicate_percentage: float
    """
    n = len(texts)
    if n < 2:
        return [False] * n, [], 0.0

    clean_texts = [t if t and t.strip() else " " for t in texts]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return [False] * n, [], 0.0

    sim_matrix = cosine_similarity(tfidf_matrix)

    duplicate_flags = [False] * n
    duplicate_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i][j]
            if sim >= SIMILARITY_THRESHOLD:
                duplicate_flags[j] = True
                duplicate_pairs.append((i, j, round(float(sim), 3)))

    duplicate_percentage = round(100 * sum(duplicate_flags) / n, 2)
    return duplicate_flags, duplicate_pairs, duplicate_percentage
