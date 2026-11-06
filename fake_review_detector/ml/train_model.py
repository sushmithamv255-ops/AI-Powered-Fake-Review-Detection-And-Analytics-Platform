import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sample_reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fake_review_model.joblib")


def train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review_text", "label"])

    X = df["review_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3000,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=2.0, solver="liblinear")),
    ])

    pipeline.fit(X_train, y_train)

    # ensure the saved object works with the current scikit-learn runtime
    _ = pipeline.predict_proba(["sample review text"])

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.2%}")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # store accuracy so the dashboard can display it
    with open(os.path.join(BASE_DIR, "model_accuracy.txt"), "w") as f:
        f.write(str(round(acc * 100, 2)))


if __name__ == "__main__":
    train()
