
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_review_model.joblib")
ACCURACY_PATH = os.path.join(BASE_DIR, "model_accuracy.txt")

_model = None


def _rebuild_model_if_needed():
    global _model
    from ml import train_model

    if not os.path.exists(MODEL_PATH):
        train_model.train()
        return

    try:
        test_model = joblib.load(MODEL_PATH)
        if hasattr(test_model, "predict_proba"):
            test_model.predict_proba(["sample text"])
            _model = test_model
            return
        raise AttributeError("Saved model does not support probability prediction.")
    except Exception:
        train_model.train()
        _model = joblib.load(MODEL_PATH)


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run `python ml/train_model.py` first."
            )
        try:
            loaded = joblib.load(MODEL_PATH)
            if hasattr(loaded, "predict_proba"):
                loaded.predict_proba(["sample text"])
                _model = loaded
            else:
                raise AttributeError("Saved model does not expose predict_proba.")
        except Exception:
            _rebuild_model_if_needed()
    return _model


def get_model_accuracy():
    if os.path.exists(ACCURACY_PATH):
        with open(ACCURACY_PATH) as f:
            return float(f.read().strip())
    return None


def predict_review(text: str):
    """Returns (label, confidence_percent) for a single review string."""
    model = _load_model()
    text = (text or "").strip()
    if not text:
        return "Genuine", 0.0

    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    confidence = round(float(proba[classes.index(label)]) * 100, 2)
    return label, confidence


def predict_batch(texts):
    """Returns list of (label, confidence_percent) for a list of review strings."""
    model = _load_model()
    texts = [(t or "").strip() for t in texts]
    labels = model.predict(texts)
    probas = model.predict_proba(texts)
    classes = list(model.classes_)
    results = []
    for label, proba in zip(labels, probas):
        confidence = round(float(proba[classes.index(label)]) * 100, 2)
        results.append((label, confidence))
    return results
