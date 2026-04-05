import pickle
import re
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "text_mental_model.keras"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

_TEXT_MODEL = None
_VECTORIZER = None
_ENCODER = None


def _clean_text(text):
    text = (text or "").strip().lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_assets():
    global _TEXT_MODEL, _VECTORIZER, _ENCODER
    from tensorflow.keras.models import load_model

    if _TEXT_MODEL is None:
        _TEXT_MODEL = load_model(MODEL_PATH)

    if _VECTORIZER is None:
        with open(VECTORIZER_PATH, "rb") as file:
            _VECTORIZER = pickle.load(file)

    if _ENCODER is None:
        with open(ENCODER_PATH, "rb") as file:
            _ENCODER = pickle.load(file)

    return _TEXT_MODEL, _VECTORIZER, _ENCODER


def predict_text_mental_state(text):
    cleaned_text = _clean_text(text)
    if not cleaned_text:
        raise ValueError("Please enter some text before analysis.")

    model, vectorizer, encoder = _load_assets()
    features = vectorizer.transform([cleaned_text]).toarray()
    probabilities = model.predict(features, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    return {
        "label": encoder.inverse_transform([best_index])[0],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            label: float(score)
            for label, score in zip(encoder.classes_, probabilities)
        },
        "cleaned_text": cleaned_text,
    }
