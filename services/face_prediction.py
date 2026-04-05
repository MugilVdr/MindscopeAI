from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_emotion_model.keras"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_FACE_MODEL = None


def _load_model():
    global _FACE_MODEL
    from tensorflow.keras.models import load_model

    if _FACE_MODEL is None:
        _FACE_MODEL = load_model(MODEL_PATH)
    return _FACE_MODEL


def _prepare_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))
    array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict_face_emotion(image_bytes):
    if not image_bytes:
        return None

    model = _load_model()
    image = _prepare_image(image_bytes)
    probabilities = model.predict(image, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    return {
        "label": CLASS_NAMES[best_index].title(),
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            label.title(): float(score)
            for label, score in zip(CLASS_NAMES, probabilities)
        },
    }
