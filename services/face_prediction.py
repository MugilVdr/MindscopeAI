from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_emotion_model.keras"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FACE_CONFIDENCE_THRESHOLD = 0.60

_FACE_MODEL = None


def _load_model():
    global _FACE_MODEL
    from tensorflow.keras.models import load_model

    if _FACE_MODEL is None:
        _FACE_MODEL = load_model(MODEL_PATH)
    return _FACE_MODEL


def _prepare_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("L")
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((64, 64))
    image = Image.eval(image, lambda px: min(255, int(px * 1.15)))
    rgb_image = Image.merge("RGB", (image, image, image))
    array = np.asarray(rgb_image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict_face_emotion(image_bytes):
    if not image_bytes:
        return None

    model = _load_model()
    image = _prepare_image(image_bytes)
    probabilities = model.predict(image, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    top_predictions = sorted(
        [
            {
                "label": label.title(),
                "score": float(score),
            }
            for label, score in zip(CLASS_NAMES, probabilities)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )[:3]
    best_label = CLASS_NAMES[best_index].title()
    best_confidence = float(probabilities[best_index])
    final_label = best_label if best_confidence >= FACE_CONFIDENCE_THRESHOLD else "Uncertain"

    return {
        "label": final_label,
        "raw_label": best_label,
        "confidence": best_confidence,
        "is_uncertain": final_label == "Uncertain",
        "threshold": FACE_CONFIDENCE_THRESHOLD,
        "top_predictions": top_predictions,
        "probabilities": {
            label.title(): float(score)
            for label, score in zip(CLASS_NAMES, probabilities)
        },
    }
