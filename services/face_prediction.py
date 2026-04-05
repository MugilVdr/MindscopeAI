from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_emotion_model.keras"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FACE_CONFIDENCE_THRESHOLD = 0.60

_FACE_MODEL = None
_FACE_CASCADE = None


def _load_model():
    global _FACE_MODEL
    from tensorflow.keras.models import load_model

    if _FACE_MODEL is None:
        _FACE_MODEL = load_model(MODEL_PATH)
    return _FACE_MODEL


def _load_face_cascade():
    global _FACE_CASCADE
    import cv2

    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _detect_face(gray_image):
    import cv2

    cascade = _load_face_cascade()
    image_array = np.asarray(gray_image)
    faces = cascade.detectMultiScale(
        image_array,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    padding = int(min(w, h) * 0.15)
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, gray_image.size[0])
    y2 = min(y + h + padding, gray_image.size[1])
    return gray_image.crop((x1, y1, x2, y2))


def _prepare_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("L")
    face_crop = _detect_face(image)
    detection_note = "Face detected and cropped." if face_crop is not None else "No clear face detected. Using center crop fallback."

    if face_crop is None:
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
    else:
        image = face_crop

    image = image.resize((64, 64))
    image = Image.eval(image, lambda px: min(255, int(px * 1.15)))
    rgb_image = Image.merge("RGB", (image, image, image))
    array = np.asarray(rgb_image, dtype="float32") / 255.0
    return np.expand_dims(array, axis=0), face_crop is not None, detection_note


def predict_face_emotion(image_bytes):
    if not image_bytes:
        return None

    model = _load_model()
    image, face_detected, detection_note = _prepare_image(image_bytes)
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
        "face_detected": face_detected,
        "detection_note": detection_note,
        "probabilities": {
            label.title(): float(score)
            for label, score in zip(CLASS_NAMES, probabilities)
        },
    }
