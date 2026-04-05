def _build_insight(final_state, text_result, face_result):
    text_label = text_result["raw_label"]
    text_conf = round(text_result["confidence"] * 100, 1)

    if face_result:
        face_label = face_result["raw_label"]
        face_conf = round(face_result["confidence"] * 100, 1)
        return (
            f"Text suggests {text_label} ({text_conf}%) and face suggests "
            f"{face_label} ({face_conf}%). Final state: {final_state}."
        )

    return f"Text suggests {text_label} ({text_conf}%). Final state: {final_state}."


def _build_recommendations(final_state):
    recommendations = {
        "Stress": [
            "Take a short break before the next task.",
            "Try a two-minute breathing reset.",
            "Reduce one avoidable stressor today.",
        ],
        "Anxiety": [
            "Ground yourself with a short breathing exercise.",
            "Write down the immediate concern and next action.",
            "Reach out to someone you trust if the feeling continues.",
        ],
        "Depression": [
            "Keep the next task very small and concrete.",
            "Check in with a trusted person today.",
            "Seek professional support if these feelings persist.",
        ],
        "Positive": [
            "Keep the routine that is supporting your mood.",
            "Capture what is going well right now.",
            "Use this window for a focused task.",
        ],
        "Neutral": [
            "Monitor how you feel through the day.",
            "Keep a balanced routine with breaks.",
            "Log another check-in later if needed.",
        ],
    }
    return recommendations.get(final_state, recommendations["Neutral"])


def _support_level(final_state, text_confidence):
    if final_state == "Uncertain":
        return "Review"
    if final_state in {"Depression", "Anxiety"} and text_confidence >= 0.8:
        return "High"
    if final_state == "Stress" and text_confidence >= 0.7:
        return "Moderate"
    if final_state == "Positive":
        return "Low"
    return "Watch"


def combine_predictions(text_result, face_result=None):
    final_state = text_result["label"]
    face_emotion = face_result["label"] if face_result else "Not Provided"

    if face_result and not face_result["is_uncertain"] and not text_result["is_uncertain"]:
        face_label = face_result["label"]
        if final_state == "Positive" and face_label in {"Sad", "Angry", "Fear"}:
            final_state = "Neutral"
    elif face_result and face_result["is_uncertain"]:
        face_emotion = "Uncertain"

    return {
        "mental_state": final_state,
        "face_emotion": face_emotion,
        "insight": _build_insight(final_state, text_result, face_result),
        "recommendations": _build_recommendations(final_state),
        "text_confidence": text_result["confidence"],
        "face_confidence": face_result["confidence"] if face_result else None,
        "support_level": _support_level(final_state, text_result["confidence"]),
        "input_source": "Text + Face" if face_result else "Text Only",
    }
