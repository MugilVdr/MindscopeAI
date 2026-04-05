HIGH_RISK_PATTERNS = {
    "High": [
        "kill myself",
        "end my life",
        "want to die",
        "suicide",
        "hurt myself",
        "self harm",
    ],
    "Moderate": [
        "hopeless",
        "can't go on",
        "nobody cares",
        "empty inside",
        "worthless",
        "panic attack",
    ],
}


def assess_text_risk(text):
    lowered = (text or "").lower()
    matched_high = [phrase for phrase in HIGH_RISK_PATTERNS["High"] if phrase in lowered]
    matched_moderate = [phrase for phrase in HIGH_RISK_PATTERNS["Moderate"] if phrase in lowered]

    if matched_high:
        return {
            "triage_level": "High Alert",
            "urgency_score": 0.95,
            "triage_reason": ", ".join(matched_high[:3]),
            "guidance": [
                "Contact local emergency support or a crisis hotline immediately.",
                "Stay with a trusted person right now if possible.",
                "Do not rely only on this app for urgent help.",
            ],
        }

    if matched_moderate:
        return {
            "triage_level": "Needs Attention",
            "urgency_score": 0.65,
            "triage_reason": ", ".join(matched_moderate[:3]),
            "guidance": [
                "Reach out to a trusted person soon.",
                "Reduce isolation and take a short grounding break.",
                "Consider professional support if this feeling persists.",
            ],
        }

    return {
        "triage_level": "Routine",
        "urgency_score": 0.2,
        "triage_reason": "No immediate high-risk phrases detected",
        "guidance": [
            "Continue regular self-check-ins.",
            "Track changes over time instead of relying on one reading.",
        ],
    }
