import streamlit as st

from database.db_service import get_emotion_counts, get_latest_emotion, get_trend_points


STATE_SCORES = {
    "Positive": 5,
    "Neutral": 4,
    "Stress": 3,
    "Anxiety": 2,
    "Depression": 1,
}


def dashboard_page():
    st.title("Check-In Dashboard")
    st.caption("This dashboard summarizes past AI-assisted check-ins. Treat it as a trend view, not a diagnosis.")

    user_id = st.session_state["user_id"]
    data = get_emotion_counts(user_id)
    trend_points = get_trend_points(user_id, limit=10)

    total = sum([d[1] for d in data]) if data else 0
    positive = next((d[1] for d in data if d[0] == "Positive"), 0)
    stress = next((d[1] for d in data if d[0] == "Stress"), 0)
    last_emotion = get_latest_emotion(user_id) or "N/A"
    avg_stability = 0
    high_alerts = 0

    if trend_points:
        scores = [STATE_SCORES.get(row[1], 3) for row in trend_points]
        avg_stability = sum(scores) / len(scores)
        high_alerts = sum(1 for row in trend_points if row[5] == "High Alert")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses", total)
    col2.metric("Positive", positive)
    col3.metric("Stress", stress)
    col4.metric("Last Emotion", last_emotion)

    col5, col6 = st.columns(2)
    col5.metric("Stability Score", f"{avg_stability:.1f}/5" if trend_points else "N/A")
    col6.metric("High Alerts", high_alerts)

    if trend_points:
        st.subheader("Recent Trend")
        for created_at, mental_state, text_conf, support_level, urgency_score, triage_level in trend_points[-5:]:
            st.write(
                f"{created_at}: {mental_state}, support {support_level or 'N/A'}, "
                f"urgency {(urgency_score or 0) * 100:.1f}%, triage {triage_level or 'N/A'}"
            )
    else:
        st.info("No check-ins yet. Start with the Emotional Check-In page.")
