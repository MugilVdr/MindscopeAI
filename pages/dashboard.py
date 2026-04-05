import streamlit as st

from database.db_service import get_emotion_counts, get_latest_emotion


def dashboard_page():
    st.title("Dashboard")

    user_id = st.session_state["user_id"]
    data = get_emotion_counts(user_id)

    total = sum([d[1] for d in data]) if data else 0
    positive = next((d[1] for d in data if d[0] == "Positive"), 0)
    stress = next((d[1] for d in data if d[0] == "Stress"), 0)
    last_emotion = get_latest_emotion(user_id) or "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses", total)
    col2.metric("Positive", positive)
    col3.metric("Stress", stress)
    col4.metric("Last Emotion", last_emotion)

    st.info("MindScope AI is ready.")
