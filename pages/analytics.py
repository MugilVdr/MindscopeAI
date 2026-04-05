import streamlit as st
import pandas as pd

from database.db_service import get_emotion_counts, get_recent_predictions, get_trend_points


def analytics_page():

    st.title("Analytics Dashboard")

    user_id = st.session_state["user_id"]

    data = get_emotion_counts(user_id)
    recent_data = get_recent_predictions(user_id, limit=5)
    trend_data = get_trend_points(user_id, limit=30)

    if data:
        try:
            import plotly.express as px
        except ImportError:
            st.error("Plotly is not installed. Run `pip install -r requirements.txt` to enable charts.")
            return

        emotions = [d[0] for d in data]
        counts = [d[1] for d in data]

        fig = px.pie(
            names=emotions,
            values=counts,
            title="Emotion Distribution"
        )

        st.plotly_chart(fig)

        if trend_data:
            timeline_df = pd.DataFrame(
                trend_data,
                columns=[
                    "Date",
                    "Mental State",
                    "Text Confidence",
                    "Support Level",
                    "Urgency Score",
                    "Triage Level",
                ],
            )
            timeline_df["Date"] = pd.to_datetime(timeline_df["Date"])

            st.subheader("Trend Timeline")
            line_fig = px.line(
                timeline_df,
                x="Date",
                y="Urgency Score",
                color="Mental State",
                markers=True,
                title="Urgency and state trend"
            )
            st.plotly_chart(line_fig, use_container_width=True)

            support_fig = px.scatter(
                timeline_df,
                x="Date",
                y="Text Confidence",
                color="Support Level",
                symbol="Triage Level",
                title="Confidence vs support level"
            )
            st.plotly_chart(support_fig, use_container_width=True)

        if recent_data:
            st.subheader("Recent Confidence Snapshot")
            for mental_state, text_conf, face_emotion, face_conf, support_level, urgency_score, triage_level, created_at in recent_data:
                st.write(
                    f"{created_at}: {mental_state} ({(text_conf or 0) * 100:.1f}%), "
                    f"Face: {face_emotion or 'N/A'} ({(face_conf or 0) * 100:.1f}%), "
                    f"Support: {support_level or 'N/A'}, "
                    f"Urgency: {(urgency_score or 0) * 100:.1f}%, "
                    f"Triage: {triage_level or 'N/A'}"
                )

    else:
        st.warning("No data available yet")
