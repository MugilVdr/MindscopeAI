import streamlit as st
import pandas as pd
import json

from database.db_service import (
    build_weekly_summary,
    get_emotion_counts,
    get_model_diagnostics,
    get_recent_predictions,
    get_trend_points,
)


def analytics_page():

    st.title("Analytics Dashboard")
    st.caption("Use these views to spot trends and uncertainty over time, not to draw clinical conclusions.")

    user_id = st.session_state["user_id"]

    data = get_emotion_counts(user_id)
    recent_data = get_recent_predictions(user_id, limit=5)
    trend_data = get_trend_points(user_id, limit=30)
    weekly_summary = build_weekly_summary(user_id, days=7)
    diagnostics = get_model_diagnostics(user_id, limit=10)

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

        if weekly_summary:
            st.subheader("Weekly Summary")
            card1, card2, card3, card4 = st.columns(4)
            card1.metric("Check-Ins", weekly_summary["total_checkins"])
            card2.metric("Dominant State", weekly_summary["dominant_state"])
            card3.metric("Trend", weekly_summary["trend_direction"])
            card4.metric("Uncertain Sessions", weekly_summary["uncertain_sessions"])

            card5, card6, card7, card8 = st.columns(4)
            card5.metric("Avg Text Confidence", f"{weekly_summary['average_text_confidence'] * 100:.1f}%")
            card6.metric("Avg Urgency", f"{weekly_summary['average_urgency'] * 100:.1f}%")
            card7.metric("Dominant Support", weekly_summary["dominant_support"])
            card8.metric("Common Triage", weekly_summary["most_common_triage"])

            st.info(
                f"Weekly interpretation: {weekly_summary['trend_direction']} trend, "
                f"dominant state {weekly_summary['dominant_state']}, "
                f"highest urgency session {weekly_summary['highest_urgency_state']} "
                f"at {weekly_summary['highest_urgency_score'] * 100:.1f}% on {weekly_summary['highest_urgency_date']}."
            )

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

        if weekly_summary:
            st.subheader("Signal Agreement")
            st.write(
                f"Text/face agreement sessions: {weekly_summary['agreement_count']} | "
                f"Disagreement sessions: {weekly_summary['disagreement_count']} | "
                f"Raw label mismatches: {weekly_summary['raw_label_mismatch_count']}"
            )

        if diagnostics:
            st.subheader("Model Diagnostics")
            diagnostic_rows = []
            for created_at, mental_state, face_emotion, text_conf, face_conf, text_raw, face_raw, text_top, face_top in diagnostics:
                text_top_list = json.loads(text_top or "[]")
                face_top_list = json.loads(face_top or "[]")
                diagnostic_rows.append({
                    "Date": created_at,
                    "Final State": mental_state,
                    "Face Signal": face_emotion,
                    "Text Confidence": round(text_conf or 0, 4),
                    "Face Confidence": round(face_conf or 0, 4),
                    "Text Raw Label": text_raw,
                    "Face Raw Label": face_raw,
                    "Text Top 3": ", ".join(
                        f"{item['label']} {item['score'] * 100:.1f}%"
                        for item in text_top_list
                    ),
                    "Face Top 3": ", ".join(
                        f"{item['label']} {item['score'] * 100:.1f}%"
                        for item in face_top_list
                    ),
                })

            diagnostics_df = pd.DataFrame(diagnostic_rows)
            st.dataframe(diagnostics_df, use_container_width=True)

    else:
        st.warning("No data available yet")
