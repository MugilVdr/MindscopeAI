import streamlit as st
import pandas as pd

from database.db_service import build_weekly_summary, get_user_history


def reports_page():

    st.title("Reports")
    st.caption("Reports combine raw check-ins with a short weekly interpretation for easier review.")

    user_id = st.session_state["user_id"]

    data = get_user_history(user_id)
    weekly_summary = build_weekly_summary(user_id, days=7)

    if data:
        if weekly_summary:
            st.subheader("Explainable Report Cards")
            card1, card2, card3, card4 = st.columns(4)
            card1.metric("Most Frequent State", weekly_summary["dominant_state"])
            card2.metric("Trend Direction", weekly_summary["trend_direction"])
            card3.metric("Highest Urgency", f"{weekly_summary['highest_urgency_score'] * 100:.1f}%")
            card4.metric("Uncertain Sessions", weekly_summary["uncertain_sessions"])

            st.write(
                f"The last 7 days show a {weekly_summary['trend_direction'].lower()} pattern. "
                f"The most frequent state was {weekly_summary['dominant_state']}, "
                f"and the highest urgency check-in was {weekly_summary['highest_urgency_state']} "
                f"on {weekly_summary['highest_urgency_date']}."
            )

            st.write(
                f"Text/face signals agreed in {weekly_summary['agreement_count']} sessions "
                f"and disagreed in {weekly_summary['disagreement_count']} sessions."
            )

        df = pd.DataFrame(data, columns=[
            "Text",
            "Face Emotion",
            "Mental State",
            "Insight",
            "Text Confidence",
            "Face Confidence",
            "Support Level",
            "Input Source",
            "Urgency Score",
            "Triage Level",
            "Triage Reason",
            "Date"
        ])

        st.dataframe(df)

        if weekly_summary:
            summary_df = pd.DataFrame([
                {"Metric": "Total Check-Ins (7d)", "Value": weekly_summary["total_checkins"]},
                {"Metric": "Dominant State", "Value": weekly_summary["dominant_state"]},
                {"Metric": "Trend Direction", "Value": weekly_summary["trend_direction"]},
                {"Metric": "Average Text Confidence", "Value": round(weekly_summary["average_text_confidence"], 4)},
                {"Metric": "Average Urgency", "Value": round(weekly_summary["average_urgency"], 4)},
                {"Metric": "Uncertain Sessions", "Value": weekly_summary["uncertain_sessions"]},
                {"Metric": "Agreement Count", "Value": weekly_summary["agreement_count"]},
                {"Metric": "Disagreement Count", "Value": weekly_summary["disagreement_count"]},
            ])
            st.subheader("Weekly Summary Table")
            st.dataframe(summary_df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "report.csv"
        )

        if weekly_summary:
            st.download_button(
                "Download Weekly Summary CSV",
                summary_df.to_csv(index=False),
                "weekly_summary.csv"
            )

    else:
        st.warning("No reports available")
