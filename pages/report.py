import streamlit as st
import pandas as pd
import json

from database.db_service import build_weekly_summary, get_model_diagnostics, get_user_history
from services.pdf_report import build_pdf_report


def reports_page():

    st.title("Reports")
    st.caption("Reports combine raw check-ins with a short weekly interpretation for easier review.")

    user_id = st.session_state["user_id"]

    data = get_user_history(user_id)
    weekly_summary = build_weekly_summary(user_id, days=7)
    diagnostics = get_model_diagnostics(user_id, limit=10)

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
            "Text Raw Label",
            "Face Raw Label",
            "Text Top Predictions",
            "Face Top Predictions",
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

        if diagnostics:
            diagnostics_rows = []
            for created_at, mental_state, face_emotion, text_conf, face_conf, text_raw, face_raw, text_top, face_top in diagnostics:
                diagnostics_rows.append({
                    "Date": created_at,
                    "Final State": mental_state,
                    "Face Signal": face_emotion,
                    "Text Raw Label": text_raw,
                    "Face Raw Label": face_raw,
                    "Text Top 3": ", ".join(
                        f"{item['label']} {item['score'] * 100:.1f}%"
                        for item in json.loads(text_top or "[]")
                    ),
                    "Face Top 3": ", ".join(
                        f"{item['label']} {item['score'] * 100:.1f}%"
                        for item in json.loads(face_top or "[]")
                    ),
                    "Text Confidence": round(text_conf or 0, 4),
                    "Face Confidence": round(face_conf or 0, 4),
                })

            diagnostics_df = pd.DataFrame(diagnostics_rows)
            st.subheader("Model Diagnostics View")
            st.dataframe(diagnostics_df, use_container_width=True)
            st.download_button(
                "Download Diagnostics CSV",
                diagnostics_df.to_csv(index=False),
                "model_diagnostics.csv"
            )

        try:
            pdf_bytes = build_pdf_report(
                st.session_state.get("user_name", "User"),
                weekly_summary,
                diagnostics,
            )
            st.download_button(
                "Download PDF Report",
                pdf_bytes,
                "mindscope_report.pdf",
                mime="application/pdf"
            )
        except ImportError:
            st.info("PDF export requires `reportlab`. Install dependencies again after the update to enable it.")

    else:
        st.warning("No reports available")
