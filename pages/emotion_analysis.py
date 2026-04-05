import streamlit as st

from database.db_service import save_prediction
from services.face_prediction import predict_face_emotion
from services.fusion_engine import combine_predictions
from services.safety_guard import assess_text_risk
from services.text_prediction import predict_text_mental_state


def _confidence_caption(value):
    return f"{value * 100:.1f}%" if value is not None else "Not available"


def _render_score(label, value):
    st.caption(f"{label}: {_confidence_caption(value)}")
    if value is not None:
        st.progress(min(max(float(value), 0.0), 1.0))


def _render_top_predictions(items):
    for item in items:
        st.write(f"{item['label']}: {item['score'] * 100:.1f}%")


def emotion_analysis_page():
    st.title("Emotional Check-In")
    st.write(
        "Use this page for a lightweight AI-assisted check-in. Results are signals, not a diagnosis or clinical assessment."
    )
    st.caption(
        "Best results come from clear text, even lighting, and a front-facing image with one visible face."
    )

    text = st.text_area(
        "What would you like to check in about?",
        placeholder="Describe how you feel, what happened today, or what is on your mind."
    )

    capture_tab, upload_tab = st.tabs(["Camera", "Upload"])
    captured_image = None
    uploaded_image = None

    with capture_tab:
        captured_image = st.camera_input("Capture a front-facing image")

    with upload_tab:
        uploaded_image = st.file_uploader(
            "Upload a clear face image",
            type=["jpg", "jpeg", "png"],
            key="face_upload"
        )

    selected_image = captured_image or uploaded_image

    if st.button("Analyze Emotion", type="primary"):
        try:
            text_result = predict_text_mental_state(text)
            risk_result = assess_text_risk(text_result["cleaned_text"])
            face_result = None

            if selected_image is not None:
                face_result = predict_face_emotion(selected_image.getvalue())

            final_result = combine_predictions(text_result, face_result)
            final_result["triage_level"] = risk_result["triage_level"]
            final_result["urgency_score"] = max(
                final_result["text_confidence"],
                risk_result["urgency_score"]
            )
            final_result["triage_reason"] = risk_result["triage_reason"]

            summary_col, support_col, source_col, urgency_col = st.columns(4)
            summary_col.metric("AI Check-In Result", final_result["mental_state"])
            support_col.metric("Suggested Support Level", final_result["support_level"])
            source_col.metric("Input Source", final_result["input_source"])
            urgency_col.metric("Safety Triage", final_result["triage_level"])

            if final_result["triage_level"] == "High Alert":
                st.error(
                    "High-risk language detected. This app is not emergency support. "
                    "Please contact local emergency services or a crisis hotline now."
                )
            elif final_result["triage_level"] == "Needs Attention":
                st.warning(
                    "This check-in suggests elevated distress. Consider contacting a trusted person or professional."
                )
            else:
                st.success("No immediate high-risk language was detected in this check-in.")

            st.subheader("Confidence Breakdown")
            conf_col1, conf_col2 = st.columns(2)
            with conf_col1:
                st.markdown("**Text Model**")
                _render_score("Mental state confidence", final_result["text_confidence"])
                st.caption(
                    f"Threshold: {text_result['threshold'] * 100:.0f}% "
                    f"| Raw top class: {text_result['raw_label']}"
                )
                _render_top_predictions(text_result["top_predictions"])

            with conf_col2:
                st.markdown("**Face Model**")
                _render_score("Face emotion confidence", final_result["face_confidence"])
                if face_result:
                    st.caption(face_result["detection_note"])
                    st.caption(
                        f"Threshold: {face_result['threshold'] * 100:.0f}% "
                        f"| Raw top class: {face_result['raw_label']}"
                    )
                    _render_top_predictions(face_result["top_predictions"])
                    if not face_result["face_detected"]:
                        st.warning(
                            "No clear face was detected. The model used a fallback crop, so this result is less reliable."
                        )
                else:
                    st.info("No face image provided for this analysis.")

            st.subheader("Result Summary")
            result_col1, result_col2 = st.columns(2)
            result_col1.success(
                f"Text Check-In Signal: {final_result['mental_state']} "
                f"({_confidence_caption(final_result['text_confidence'])})"
            )
            result_col2.warning(
                f"Face Signal: {final_result['face_emotion']} "
                f"({_confidence_caption(final_result['face_confidence'])})"
            )

            if text_result["is_uncertain"] or (face_result and face_result["is_uncertain"]):
                st.warning(
                    "One or more outputs are below the confidence threshold, so the app is marking them as Uncertain instead of overstating the result."
                )

            st.subheader("Model Summary")
            st.info(final_result["insight"])

            st.subheader("Triage Detail")
            st.caption(f"Reason: {final_result['triage_reason']}")
            st.progress(min(max(float(final_result["urgency_score"]), 0.0), 1.0))
            st.caption(f"Urgency score: {final_result['urgency_score'] * 100:.1f}%")

            st.subheader("Suggested Next Steps")
            for recommendation in final_result["recommendations"]:
                st.write(f"- {recommendation}")

            st.subheader("Safety Guidance")
            for line in risk_result["guidance"]:
                st.write(f"- {line}")

            save_prediction(
                st.session_state["user_id"],
                text_result["cleaned_text"],
                final_result["face_emotion"],
                final_result["mental_state"],
                final_result["insight"],
                text_confidence=final_result["text_confidence"],
                face_confidence=final_result["face_confidence"],
                support_level=final_result["support_level"],
                input_source=final_result["input_source"],
                urgency_score=final_result["urgency_score"],
                triage_level=final_result["triage_level"],
                triage_reason=final_result["triage_reason"],
            )

            st.success("Analysis saved successfully.")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
