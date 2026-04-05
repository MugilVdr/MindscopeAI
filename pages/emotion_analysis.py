import streamlit as st

from database.db_service import save_prediction
from services.face_prediction import predict_face_emotion
from services.fusion_engine import combine_predictions
from services.text_prediction import predict_text_mental_state


def emotion_analysis_page():
    st.title("Emotion Analysis")

    text = st.text_area("Enter your thoughts")
    face_image = st.file_uploader(
        "Upload a face image (optional)",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Analyze Emotion"):
        try:
            text_result = predict_text_mental_state(text)
            face_result = None

            if face_image is not None:
                face_result = predict_face_emotion(face_image.getvalue())

            final_result = combine_predictions(text_result, face_result)

            st.subheader("Results")
            col1, col2 = st.columns(2)
            col1.success(
                f"Text Mental State: {final_result['mental_state']} "
                f"({final_result['text_confidence'] * 100:.1f}%)"
            )

            face_line = f"Face Emotion: {final_result['face_emotion']}"
            if final_result["face_confidence"] is not None:
                face_line += f" ({final_result['face_confidence'] * 100:.1f}%)"
            col2.warning(face_line)

            st.subheader("Insight")
            st.info(final_result["insight"])

            st.subheader("Recommendation")
            for recommendation in final_result["recommendations"]:
                st.write(f"- {recommendation}")

            save_prediction(
                st.session_state["user_id"],
                text_result["cleaned_text"],
                final_result["face_emotion"],
                final_result["mental_state"],
                final_result["insight"]
            )

            st.success("Saved successfully")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
