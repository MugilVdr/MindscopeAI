import streamlit as st
from database.db_service import get_emotion_counts


def analytics_page():

    st.title("Analytics Dashboard")

    user_id = st.session_state["user_id"]

    data = get_emotion_counts(user_id)

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

    else:
        st.warning("No data available yet")
