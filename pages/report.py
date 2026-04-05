import streamlit as st
import pandas as pd
from database.db_service import get_user_history


def reports_page():

    st.title("Reports")

    user_id = st.session_state["user_id"]

    data = get_user_history(user_id)

    if data:

        df = pd.DataFrame(data, columns=[
            "Text", "Face Emotion", "Mental State", "Insight", "Date"
        ])

        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "report.csv"
        )

    else:
        st.warning("No reports available")