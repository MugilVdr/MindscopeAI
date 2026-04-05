import streamlit as st

from auth.login import login_page
from auth.register import register_page
from database.db_setup import ensure_database
from pages.analytics import analytics_page
from pages.dashboard import dashboard_page
from pages.emotion_analysis import emotion_analysis_page
from pages.report import reports_page


st.set_page_config(page_title="MindScope AI", layout="wide")
ensure_database()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


if not st.session_state["logged_in"]:
    st.title("MindScope AI")
    st.write("AI-assisted emotional check-ins for reflection, trend tracking, and demo exploration.")
    st.caption("This app is not a medical device and should not be used as a substitute for professional care.")
    option = st.radio("Select Option", ["Login", "Register"])

    if option == "Login":
        login_page()
    else:
        register_page()
else:
    st.sidebar.title("MindScope AI")
    st.sidebar.success(f"Welcome {st.session_state['user_name']}")
    st.sidebar.caption("Use this app as a check-in and tracking tool, not as a clinical judgment system.")

    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Emotion Analysis", "Analytics", "Reports", "Logout"]
    )

    if page == "Dashboard":
        dashboard_page()
    elif page == "Emotion Analysis":
        emotion_analysis_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Reports":
        reports_page()
    elif page == "Logout":
        st.session_state.clear()
        st.success("Logged out successfully")
        st.rerun()
