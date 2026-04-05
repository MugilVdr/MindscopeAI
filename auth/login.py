import streamlit as st

from database.db_service import login_user


def login_page():
    st.title("MindScope AI - Login")
    st.caption("Sign in to review your check-ins, trends, and report history.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)

        if user:
            st.session_state["user_id"] = user[0]
            st.session_state["user_name"] = user[1]
            st.session_state["logged_in"] = True
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password")
