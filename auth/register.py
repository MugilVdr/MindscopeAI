import streamlit as st

from database.db_service import register_user


def register_page():
    st.title("MindScope AI - Register")
    st.caption("Create an account to save check-ins and view your own trend history.")

    name = st.text_input("Name")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if name and username and email and password:
            success = register_user(name, username, email, password)

            if success:
                st.success("Account created successfully. You can now sign in.")
                st.rerun()
            else:
                st.error("Username already exists.")
        else:
            st.warning("Please fill all fields.")
