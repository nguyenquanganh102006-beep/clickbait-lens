"""One selected job shared by history and results pages."""
import streamlit as st


def select_job_from_widget(key):
    st.session_state["selected_job_id"] = st.session_state[key]


def sync_job_selector(key, valid_ids):
    current = st.session_state.get("selected_job_id")
    selected = current if current in valid_ids else valid_ids[0]
    st.session_state[key] = selected
    return selected
