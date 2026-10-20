"""Render defaults only for new widgets, preserving Streamlit state cleanly."""
import streamlit as st


def number_input(label, *, value, key, **kwargs):
    defaults = {} if key in st.session_state else {"value": value}
    return st.number_input(label, key=key, **defaults, **kwargs)


def selectbox(label, options, *, index=0, key, **kwargs):
    defaults = {} if key in st.session_state else {"index": index}
    return st.selectbox(label, options, key=key, **defaults, **kwargs)
