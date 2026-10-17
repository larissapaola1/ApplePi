"""Multipage dashboard: what we found, how well it works, why it works.

Run from the AET-HACKATHON folder:
    PYTHONPATH=. python -m streamlit run aet/app.py --server.address localhost
"""
import streamlit as st

from aet import nav

st.set_page_config(page_title="Quarter-hour prices", layout="wide")
nav.keep_shared_state()
st.navigation(nav.PAGES, position="top").run()
