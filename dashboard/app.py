"""Multipage dashboard: what we found, how well it works, why it works.

Run from the repository root:
    PYTHONPATH=. python -m streamlit run dashboard/app.py --server.address localhost
"""
import streamlit as st

import nav

st.set_page_config(page_title="Quarter-hour prices", layout="wide")
nav.keep_shared_state()
st.navigation(nav.PAGES, position="top").run()
