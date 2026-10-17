"""The app's pages, defined once so pages can link to each other (st.switch_page / st.page_link)."""
import streamlit as st

from aet import dashboard_2, eda_page, explain_page

DATA_STORY = st.Page(eda_page.render, title="Data Story", icon=":material/auto_stories:", url_path="data-story",
                     default=True)
MODEL_RESULTS = st.Page(dashboard_2.main, title="Model Results", icon=":material/query_stats:", url_path="model-results")
UNDERSTANDING = st.Page(explain_page.render, title="Understanding the Model", icon=":material/psychology:", url_path="understanding-the-model")
PAGES = [DATA_STORY, MODEL_RESULTS, UNDERSTANDING]

# Widget values that should survive moving between pages (Streamlit forgets widgets that are not drawn).
SHARED_KEYS = ("zone", "period", "day", "week", "custom_range", "show",          # Model Results
               "explain_zone", "explain_date", "explain_hour",                    # Understanding the Model
               "eda_zone", "eda_trend")                                           # Data Story


def keep_shared_state() -> None:
    for key in SHARED_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
