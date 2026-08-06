"""
ui_helpers.py
-------------
Small shared UI helpers used across multiple pages: the "no data loaded"
guard, KPI metric cards, and quality-score pill rendering.
"""

import streamlit as st


def require_data():
    """
    Stop rendering and show a friendly prompt if no dataset has been
    uploaded yet. Returns the active dataframe if available.
    """
    if st.session_state.get("df") is None:
        st.warning("📂 No dataset loaded yet. Head to **Upload Data** to get started.")
        if st.button("Go to Upload Data"):
            st.session_state["page"] = "Upload Data"
            st.rerun()
        st.stop()
    return st.session_state["df"]


def metric_card(label, value, col=None):
    target = col if col is not None else st
    target.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def quality_pill(score):
    if score >= 80:
        cls, label = "pill-good", "Good"
    elif score >= 60:
        cls, label = "pill-warn", "Fair"
    else:
        cls, label = "pill-bad", "Needs Attention"
    return f'<span class="pill {cls}">{label}</span>'


def page_header(title, subtitle=None, icon=""):
    st.markdown(f"## {icon} {title}")
    if subtitle:
        st.caption(subtitle)
    st.write("")
