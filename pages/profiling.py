"""
profiling.py
------------
Data Profiling page: shape, dtypes, missing values, duplicates,
statistical summary, and categorical breakdowns.
"""

import streamlit as st

from utils.ui_helpers import require_data, page_header, metric_card
from utils import profiler


def render():
    page_header("Data Profiling", "A complete statistical profile of your dataset.", "🔍")
    df = require_data()

    basic_info = profiler.get_basic_info(df)
    dup_info = profiler.get_duplicate_report(df)

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Rows", f"{basic_info['rows']:,}", c1)
    metric_card("Columns", f"{basic_info['columns']:,}", c2)
    metric_card("Memory Usage", f"{basic_info['memory_usage_mb']} MB", c3)
    metric_card("Duplicate Rows", f"{dup_info['duplicate_rows']:,}", c4)

    st.write("")
    st.markdown("#### Column-wise Analysis")
    st.dataframe(profiler.get_dtype_summary(df), use_container_width=True)

    st.write("")
    missing_report = profiler.get_missing_value_report(df)
    st.markdown("#### Missing Values Report")
    if missing_report.empty:
        st.success("No missing values detected in this dataset.")
    else:
        st.dataframe(missing_report, use_container_width=True)

    st.write("")
    st.markdown("#### Statistical Summary (Numeric Columns)")
    stat_summary = profiler.get_statistical_summary(df)
    if stat_summary.empty:
        st.info("No numeric columns found in this dataset.")
    else:
        st.dataframe(stat_summary, use_container_width=True)

    st.write("")
    st.markdown("#### Categorical Analysis")
    cat_summaries = profiler.get_categorical_summary(df)
    if not cat_summaries:
        st.info("No categorical columns found in this dataset.")
    else:
        cols = list(cat_summaries.keys())
        tabs = st.tabs(cols)
        for tab, col in zip(tabs, cols):
            with tab:
                st.write(f"**{df[col].nunique()}** unique values in `{col}`")
                st.dataframe(cat_summaries[col], use_container_width=True)
