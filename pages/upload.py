"""
upload.py
---------
File upload page: accepts CSV/Excel, displays file info, and stores
the raw dataframe in session state for downstream pages.
"""

import re

import pandas as pd
import streamlit as st

from utils.ui_helpers import page_header, metric_card


def _read_file(uploaded_file):
    name = uploaded_file.name
    if name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")


def _drop_stray_index_columns(df):
    """
    Files exported from pandas/Excel with the row index included (e.g.
    df.to_csv(...) without index=False) produce one or more unnamed,
    near-sequential columns like 'Unnamed: 0'. These aren't real data —
    they're row numbers — and including them in charts/analysis produces
    meaningless results (e.g. a bar chart with one bar per row). This
    detects and removes them automatically, reporting what was dropped.
    """
    dropped = []
    for col in list(df.columns):
        looks_unnamed = bool(re.match(r"^Unnamed:\s*\d+$", str(col)))
        if not looks_unnamed:
            continue
        series = df[col]
        # Only drop if it actually looks like a row-index column: numeric
        # and either a 0..n-1 / 1..n sequence or fully unique integers.
        if pd.api.types.is_numeric_dtype(series):
            is_sequential = series.equals(pd.Series(range(len(series)), index=series.index)) or \
                             series.equals(pd.Series(range(1, len(series) + 1), index=series.index))
            is_fully_unique = series.nunique() == len(series)
            if is_sequential or is_fully_unique:
                dropped.append(col)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def render():
    page_header("Upload Data", "Bring in a CSV or Excel file to start your analysis.", "📤")

    uploaded_file = st.file_uploader(
        "Drag and drop or browse for a file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV (.csv), Excel (.xlsx, .xls)",
    )

    if uploaded_file is not None:
        with st.spinner("Reading file..."):
            try:
                df = _read_file(uploaded_file)
            except Exception as e:
                st.error(f"Could not read this file: {e}")
                return

        if df.empty:
            st.error("The uploaded file appears to be empty.")
            return

        df, dropped_cols = _drop_stray_index_columns(df)

        st.session_state["raw_df"] = df.copy()
        st.session_state["df"] = df.copy()
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["file_size_kb"] = round(uploaded_file.size / 1024, 1)
        st.session_state["clean_log"] = []
        st.session_state["ml_results"] = None

        st.success(f"✅ Loaded **{uploaded_file.name}** successfully.")
        if dropped_cols:
            st.info(
                f"ℹ️ Removed {len(dropped_cols)} stray index column(s) that had no real data "
                f"(just row numbers carried over from the export): {', '.join(dropped_cols)}."
            )

    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.write("")
        st.markdown("#### File Summary")

        c1, c2, c3, c4 = st.columns(4)
        metric_card("File Name", st.session_state["file_name"], c1)
        metric_card("Rows", f"{df.shape[0]:,}", c2)
        metric_card("Columns", f"{df.shape[1]:,}", c3)
        metric_card("File Size", f"{st.session_state['file_size_kb']} KB", c4)

        st.write("")
        st.markdown("#### Column Names")
        st.write(", ".join([f"`{c}`" for c in df.columns]))

        st.write("")
        st.markdown("#### Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 Continue to Data Cleaning", type="primary", use_container_width=True):
                st.session_state["page"] = "Data Cleaning"
                st.rerun()
        with col_b:
            if st.button("🔍 Skip to Data Profiling", use_container_width=True):
                st.session_state["page"] = "Data Profiling"
                st.rerun()
    else:
        st.info("No file uploaded yet. Choose a CSV or Excel file above to begin.")
