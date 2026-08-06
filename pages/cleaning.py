"""
cleaning.py
-----------
Data Cleaning page: one-click automatic cleaning plus a full suite of
manual cleaning operations (drop duplicates, fill/drop NA, rename,
delete columns, convert dtypes).
"""

import streamlit as st

from utils.ui_helpers import require_data, page_header, metric_card
from utils import cleaner


def render():
    page_header("Data Cleaning", "Automatically clean your dataset or apply manual fixes.", "🧹")
    df = require_data()

    tab_auto, tab_manual = st.tabs(["⚡ Automatic Cleaning", "🛠️ Manual Cleaning"])

    # -------------------------------------------------------------------
    # AUTOMATIC CLEANING
    # -------------------------------------------------------------------
    with tab_auto:
        st.write(
            "Runs a safe, ordered pipeline: standardizes column names, trims whitespace, "
            "removes duplicates, fixes inconsistent text casing, detects numeric/date columns "
            "stored as text, and fills missing values intelligently."
        )

        if st.button("⚡ Run Automatic Cleaning", type="primary"):
            progress = st.progress(0, text="Starting cleaning pipeline...")
            progress.progress(30, text="Standardizing structure...")
            result = cleaner.auto_clean_pipeline(st.session_state["raw_df"])
            progress.progress(80, text="Finalizing...")
            st.session_state["df"] = result["df"]
            st.session_state["clean_log"] = result["log"]
            progress.progress(100, text="Done!")
            st.success("Automatic cleaning complete.")
            st.rerun()

        if st.session_state["clean_log"]:
            st.write("")
            st.markdown("#### What changed")
            for line in st.session_state["clean_log"]:
                st.write(f"✅ {line}")

        st.write("")
        st.markdown("#### Current Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        if st.session_state["raw_df"] is not None:
            st.write("")
            if st.button("↩️ Reset to Original Uploaded Data"):
                st.session_state["df"] = st.session_state["raw_df"].copy()
                st.session_state["clean_log"] = []
                st.rerun()

    # -------------------------------------------------------------------
    # MANUAL CLEANING
    # -------------------------------------------------------------------
    with tab_manual:
        st.write("Apply individual cleaning operations one at a time, with full control.")

        op = st.selectbox(
            "Choose an operation",
            [
                "Remove duplicate rows",
                "Drop rows with missing values",
                "Fill missing values (mean)",
                "Fill missing values (median)",
                "Fill missing values (mode)",
                "Rename a column",
                "Delete column(s)",
                "Convert column data type",
            ],
        )

        current_df = st.session_state["df"]

        if op == "Remove duplicate rows":
            subset = st.multiselect(
                "Consider only these columns when checking duplicates (optional)",
                current_df.columns.tolist(),
            )
            if st.button("Apply", type="primary"):
                before = len(current_df)
                new_df = cleaner.manual_remove_duplicates(current_df, subset=subset or None)
                st.session_state["df"] = new_df
                st.success(f"Removed {before - len(new_df)} duplicate row(s).")
                st.rerun()

        elif op == "Drop rows with missing values":
            cols = st.multiselect(
                "Limit to specific columns (optional — leave empty to check all columns)",
                current_df.columns.tolist(),
            )
            how = st.radio("Drop if...", ["any selected column is missing", "all selected columns are missing"])
            how_arg = "any" if "any" in how else "all"
            if st.button("Apply", type="primary"):
                before = len(current_df)
                new_df = cleaner.manual_drop_na(current_df, columns=cols or None, how=how_arg)
                st.session_state["df"] = new_df
                st.success(f"Dropped {before - len(new_df)} row(s).")
                st.rerun()

        elif op in ["Fill missing values (mean)", "Fill missing values (median)", "Fill missing values (mode)"]:
            method = "mean" if "mean" in op else ("median" if "median" in op else "mode")
            eligible_cols = current_df.columns.tolist() if method == "mode" else current_df.select_dtypes("number").columns.tolist()
            col = st.selectbox("Column to fill", eligible_cols)
            if st.button("Apply", type="primary"):
                missing_before = current_df[col].isna().sum()
                new_df = cleaner.manual_fill_na(current_df, col, method=method)
                st.session_state["df"] = new_df
                st.success(f"Filled {missing_before} missing value(s) in '{col}' using {method}.")
                st.rerun()

        elif op == "Rename a column":
            old_name = st.selectbox("Column to rename", current_df.columns.tolist())
            new_name = st.text_input("New column name", value=old_name)
            if st.button("Apply", type="primary") and new_name.strip():
                new_df = cleaner.manual_rename_column(current_df, old_name, new_name.strip())
                st.session_state["df"] = new_df
                st.success(f"Renamed '{old_name}' to '{new_name.strip()}'.")
                st.rerun()

        elif op == "Delete column(s)":
            cols = st.multiselect("Columns to delete", current_df.columns.tolist())
            if st.button("Apply", type="primary") and cols:
                new_df = cleaner.manual_delete_columns(current_df, cols)
                st.session_state["df"] = new_df
                st.success(f"Deleted column(s): {', '.join(cols)}")
                st.rerun()

        elif op == "Convert column data type":
            col = st.selectbox("Column", current_df.columns.tolist())
            dtype = st.selectbox("Convert to", ["numeric", "string", "datetime", "category", "boolean"])
            if st.button("Apply", type="primary"):
                new_df, success, error = cleaner.manual_convert_dtype(current_df, col, dtype)
                if success:
                    st.session_state["df"] = new_df
                    st.success(f"Converted '{col}' to {dtype}.")
                    st.rerun()
                else:
                    st.error(f"Conversion failed: {error}")

        st.write("")
        st.markdown("#### Current Data Preview")
        st.dataframe(st.session_state["df"].head(20), use_container_width=True)

    st.write("")
    st.divider()
    c1, c2, c3 = st.columns(3)
    metric_card("Rows", f"{st.session_state['df'].shape[0]:,}", c1)
    metric_card("Columns", f"{st.session_state['df'].shape[1]:,}", c2)
    metric_card("Missing Values", f"{int(st.session_state['df'].isna().sum().sum()):,}", c3)
