"""
profiler.py
-----------
Data profiling, data-quality scoring, and column-role detection.
These helpers are used across the Profiling, Quality, Analytics,
Insights, and ML pages so column type/role detection is consistent
everywhere.

Performance note: the functions below are decorated with
@st.cache_data so repeated Streamlit reruns (which happen on every
widget interaction) don't re-run expensive pandas/numpy work on a
dataframe that hasn't actually changed. Streamlit hashes the
dataframe's contents to decide when to invalidate the cache.
"""

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def get_basic_info(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "column_names": list(df.columns),
    }


@st.cache_data(show_spinner=False)
def get_dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Missing Values": [df[c].isna().sum() for c in df.columns],
        "Missing %": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    })


@st.cache_data(show_spinner=False)
def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    report = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
    })
    return report[report["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_duplicate_report(df: pd.DataFrame) -> dict:
    dup_count = df.duplicated().sum()
    return {
        "duplicate_rows": int(dup_count),
        "duplicate_pct": round((dup_count / len(df)) * 100, 2) if len(df) else 0,
    }


@st.cache_data(show_spinner=False)
def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.DataFrame()
    summary = numeric_df.describe().T
    summary["variance"] = numeric_df.var()
    summary = summary.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std Dev",
        "min": "Min", "25%": "Q1 (25%)", "50%": "Median",
        "75%": "Q3 (75%)", "max": "Max", "variance": "Variance"
    })
    cols_order = ["Count", "Mean", "Median", "Std Dev", "Variance",
                  "Min", "Q1 (25%)", "Q3 (75%)", "Max"]
    return summary[cols_order].round(3)


@st.cache_data(show_spinner=False)
def get_categorical_summary(df: pd.DataFrame, top_n: int = 5) -> dict:
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    summaries = {}
    for col in cat_cols:
        vc = df[col].value_counts().head(top_n)
        summaries[col] = pd.DataFrame({
            "Value": vc.index,
            "Count": vc.values,
            "Percent": (vc.values / len(df) * 100).round(2),
        })
    return summaries


# ---------------------------------------------------------------------------
# Column role detection — shared "brain" used by analytics/insights/ML
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def detect_column_roles(df: pd.DataFrame) -> dict:
    """
    Classify each column into a role: numeric, categorical, datetime, or text.
    Also attempts to semantically tag likely business columns (revenue-like,
    region-like, product-like, date-like) purely from column name + data
    shape heuristics, with no hardcoded dataset assumptions.
    """
    numeric_cols, categorical_cols, datetime_cols, text_cols = [], [], [], []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            non_null = df[col].dropna()
            if len(non_null) == 0:
                categorical_cols.append(col)
                continue
            uniqueness_ratio = non_null.nunique() / len(non_null)
            if uniqueness_ratio < 0.5:
                categorical_cols.append(col)
            else:
                text_cols.append(col)

    # Semantic tagging by keyword heuristics on column names (lowercased)
    def name_matches(col, keywords):
        cl = col.lower()
        return any(k in cl for k in keywords)

    revenue_like = [c for c in numeric_cols if name_matches(
        c, ["revenue", "sales", "amount", "price", "income", "profit", "total", "cost"]
    )]
    region_like = [c for c in categorical_cols if name_matches(
        c, ["region", "country", "state", "city", "location", "area", "territory"]
    )]
    product_like = [c for c in categorical_cols if name_matches(
        c, ["product", "item", "sku", "category", "service"]
    )]
    customer_like = [c for c in categorical_cols if name_matches(
        c, ["customer", "client", "user", "buyer"]
    )]
    date_like = list(datetime_cols)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "text": text_cols,
        "revenue_like": revenue_like,
        "region_like": region_like,
        "product_like": product_like,
        "customer_like": customer_like,
        "date_like": date_like,
        "is_business_dataset": bool(revenue_like or region_like or product_like),
    }


# ---------------------------------------------------------------------------
# Data quality scoring
# ---------------------------------------------------------------------------

def detect_outliers_iqr(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=bool)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


@st.cache_data(show_spinner=False)
def calculate_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Composite quality score out of 100, built from four weighted components:
    completeness (missing values), uniqueness (duplicates), consistency
    (mixed/invalid types), and validity (outlier rate in numeric columns).
    """
    total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 1

    # Completeness (40 pts)
    missing_cells = df.isna().sum().sum()
    completeness_score = max(0, 40 * (1 - missing_cells / total_cells))

    # Uniqueness (20 pts) — penalize duplicate rows
    dup_pct = df.duplicated().mean() if len(df) else 0
    uniqueness_score = max(0, 20 * (1 - dup_pct))

    # Validity (20 pts) — penalize heavy outlier presence in numeric cols
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        outlier_ratios = []
        for col in numeric_df.columns:
            outliers = detect_outliers_iqr(numeric_df[col])
            if len(outliers):
                outlier_ratios.append(outliers.mean())
        avg_outlier_ratio = np.mean(outlier_ratios) if outlier_ratios else 0
    else:
        avg_outlier_ratio = 0
    validity_score = max(0, 20 * (1 - avg_outlier_ratio))

    # Consistency (20 pts) — penalize columns with very high missingness (>50%)
    cols_high_missing = (df.isna().mean() > 0.5).sum()
    consistency_score = max(0, 20 * (1 - cols_high_missing / max(df.shape[1], 1)))

    total_score = round(
        completeness_score + uniqueness_score + validity_score + consistency_score, 1
    )

    return {
        "overall_score": min(total_score, 100),
        "completeness": round(completeness_score / 40 * 100, 1),
        "uniqueness": round(uniqueness_score / 20 * 100, 1),
        "validity": round(validity_score / 20 * 100, 1),
        "consistency": round(consistency_score / 20 * 100, 1),
        "missing_cells": int(missing_cells),
        "duplicate_rows": int(df.duplicated().sum()),
    }


@st.cache_data(show_spinner=False)
def get_quality_issues(df: pd.DataFrame) -> list:
    """Return a list of plain-English data quality issue descriptions."""
    issues = []

    missing = df.isna().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        issues.append(
            f"{len(cols_with_missing)} column(s) contain missing values: "
            f"{', '.join(cols_with_missing.index[:5])}"
            + (" ..." if len(cols_with_missing) > 5 else "")
        )

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate row(s) detected.")

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        issues.append(f"{len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}")

    numeric_df = df.select_dtypes(include=np.number)
    for col in numeric_df.columns:
        outliers = detect_outliers_iqr(numeric_df[col])
        if len(outliers) and outliers.sum() > 0:
            pct = round(outliers.mean() * 100, 1)
            if pct > 5:
                issues.append(f"Column '{col}' has {outliers.sum()} outlier(s) ({pct}% of values).")

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(50)
            if sample.str.match(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$").any():
                parsed = pd.to_datetime(df[col], errors="coerce")
                invalid_pct = parsed.isna().mean() if len(df[col].dropna()) else 0
                if 0 < invalid_pct < 1:
                    issues.append(
                        f"Column '{col}' looks like dates but has inconsistent/invalid formats."
                    )

    if not issues:
        issues.append("No major data quality issues detected. Dataset looks clean.")

    return issues
