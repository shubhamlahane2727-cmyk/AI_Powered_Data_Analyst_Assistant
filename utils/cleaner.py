"""
cleaner.py
----------
Data cleaning utilities: automatic cleaning pipeline and manual,
on-demand cleaning operations used by the Data Cleaning page.
"""

import re
import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and snake_case all column names."""
    df = df.copy()
    new_cols = []
    for col in df.columns:
        col = str(col).strip()
        col = re.sub(r"[^\w\s]", "", col)        # drop punctuation
        col = re.sub(r"\s+", "_", col)           # spaces -> underscore
        col = col.lower()
        new_cols.append(col if col else "unnamed")
    # de-duplicate any resulting collisions
    seen = {}
    final_cols = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)
    df.columns = final_cols
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing/extra internal whitespace from string cells."""
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].apply(
            lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x
        )
    return df


def fix_text_casing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize obviously inconsistent text casing in categorical-like columns,
    e.g. 'new york', 'New York', 'NEW YORK' -> 'New York'.
    Only applied to object columns with low cardinality relative to row count,
    so we don't mangle free-text fields.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        uniqueness_ratio = non_null.nunique() / len(non_null)
        if uniqueness_ratio < 0.5:  # looks categorical, not free text
            df[col] = df[col].apply(
                lambda x: x.strip().title() if isinstance(x, str) else x
            )
    return df


def detect_and_convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to intelligently convert object columns to numeric or datetime
    where the values genuinely support it.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(200)
            if sample.empty:
                continue

            # Try numeric conversion (handles things like "1,200", "$50")
            cleaned_sample = sample.str.replace(r"[,$%]", "", regex=True).str.strip()
            numeric_attempt = pd.to_numeric(cleaned_sample, errors="coerce")
            if numeric_attempt.notna().mean() > 0.9:
                cleaned_full = (
                    df[col]
                    .astype(str)
                    .str.replace(r"[,$%]", "", regex=True)
                    .str.strip()
                )
                df[col] = pd.to_numeric(cleaned_full, errors="coerce")
                continue

            # Try datetime conversion
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    date_attempt = pd.to_datetime(sample, errors="coerce")
                if date_attempt.notna().mean() > 0.9:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    continue
            except (ValueError, TypeError):
                pass
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing_values_auto(df: pd.DataFrame):
    """
    Intelligent default missing-value handling:
    - Numeric columns: fill with median
    - Datetime columns: leave as-is (imputing dates is risky)
    - Categorical/text columns: fill with mode, or 'Unknown' if no mode exists
    - Columns that are entirely empty: dropped
    """
    df = df.copy()

    # Drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)

    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        else:
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df, empty_cols


def auto_clean_pipeline(df: pd.DataFrame) -> dict:
    """
    Run the full automatic cleaning pipeline in a safe, ordered sequence.
    Returns a dict with the cleaned dataframe and a human-readable log of
    what was changed, so the UI can show a transparent before/after summary.
    """
    log = []
    original_shape = df.shape

    df = standardize_column_names(df)
    log.append("Standardized column names (lowercase, snake_case).")

    df = strip_whitespace(df)
    log.append("Removed extra/leading/trailing whitespace from text columns.")

    before_dupes = len(df)
    df = remove_duplicate_rows(df)
    removed_dupes = before_dupes - len(df)
    if removed_dupes > 0:
        log.append(f"Removed {removed_dupes} duplicate row(s).")
    else:
        log.append("No duplicate rows found.")

    df = fix_text_casing(df)
    log.append("Standardized inconsistent text casing in categorical columns.")

    df = detect_and_convert_dtypes(df)
    log.append("Auto-detected and converted numeric/date columns stored as text.")

    df, empty_cols = handle_missing_values_auto(df)
    if empty_cols:
        log.append(f"Dropped {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}.")
    log.append("Filled remaining missing values (median for numeric, mode for categorical).")

    final_shape = df.shape
    log.append(
        f"Shape changed from {original_shape[0]} rows x {original_shape[1]} cols "
        f"to {final_shape[0]} rows x {final_shape[1]} cols."
    )

    return {"df": df, "log": log}


# ---------------------------------------------------------------------------
# Manual cleaning operations (each takes a df + params, returns a new df)
# ---------------------------------------------------------------------------

def manual_remove_duplicates(df, subset=None):
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def manual_drop_na(df, columns=None, how="any"):
    if columns:
        return df.dropna(subset=columns, how=how).reset_index(drop=True)
    return df.dropna(how=how).reset_index(drop=True)


def manual_fill_na(df, column, method="mean"):
    df = df.copy()
    if method == "mean" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median" and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        mode_vals = df[column].mode()
        if not mode_vals.empty:
            df[column] = df[column].fillna(mode_vals.iloc[0])
    elif method == "zero":
        df[column] = df[column].fillna(0)
    elif method == "unknown":
        df[column] = df[column].fillna("Unknown")
    return df


def manual_rename_column(df, old_name, new_name):
    return df.rename(columns={old_name: new_name})


def manual_delete_columns(df, columns):
    return df.drop(columns=columns, errors="ignore")


def manual_convert_dtype(df, column, dtype):
    df = df.copy()
    try:
        if dtype == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "string":
            df[column] = df[column].astype(str)
        elif dtype == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif dtype == "category":
            df[column] = df[column].astype("category")
        elif dtype == "boolean":
            df[column] = df[column].astype(bool)
        return df, True, None
    except Exception as e:
        return df, False, str(e)
