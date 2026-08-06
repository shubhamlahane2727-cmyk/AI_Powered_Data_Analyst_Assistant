"""
ml_engine.py
------------
Lightweight ML module: Linear Regression and Random Forest Regression
against a user-chosen numeric target column, with evaluation metrics
and simple forward predictions. Also includes feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def prepare_features(df, target_col, feature_cols):
    """
    Build a model-ready feature matrix: numeric columns pass through,
    categorical columns get label-encoded. Datetime columns are converted
    to ordinal (days since epoch) so they can be used as numeric features.
    Rows with missing target are dropped; remaining missing feature values
    are filled with median/mode.
    """
    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    encoders = {}

    for col in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].map(pd.Timestamp.toordinal)
        elif not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].astype(str).fillna("Unknown")
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median() if pd.api.types.is_numeric_dtype(data[col]) else 0)

    X = data[feature_cols]
    y = data[target_col]
    return X, y, encoders


def train_models(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """Train both Linear Regression and Random Forest; return metrics + models."""
    X, y, encoders = prepare_features(df, target_col, feature_cols)

    if len(X) < 10:
        return {"error": "Not enough rows with valid data to train a model (need at least 10)."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["linear_regression"] = {
        "model": lr,
        "r2": r2_score(y_test, lr_pred),
        "mae": mean_absolute_error(y_test, lr_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, lr_pred))),
        "predictions": lr_pred,
        "actuals": y_test.values,
    }

    rf = RandomForestRegressor(n_estimators=200, random_state=random_state, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "r2": r2_score(y_test, rf_pred),
        "mae": mean_absolute_error(y_test, rf_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        "predictions": rf_pred,
        "actuals": y_test.values,
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
    }

    results["feature_cols"] = feature_cols
    results["encoders"] = encoders
    results["X_columns"] = list(X.columns)
    return results


def predict_future(model, df, target_col, feature_cols, encoders, n_periods, date_col=None):
    """
    Simple forward prediction: repeats the most recent feature row n_periods
    times, advancing the date column (if provided) by one period each step,
    and holding other features at their last observed / mean values.
    This is intentionally simple and transparent rather than a full
    time-series model, and is presented to users as a projection, not a guarantee.
    """
    last_row = df[feature_cols].dropna().iloc[-1:].copy()
    if last_row.empty:
        return None

    future_rows = []
    for i in range(1, n_periods + 1):
        row = last_row.copy()
        if date_col and date_col in row.columns:
            base_date = df[date_col].dropna().iloc[-1]
            row[date_col] = base_date + pd.DateOffset(months=i)
        future_rows.append(row)
    future_df = pd.concat(future_rows, ignore_index=True)

    X_future = future_df.copy()
    for col in feature_cols:
        if col in encoders:
            X_future[col] = X_future[col].astype(str)
            known_classes = set(encoders[col].classes_)
            X_future[col] = X_future[col].apply(lambda v: v if v in known_classes else encoders[col].classes_[0])
            X_future[col] = encoders[col].transform(X_future[col])
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            X_future[col] = X_future[col].map(pd.Timestamp.toordinal)

    preds = model.predict(X_future[feature_cols])
    result = pd.DataFrame({"period": range(1, n_periods + 1), "predicted_" + target_col: preds})
    if date_col:
        result["date"] = future_df[date_col].values if date_col in future_df.columns else None
    return result


def get_feature_importance_df(importance_dict):
    df = pd.DataFrame({
        "Feature": list(importance_dict.keys()),
        "Importance": list(importance_dict.values()),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Importance %"] = (df["Importance"] / df["Importance"].sum() * 100).round(2)
    return df
