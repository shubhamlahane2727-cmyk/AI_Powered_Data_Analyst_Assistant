"""
ml.py
-----
Machine Learning page: train Linear Regression and Random Forest models
against a user-chosen numeric target, show evaluation metrics and
feature importance, and generate forward projections.
"""

import streamlit as st

from utils.ui_helpers import require_data, page_header, metric_card
from utils import profiler, ml_engine, visualizer


def render():
    page_header("Machine Learning", "Train regression models and project future values.", "🤖")
    df = require_data()
    roles = profiler.detect_column_roles(df)

    if not roles["numeric"]:
        st.info("No numeric columns available to use as a prediction target.")
        return

    st.markdown("#### 1. Choose a Target & Features")
    target_col = st.selectbox("Target column (what to predict)", roles["numeric"])

    feature_candidates = [c for c in df.columns if c != target_col]
    default_features = [c for c in feature_candidates if c in (roles["numeric"] + roles["categorical"])][:5]
    feature_cols = st.multiselect(
        "Feature columns (what to predict from)",
        feature_candidates,
        default=default_features or feature_candidates[:3],
    )

    if not feature_cols:
        st.warning("Select at least one feature column.")
        return

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training Linear Regression and Random Forest models..."):
            results = ml_engine.train_models(df, target_col, feature_cols)
        st.session_state["ml_results"] = results
        st.session_state["ml_target"] = target_col
        st.session_state["ml_features"] = feature_cols

    results = st.session_state.get("ml_results")
    if results and "error" in results:
        st.error(results["error"])
        return

    if results and st.session_state.get("ml_target") == target_col:
        st.write("")
        st.markdown("#### 2. Model Performance")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Linear Regression")
            lr = results["linear_regression"]
            m1, m2, m3 = st.columns(3)
            metric_card("R² Score", f"{lr['r2']:.3f}", m1)
            metric_card("MAE", f"{lr['mae']:.2f}", m2)
            metric_card("RMSE", f"{lr['rmse']:.2f}", m3)
        with c2:
            st.markdown("##### Random Forest Regression")
            rf = results["random_forest"]
            m1, m2, m3 = st.columns(3)
            metric_card("R² Score", f"{rf['r2']:.3f}", m1)
            metric_card("MAE", f"{rf['mae']:.2f}", m2)
            metric_card("RMSE", f"{rf['rmse']:.2f}", m3)

        better_model = "Random Forest" if rf["r2"] > lr["r2"] else "Linear Regression"
        st.info(f"📌 **{better_model}** has the better fit on this dataset (higher R²).")

        st.write("")
        st.markdown("#### 3. Feature Importance (Random Forest)")
        fi_df = ml_engine.get_feature_importance_df(rf["feature_importance"])
        st.dataframe(fi_df, use_container_width=True)
        st.plotly_chart(
            visualizer.bar_chart(fi_df, "Feature", "Importance %", title="Feature Importance"),
            use_container_width=True,
        )

        st.write("")
        st.markdown("#### 4. Generate Future Predictions")
        date_candidates = [c for c in feature_cols if c in roles["date_like"]]
        n_periods = st.slider("Number of future periods to project", 1, 12, 3)
        model_choice = st.radio("Model to use for projection", ["Random Forest", "Linear Regression"], horizontal=True)
        model_obj = rf["model"] if model_choice == "Random Forest" else lr["model"]

        if st.button("Generate Projections"):
            future_df = ml_engine.predict_future(
                model_obj, df, target_col, feature_cols, results["encoders"],
                n_periods, date_col=date_candidates[0] if date_candidates else None,
            )
            if future_df is None:
                st.error("Couldn't generate future predictions — not enough valid feature data.")
            else:
                st.dataframe(future_df, use_container_width=True)
                st.caption(
                    "Note: these projections hold most features at their last observed values and are "
                    "intended as a directional estimate, not a guaranteed forecast."
                )
    elif results and st.session_state.get("ml_target") != target_col:
        st.info("Target column changed — click 'Train Models' again to retrain.")
