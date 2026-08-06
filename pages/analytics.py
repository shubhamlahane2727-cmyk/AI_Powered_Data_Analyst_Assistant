"""
analytics.py
------------
Analytics page: numerical & categorical EDA, trend analysis
(MoM/YoY growth), correlation analysis, and outlier detection
(IQR and Z-score methods).
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

from utils.ui_helpers import require_data, page_header, metric_card
from utils import profiler, visualizer


def render():
    page_header("Analytics", "Exploratory data analysis and advanced statistical insight.", "📈")
    df = require_data()
    roles = profiler.detect_column_roles(df)

    tab_eda, tab_trend, tab_corr, tab_outlier = st.tabs(
        ["📊 EDA", "📅 Trend Analysis", "🔗 Correlation Analysis", "⚠️ Outlier Detection"]
    )

    # -------------------------------------------------------------------
    # EDA
    # -------------------------------------------------------------------
    with tab_eda:
        st.markdown("#### Numerical Analysis")
        if roles["numeric"]:
            col = st.selectbox("Select a numeric column", roles["numeric"], key="eda_num_col")
            series = df[col].dropna()
            stats_dict = {
                "Mean": series.mean(), "Median": series.median(),
                "Mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
                "Variance": series.var(), "Std Dev": series.std(),
                "Minimum": series.min(), "Maximum": series.max(),
                "Q1 (25%)": series.quantile(0.25), "Q3 (75%)": series.quantile(0.75),
            }
            cols = st.columns(3)
            for i, (label, val) in enumerate(stats_dict.items()):
                metric_card(label, f"{val:,.2f}", cols[i % 3])
            st.write("")
            st.plotly_chart(visualizer.histogram(df, col), use_container_width=True)
        else:
            st.info("No numeric columns found in this dataset.")

        st.write("")
        st.markdown("#### Categorical Analysis")
        if roles["categorical"]:
            cat_col = st.selectbox("Select a categorical column", roles["categorical"], key="eda_cat_col")
            vc = df[cat_col].value_counts()
            st.write(f"**{len(vc)}** unique categories. Top category: **{vc.index[0]}** ({vc.iloc[0]} records)")
            st.plotly_chart(visualizer.bar_chart(df, cat_col), use_container_width=True)
        else:
            st.info("No categorical columns found in this dataset.")

    # -------------------------------------------------------------------
    # TREND ANALYSIS
    # -------------------------------------------------------------------
    with tab_trend:
        if not roles["date_like"]:
            st.info("No date/time column was detected. Trend analysis requires a date column.")
        elif not roles["numeric"]:
            st.info("No numeric column available to analyze trends for.")
        else:
            date_col = st.selectbox("Date column", roles["date_like"], key="trend_date_col")
            num_col = st.selectbox("Metric to analyze", roles["numeric"], key="trend_num_col")
            granularity = st.radio("Granularity", ["Monthly", "Yearly"], horizontal=True)
            freq = "ME" if granularity == "Monthly" else "YE"

            data = df[[date_col, num_col]].dropna().copy()
            data = data.set_index(date_col).resample(freq)[num_col].sum().reset_index()
            data["Growth %"] = data[num_col].pct_change() * 100

            if len(data) < 2:
                st.warning("Not enough time periods in the data to compute growth rates.")
            else:
                st.plotly_chart(
                    visualizer.line_chart(data, date_col, num_col, title=f"{num_col} Trend ({granularity})"),
                    use_container_width=True,
                )
                st.write("")
                st.markdown("##### Period-over-Period Growth")
                display_data = data.copy()
                display_data["Growth %"] = display_data["Growth %"].round(2)
                st.dataframe(display_data, use_container_width=True)

                latest_growth = data["Growth %"].iloc[-1]
                if not pd.isna(latest_growth):
                    direction = "📈 increased" if latest_growth > 0 else "📉 decreased"
                    st.metric(
                        f"Most Recent {granularity[:-2] if granularity=='Monthly' else 'Year'}-over-Period Growth",
                        f"{latest_growth:.1f}%",
                        delta=f"{latest_growth:.1f}%",
                    )

    # -------------------------------------------------------------------
    # CORRELATION ANALYSIS
    # -------------------------------------------------------------------
    with tab_corr:
        if len(roles["numeric"]) < 2:
            st.info("Need at least 2 numeric columns to compute correlations.")
        else:
            corr = visualizer.correlation_matrix(df)
            st.plotly_chart(visualizer.heatmap(df), use_container_width=True)

            st.write("")
            st.markdown("##### Strong Correlations (|r| ≥ 0.6)")
            pairs = []
            cols = corr.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr.iloc[i, j]
                    if pd.notna(val) and abs(val) >= 0.6:
                        pairs.append((cols[i], cols[j], val))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            if not pairs:
                st.write("No strong correlations (|r| ≥ 0.6) found among numeric columns.")
            else:
                pos = [p for p in pairs if p[2] > 0]
                neg = [p for p in pairs if p[2] < 0]
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Strong Positive**")
                    for a, b, v in pos:
                        st.write(f"`{a}` ↔ `{b}` — r = {v:.2f}")
                with c2:
                    st.write("**Strong Negative**")
                    for a, b, v in neg:
                        st.write(f"`{a}` ↔ `{b}` — r = {v:.2f}")

    # -------------------------------------------------------------------
    # OUTLIER DETECTION
    # -------------------------------------------------------------------
    with tab_outlier:
        if not roles["numeric"]:
            st.info("No numeric columns found in this dataset.")
        else:
            col = st.selectbox("Select a numeric column", roles["numeric"], key="outlier_col")
            method = st.radio("Detection method", ["IQR Method", "Z-Score Method"], horizontal=True)

            series = df[col].dropna()
            if method == "IQR Method":
                outlier_mask = profiler.detect_outliers_iqr(series)
                outlier_count = int(outlier_mask.sum())
                outlier_values = series[outlier_mask]
            else:
                z_scores = np.abs(scipy_stats.zscore(series))
                outlier_mask = z_scores > 3
                outlier_count = int(outlier_mask.sum())
                outlier_values = series[outlier_mask]

            c1, c2, c3 = st.columns(3)
            metric_card("Outliers Found", f"{outlier_count}", c1)
            metric_card("% of Records", f"{outlier_count/len(series)*100:.1f}%" if len(series) else "0%", c2)
            metric_card("Method", method.split(" ")[0], c3)

            st.write("")
            st.plotly_chart(visualizer.box_plot(df, col), use_container_width=True)

            if outlier_count > 0:
                st.write("")
                st.markdown("##### Outlier Values")
                st.dataframe(outlier_values.sort_values(ascending=False).to_frame(), use_container_width=True)
