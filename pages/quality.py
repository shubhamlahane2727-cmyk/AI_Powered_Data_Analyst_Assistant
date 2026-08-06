"""
quality.py
----------
Data Quality Validation page: composite quality score, breakdown by
dimension, and a plain-English list of detected issues.
"""

import streamlit as st

from utils.ui_helpers import require_data, page_header, metric_card, quality_pill
from utils import profiler


def render():
    page_header("Data Quality", "Validate your dataset and see a composite quality score.", "✅")
    df = require_data()

    score = profiler.calculate_data_quality_score(df)
    issues = profiler.get_quality_issues(df)

    st.markdown("#### Overall Data Quality Score")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(
            f"""<div class="metric-card" style="text-align:center;">
                <div style="font-size:48px; font-weight:800;">{score['overall_score']}</div>
                <div class="metric-label" style="margin-top:2px;">out of 100</div>
                <div style="margin-top:8px;">{quality_pill(score['overall_score'])}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.progress(min(score["overall_score"], 100) / 100)
        st.caption(
            "Composite score across four dimensions: completeness (missing values), "
            "uniqueness (duplicates), validity (outlier rate), and consistency "
            "(columns with excessive missingness)."
        )

    st.write("")
    st.markdown("#### Score Breakdown")
    b1, b2, b3, b4 = st.columns(4)
    metric_card("Completeness", f"{score['completeness']}%", b1)
    metric_card("Uniqueness", f"{score['uniqueness']}%", b2)
    metric_card("Validity", f"{score['validity']}%", b3)
    metric_card("Consistency", f"{score['consistency']}%", b4)

    st.write("")
    st.markdown("#### Detected Issues")
    for issue in issues:
        if "No major data quality issues" in issue:
            st.success(issue)
        else:
            st.warning(issue)

    st.write("")
    st.markdown("#### Key Stats")
    s1, s2 = st.columns(2)
    metric_card("Missing Cells", f"{score['missing_cells']:,}", s1)
    metric_card("Duplicate Rows", f"{score['duplicate_rows']:,}", s2)

    st.write("")
    st.info(
        "💡 Head to the **Data Cleaning** page to automatically resolve most of "
        "these issues with one click, or fix them individually with manual tools."
    )

    # -------------------------------------------------------------------
    # BONUS: Correlation Alerts + Smart Data Recommendations
    # -------------------------------------------------------------------
    st.write("")
    st.divider()
    st.markdown("#### 🔔 Correlation Alerts")
    from utils import insights_engine
    corr_alerts = insights_engine.correlation_insights(df, threshold=0.75)
    if corr_alerts:
        for c in corr_alerts[:3]:
            st.warning(c["text"])
    else:
        st.write("No high-strength correlations (|r| ≥ 0.75) detected — no alerts to show.")

    st.write("")
    st.markdown("#### 🧭 Smart Data Recommendations")
    recommendations = []
    if score["completeness"] < 90:
        recommendations.append("Several columns have meaningful missing data — consider running Automatic Cleaning before deeper analysis.")
    if score["uniqueness"] < 95:
        recommendations.append("Duplicate rows are present — removing them will improve the accuracy of aggregated metrics.")
    if score["validity"] < 90:
        recommendations.append("Outliers are affecting data validity — review them on the Analytics > Outlier Detection tab before training ML models.")
    if not recommendations:
        recommendations.append("This dataset is in good shape — you can proceed directly to Analytics, Visualizations, or AI Insights.")
    for r in recommendations:
        st.write(f"➡️ {r}")
