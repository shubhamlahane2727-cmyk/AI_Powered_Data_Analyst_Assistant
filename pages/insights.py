"""
insights.py
-----------
AI Insights page: business insights (best performer, regional, growth,
seasonal, correlations, outliers, customer behavior) plus the automated
data-storytelling output (executive summary, key findings, recommendations).
"""

import streamlit as st

from utils.ui_helpers import require_data, page_header
from utils import profiler, insights_engine, visualizer


def render():
    page_header("AI Insights", "Automatically generated business insights and data storytelling.", "💡")
    df = require_data()
    roles = profiler.detect_column_roles(df)

    with st.spinner("Analyzing dataset for insights..."):
        quality_score = profiler.calculate_data_quality_score(df)
        all_insights = insights_engine.generate_all_insights(df, roles)
        exec_summary = insights_engine.generate_executive_summary(df, roles, quality_score)
        key_findings = insights_engine.generate_key_findings(all_insights)

    # store for the Reports page to reuse without recomputation
    st.session_state["_exec_summary"] = exec_summary
    st.session_state["_key_findings"] = key_findings
    st.session_state["_recommendations"] = all_insights["recommendations"]
    st.session_state["_all_insights"] = all_insights

    tab_story, tab_insights, tab_alerts = st.tabs(
        ["📝 Data Story", "💡 Business Insights", "🚨 Alerts & Opportunities"]
    )

    # -------------------------------------------------------------------
    # AUTOMATED DATA STORYTELLING
    # -------------------------------------------------------------------
    with tab_story:
        st.markdown("#### Executive Summary")
        st.write(exec_summary)

        st.write("")
        st.markdown("#### Key Findings")
        for f in key_findings:
            st.write(f"• {f}")

        st.write("")
        st.markdown("#### Recommendations")
        for r in all_insights["recommendations"]:
            st.write(f"✅ {r}")

    # -------------------------------------------------------------------
    # BUSINESS INSIGHTS
    # -------------------------------------------------------------------
    with tab_insights:
        if not roles["is_business_dataset"]:
            st.info(
                "This dataset doesn't contain obvious business dimensions "
                "(revenue, region, product). Insights below are generated generically "
                "from the available numeric and categorical columns."
            )

        if all_insights["performance"]:
            st.markdown("#### Best Performing Segments")
            for p in all_insights["performance"]:
                st.write(p["text"])
                with st.expander(f"View chart: {p['numeric_column']} by {p['category_column']}"):
                    st.plotly_chart(
                        visualizer.bar_chart(df, p["category_column"], p["numeric_column"]),
                        use_container_width=True,
                    )
            st.write("")

        if all_insights["regional"]:
            st.markdown("#### Regional / Segment Performance")
            st.write(all_insights["regional"]["text"])
            st.write("")

        if all_insights["growth"]:
            st.markdown("#### Growth Trends")
            st.write(all_insights["growth"]["text"])
            st.write("")

        if all_insights["seasonal"]:
            st.markdown("#### Seasonal Patterns")
            st.write(all_insights["seasonal"]["text"])
            st.write("")

        if all_insights["customer"]:
            st.markdown("#### Customer Behavior Patterns")
            for c in all_insights["customer"]:
                st.write(c)
            st.write("")

        if all_insights["correlations"]:
            st.markdown("#### Correlation Insights")
            for c in all_insights["correlations"]:
                st.write(c["text"])

        if not any([
            all_insights["performance"], all_insights["regional"], all_insights["growth"],
            all_insights["customer"], all_insights["correlations"]
        ]):
            st.write("Not enough structured signal in this dataset to generate detailed business insights.")

    # -------------------------------------------------------------------
    # ALERTS & OPPORTUNITIES
    # -------------------------------------------------------------------
    with tab_alerts:
        if all_insights["outliers"]:
            st.markdown("#### Outlier Alerts")
            for o in all_insights["outliers"]:
                st.warning(o)
        else:
            st.success("No significant outlier alerts detected.")

        st.write("")
        st.markdown("#### Opportunities for Improvement")
        for r in all_insights["recommendations"]:
            st.write(f"🎯 {r}")
