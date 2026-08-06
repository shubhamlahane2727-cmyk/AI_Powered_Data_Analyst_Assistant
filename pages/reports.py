"""
reports.py
----------
Reports page: download cleaned data (CSV/Excel) and generate PDF
reports for data quality, analytics, and business insights.
"""

import streamlit as st

from utils.ui_helpers import require_data, page_header
from utils import profiler, visualizer, insights_engine, report_generator


def render():
    page_header("Reports", "Export your cleaned data and generate professional PDF reports.", "📄")
    df = require_data()
    file_name = st.session_state.get("file_name", "dataset.csv")
    roles = profiler.detect_column_roles(df)

    st.markdown("#### Export Cleaned Data")
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = report_generator.df_to_csv_bytes(df)
        st.download_button(
            "⬇️ Download Cleaned CSV", data=csv_bytes,
            file_name="cleaned_data.csv", mime="text/csv", use_container_width=True,
        )
    with c2:
        excel_bytes = report_generator.df_to_excel_bytes(df)
        st.download_button(
            "⬇️ Download Cleaned Excel", data=excel_bytes,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.write("")
    st.divider()
    st.markdown("#### Generate PDF Reports")

    tab_quality, tab_analytics, tab_insights = st.tabs(
        ["✅ Data Quality Report", "📈 Analytics Report", "💡 Business Insights Report"]
    )

    with tab_quality:
        st.write("Includes dataset overview, quality score breakdown, and detected issues.")
        if st.button("Generate Data Quality PDF", type="primary"):
            with st.spinner("Building report..."):
                basic_info = profiler.get_basic_info(df)
                quality_score = profiler.calculate_data_quality_score(df)
                issues = profiler.get_quality_issues(df)
                missing_report = profiler.get_missing_value_report(df)
                pdf_buffer = report_generator.generate_data_quality_pdf(
                    file_name, basic_info, quality_score, issues, missing_report
                )
            st.download_button(
                "⬇️ Download Data Quality Report (PDF)", data=pdf_buffer,
                file_name="data_quality_report.pdf", mime="application/pdf",
            )

    with tab_analytics:
        st.write("Includes statistical summary and correlation matrix.")
        if st.button("Generate Analytics PDF", type="primary"):
            with st.spinner("Building report..."):
                basic_info = profiler.get_basic_info(df)
                stat_summary = profiler.get_statistical_summary(df)
                corr_df = visualizer.correlation_matrix(df)
                pdf_buffer = report_generator.generate_analytics_pdf(
                    file_name, basic_info, stat_summary, corr_df
                )
            st.download_button(
                "⬇️ Download Analytics Report (PDF)", data=pdf_buffer,
                file_name="analytics_report.pdf", mime="application/pdf",
            )

    with tab_insights:
        st.write("Includes the executive summary, key findings, and recommendations.")
        if st.button("Generate Business Insights PDF", type="primary"):
            with st.spinner("Building report..."):
                if "_exec_summary" in st.session_state:
                    exec_summary = st.session_state["_exec_summary"]
                    key_findings = st.session_state["_key_findings"]
                    recommendations = st.session_state["_recommendations"]
                else:
                    quality_score = profiler.calculate_data_quality_score(df)
                    all_insights = insights_engine.generate_all_insights(df, roles)
                    exec_summary = insights_engine.generate_executive_summary(df, roles, quality_score)
                    key_findings = insights_engine.generate_key_findings(all_insights)
                    recommendations = all_insights["recommendations"]

                pdf_buffer = report_generator.generate_insights_pdf(
                    file_name, exec_summary, key_findings, recommendations
                )
            st.download_button(
                "⬇️ Download Business Insights Report (PDF)", data=pdf_buffer,
                file_name="business_insights_report.pdf", mime="application/pdf",
            )

    st.write("")
    st.divider()
    st.markdown("#### 🔁 Bonus: Dataset Comparison Tool")
    st.write("Upload a second file to compare its shape and quality score against the current dataset.")
    compare_file = st.file_uploader("Upload a comparison file (CSV or Excel)", type=["csv", "xlsx", "xls"], key="compare_uploader")

    if compare_file is not None:
        try:
            if compare_file.name.lower().endswith(".csv"):
                compare_df = __import__("pandas").read_csv(compare_file)
            else:
                compare_df = __import__("pandas").read_excel(compare_file)

            quality_a = profiler.calculate_data_quality_score(df)
            quality_b = profiler.calculate_data_quality_score(compare_df)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Current: {file_name}**")
                st.write(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
                st.write(f"Quality Score: {quality_a['overall_score']}/100")
            with c2:
                st.markdown(f"**Comparison: {compare_file.name}**")
                st.write(f"Rows: {compare_df.shape[0]:,} | Columns: {compare_df.shape[1]}")
                st.write(f"Quality Score: {quality_b['overall_score']}/100")

            common_cols = set(df.columns) & set(compare_df.columns)
            st.write("")
            st.write(f"**{len(common_cols)}** column(s) in common: {', '.join(sorted(common_cols)) if common_cols else 'None'}")
        except Exception as e:
            st.error(f"Couldn't read the comparison file: {e}")

    st.write("")
    st.divider()
    st.caption(
        "💡 Visit the **AI Insights** page first if you'd like the Business Insights "
        "report to reflect a freshly-computed analysis."
    )
