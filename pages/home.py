"""
home.py
-------
Landing page: product overview, feature highlights, and quick-start CTA.
"""

import streamlit as st


def render():
    is_dark = st.session_state.get("theme") == "dark"
    accent = "#7C6FFF" if is_dark else "#6953F0"
    accent_2 = "#37D6C4" if is_dark else "#15B8A6"
    subtext = "#8B95AC" if is_dark else "#6B6B85"

    brand_name = st.session_state.get("_brand_name", "AnalytixAI")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:16px; animation: fadeInUp 0.45s cubic-bezier(0.16,1,0.3,1);">
            <div style="position:relative; width:54px; height:54px; flex-shrink:0; display:flex; align-items:center; justify-content:center;">
                <div style="position:absolute; inset:-8px; border-radius:18px;
                            background:radial-gradient(circle, {accent} 0%, {accent_2} 55%, transparent 75%);
                            filter:blur(14px); opacity:0.6; animation: glowPulse 3.2s ease-in-out infinite;"></div>
                <svg width="54" height="54" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg" style="position:relative; z-index:1;">
                    <defs>
                        <linearGradient id="heroGrad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                            <stop offset="0" stop-color="{accent}"/>
                            <stop offset="1" stop-color="{accent_2}"/>
                        </linearGradient>
                    </defs>
                    <rect x="0" y="0" width="36" height="36" rx="10" fill="url(#heroGrad)"/>
                    <path d="M7 21 L12 21 L15 13 L19 26 L22 17 L24 21 L29 21"
                          fill="none" stroke="#FFFFFF" stroke-width="2.1"
                          stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>
                </svg>
            </div>
            <h1 style='font-family:Manrope,sans-serif; font-size:42px; font-weight:800;
                       margin:0; letter-spacing:-0.02em;
                       background:linear-gradient(90deg, {accent}, {accent_2});
                       -webkit-background-clip:text; background-clip:text; color:transparent;'>
                {brand_name}
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:18px; color:{subtext}; margin-top:8px; max-width:640px;'>"
        "Upload any CSV or Excel file and instantly clean, analyze, visualize, "
        "and generate business insights — no code required.</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📤  Upload a Dataset", use_container_width=True, type="primary"):
            st.session_state["page"] = "Upload Data"
            st.rerun()
    with col2:
        if st.button("🧠  View AI Insights", use_container_width=True):
            st.session_state["page"] = "AI Insights"
            st.rerun()
    with col3:
        if st.button("📈  Build Visualizations", use_container_width=True):
            st.session_state["page"] = "Visualizations"
            st.rerun()

    if st.session_state.get("df") is not None:
        st.write("")
        if st.button("⚡ One-Click Full Analysis (Clean → Profile → Insights)", type="primary", use_container_width=True):
            with st.spinner("Running full analysis pipeline..."):
                from utils import cleaner
                result = cleaner.auto_clean_pipeline(st.session_state["raw_df"])
                st.session_state["df"] = result["df"]
                st.session_state["clean_log"] = result["log"]
            st.success("Full analysis complete — cleaned data is ready. Jump to AI Insights to see the story.")
            st.session_state["page"] = "AI Insights"
            st.rerun()

    st.write("")
    st.divider()

    st.markdown("### What this studio does")

    features = [
        ("🧹", "Automatic Data Cleaning", "Deduplication, missing-value handling, type detection, and text normalization — all in one click, with full manual controls available too."),
        ("🔍", "Data Profiling & Quality Scoring", "Instant dataset shape, types, missing-value reports, and a composite 0–100 quality score across completeness, uniqueness, validity, and consistency."),
        ("📊", "Interactive Visualizations", "Bar, line, pie, histogram, box, scatter, and correlation heatmaps — built with Plotly and fully interactive."),
        ("🤖", "Smart Chart Recommendations", "Pick any two columns and the engine recommends the best chart type for the relationship, with a plain-English reason."),
        ("💡", "AI-Powered Business Insights", "Best-performing segments, regional breakdowns, growth trends, correlations, and outlier alerts — written in plain business language."),
        ("📈", "Machine Learning Forecasts", "Train Linear Regression and Random Forest models on any numeric target, see R²/MAE/RMSE, and project future values."),
        ("📝", "Automated Data Storytelling", "Executive summaries, key findings, and recommendations generated directly from your data — ready to share."),
        ("📄", "One-Click PDF & Excel Exports", "Download cleaned data and professional PDF reports for data quality, analytics, and business insights."),
    ]

    for i in range(0, len(features), 2):
        c1, c2 = st.columns(2)
        for col, (icon, title, desc) in zip([c1, c2], features[i:i+2]):
            with col:
                st.markdown(
                    f"""<div class="metric-card" style="margin-bottom:14px;">
                        <div style="font-size:26px;">{icon}</div>
                        <div style="font-weight:700; font-size:16px; margin-top:6px;">{title}</div>
                        <div style="color:{subtext}; font-size:14px; margin-top:4px;">{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(
        "Tip: this entire workflow runs locally in this app — your data never leaves "
        "the session. All analysis is computed live from the file you upload."
    )
