"""
app.py
------
Main entry point for the AI-Powered Data Cleaning & Analytics Studio.
Handles global page config, theming, sidebar navigation, and session
state initialization, then routes to the selected page module.
"""

import streamlit as st

st.set_page_config(
    page_title="AI-Powered Data Analyst Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_NAME = "AnalytixAI"
BRAND_TAGLINE = "Analytics Studio"
st.session_state.setdefault("_brand_name", BRAND_NAME)
st.session_state.setdefault("theme", "light")

# ---------------------------------------------------------------------------
# Instant background paint — applied before anything else renders, so the
# very first frame already matches the active theme instead of flashing
# Streamlit's default white background while the full CSS loads below.
# ---------------------------------------------------------------------------
_instant_bg = "#0A0E1A" if st.session_state["theme"] == "dark" else "#F7F7FB"
st.markdown(
    f"<style>html, body, .stApp {{ background-color: {_instant_bg} !important; }}</style>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
defaults = {
    "raw_df": None,
    "df": None,
    "file_name": None,
    "file_size_kb": None,
    "clean_log": [],
    "theme": "light",
    "page": "Home",
    "ml_results": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Theme (Light / Dark) — injected as CSS since Streamlit has no native
# per-app dark mode toggle that works reliably across versions.
# ---------------------------------------------------------------------------
def inject_theme_css():
    # Load Google Fonts once via <link> tags (cached aggressively by the
    # browser) instead of a CSS @import, which re-triggers a network
    # round-trip on every single Streamlit rerun and was a real source
    # of perceived lag/jank on slower connections.
    if not st.session_state.get("_fonts_loaded"):
        st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        """, unsafe_allow_html=True)
        st.session_state["_fonts_loaded"] = True

    if st.session_state["theme"] == "dark":
        bg = "#0A0E1A"
        card = "#121826"
        card_hover = "#1A2235"
        text = "#EEF1F8"
        subtext = "#8B95AC"
        border = "#252E45"
        accent = "#7C6FFF"
        accent_2 = "#37D6C4"
        accent_soft = "rgba(124, 111, 255, 0.16)"
        sidebar_bg = "#0D1220"
        nav_active_bg = "rgba(124, 111, 255, 0.16)"
    else:
        bg = "#F7F7FB"
        card = "#FFFFFF"
        card_hover = "#FBFBFF"
        text = "#161325"
        subtext = "#6B6B85"
        border = "#E7E6F2"
        accent = "#6953F0"
        accent_2 = "#15B8A6"
        accent_soft = "rgba(105, 83, 240, 0.08)"
        sidebar_bg = "#FFFFFF"
        nav_active_bg = "rgba(105, 83, 240, 0.10)"

    st.markdown(f"""
    <style>
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        @keyframes glowPulse {{
            0%, 100% {{ opacity: 0.55; transform: scale(1); }}
            50% {{ opacity: 0.9; transform: scale(1.08); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            transition: background-color 0.25s ease, color 0.25s ease;
        }}

        .stApp {{
            background-color: {bg};
        }}

        [data-testid="stAppViewContainer"] .main .block-container {{
            animation: fadeIn 0.22s ease-out;
            padding-top: 2.2rem;
            max-width: 1180px;
        }}

        [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {border};
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Manrope', sans-serif;
            color: {text};
            letter-spacing: -0.015em;
        }}
        p, span, div, label {{
            color: {text};
        }}

        /* ---------------- Brand header (signature element) ---------------- */
        .brand-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 0 24px 0;
            animation: fadeIn 0.5s ease;
        }}
        .brand-logo-wrap {{
            position: relative;
            flex-shrink: 0;
            width: 38px;
            height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .brand-logo-glow {{
            position: absolute;
            inset: -6px;
            border-radius: 14px;
            background: radial-gradient(circle, {accent} 0%, {accent_2} 55%, transparent 75%);
            filter: blur(10px);
            animation: glowPulse 3.2s ease-in-out infinite;
            z-index: 0;
        }}
        .brand-logo {{
            position: relative;
            z-index: 1;
            flex-shrink: 0;
            display: flex;
            transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        .brand-header:hover .brand-logo {{
            transform: scale(1.07) rotate(-3deg);
        }}
        .brand-name {{
            font-family: 'Manrope', sans-serif;
            font-size: 19px;
            font-weight: 800;
            color: {text};
            letter-spacing: -0.02em;
            line-height: 1.1;
        }}
        .brand-tagline {{
            font-size: 10px;
            font-weight: 600;
            color: {subtext};
            letter-spacing: 0.09em;
            text-transform: uppercase;
            line-height: 1.2;
        }}

        /* ---------------- Metric cards ---------------- */
        .metric-card {{
            position: relative;
            background-color: {card};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            transition: transform 0.22s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.22s ease, border-color 0.22s ease;
            animation: fadeInUp 0.35s ease;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, {accent}, {accent_2});
            opacity: 0;
            transition: opacity 0.25s ease;
            z-index: 0;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.10);
            border-color: transparent;
        }}
        .metric-card:hover::before {{
            opacity: 1;
        }}
        .metric-card > * {{
            position: relative;
            z-index: 1;
        }}
        .metric-label {{
            color: {subtext};
            font-size: 12.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .metric-value {{
            font-family: 'Manrope', sans-serif;
            font-size: 27px;
            font-weight: 800;
            color: {text};
            margin-top: 4px;
            letter-spacing: -0.01em;
        }}

        /* ---------------- Pills / badges ---------------- */
        .pill {{
            display: inline-block;
            padding: 3px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            transition: transform 0.15s ease;
        }}
        .pill-good {{ background-color: rgba(21, 184, 166, 0.16); color: #0E9686; }}
        .pill-warn {{ background-color: rgba(245, 158, 11, 0.16); color: #B45309; }}
        .pill-bad {{ background-color: rgba(239, 68, 68, 0.14); color: #DC2626; }}

        /* ---------------- Sidebar nav (radio) ---------------- */
        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
            gap: 3px;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            font-size: 14.5px;
            font-weight: 500;
            padding: 9px 12px;
            border-radius: 9px;
            width: 100%;
            transition: background-color 0.18s ease, padding-left 0.18s cubic-bezier(0.16, 1, 0.3, 1), color 0.18s ease;
            cursor: pointer;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            background-color: {accent_soft};
            padding-left: 17px;
        }}
        section[data-testid="stSidebar"] .stRadio label > div:first-child {{
            display: none;
        }}
        section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
            background: linear-gradient(90deg, {nav_active_bg}, transparent);
            font-weight: 700;
            border-left: 3px solid {accent};
            padding-left: 14px;
        }}

        /* ---------------- Buttons ---------------- */
        .stButton button {{
            border-radius: 10px;
            transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.18s ease, filter 0.18s ease;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }}
        .stButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.14);
            filter: brightness(1.04);
        }}
        .stButton button:active {{
            transform: translateY(0);
        }}
        .stDownloadButton button {{
            border-radius: 10px;
            transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.18s ease;
        }}
        .stDownloadButton button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.14);
        }}

        /* ---------------- Tabs ---------------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid {border};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            transition: background-color 0.18s ease, color 0.18s ease;
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            animation: fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .stTabs [aria-selected="true"] {{
            color: {accent} !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {accent} !important;
            height: 2.5px !important;
        }}

        /* ---------------- Inputs ---------------- */
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput input,
        .stMultiSelect div[data-baseweb="select"] > div {{
            border-radius: 9px !important;
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }}
        .stSelectbox div[data-baseweb="select"] > div:focus-within,
        .stTextInput input:focus,
        .stMultiSelect div[data-baseweb="select"] > div:focus-within {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 3px {accent_soft} !important;
        }}

        /* ---------------- Progress bar ---------------- */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {accent}, {accent_2}) !important;
        }}

        /* ---------------- Misc polish ---------------- */
        div[data-testid="stMetricValue"] {{
            font-family: 'Manrope', sans-serif;
            font-weight: 800;
        }}
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
            animation: fadeIn 0.35s ease;
            border: 1px solid {border};
        }}
        .stAlert {{
            border-radius: 10px;
            animation: fadeInUp 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        hr {{
            margin: 1.4rem 0;
            opacity: 0.5;
        }}
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-thumb {{
            background-color: {border};
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background-color: {subtext};
        }}
    </style>
    """, unsafe_allow_html=True)


def render_brand_header():
    is_dark = st.session_state["theme"] == "dark"
    accent = "#7C6FFF" if is_dark else "#6953F0"
    accent_2 = "#37D6C4" if is_dark else "#15B8A6"
    st.markdown(
        f"""
        <div class="brand-header">
            <div class="brand-logo-wrap">
                <div class="brand-logo-glow"></div>
                <span class="brand-logo">
                    <svg width="36" height="36" viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="brandGrad" x1="0" y1="0" x2="36" y2="36" gradientUnits="userSpaceOnUse">
                                <stop offset="0" stop-color="{accent}"/>
                                <stop offset="1" stop-color="{accent_2}"/>
                            </linearGradient>
                        </defs>
                        <rect x="0" y="0" width="36" height="36" rx="10" fill="url(#brandGrad)"/>
                        <path d="M7 21 L12 21 L15 13 L19 26 L22 17 L24 21 L29 21"
                              fill="none" stroke="#FFFFFF" stroke-width="2.1"
                              stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>
                    </svg>
                </span>
            </div>
            <div>
                <div class="brand-name">{BRAND_NAME}</div>
                <div class="brand-tagline">{BRAND_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_theme_css()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Home",
    "Upload Data",
    "Data Cleaning",
    "Data Profiling",
    "Data Quality",
    "Analytics",
    "Visualizations",
    "AI Insights",
    "Machine Learning",
    "Reports",
]

with st.sidebar:
    render_brand_header()

    st.session_state["page"] = st.radio(
        "Navigate", PAGES, index=PAGES.index(st.session_state["page"]),
        label_visibility="collapsed",
    )

    st.divider()

    theme_choice = st.toggle("🌙 Dark Mode", value=(st.session_state["theme"] == "dark"))
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.divider()

    if st.session_state["df"] is not None:
        accent = "#7C6FFF" if st.session_state["theme"] == "dark" else "#6953F0"
        st.markdown(
            f"""
            <div style="font-size:12px; font-weight:600; color:{accent};
                        text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;">
                Active dataset
            </div>
            <div style="font-size:14px; font-weight:600; margin-bottom:2px;">
                {st.session_state['file_name']}
            </div>
            <div style="font-size:12.5px; opacity:0.65;">
                {st.session_state['df'].shape[0]:,} rows · {st.session_state['df'].shape[1]} columns
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No dataset loaded yet — start on the Upload Data page.")

    st.divider()
    st.caption(f"{BRAND_NAME} · built with Streamlit, Pandas & Plotly")


# ---------------------------------------------------------------------------
# Route to selected page
# ---------------------------------------------------------------------------
page = st.session_state["page"]

if page == "Home":
    from pages import home
    home.render()
elif page == "Upload Data":
    from pages import upload
    upload.render()
elif page == "Data Cleaning":
    from pages import cleaning
    cleaning.render()
elif page == "Data Profiling":
    from pages import profiling
    profiling.render()
elif page == "Data Quality":
    from pages import quality
    quality.render()
elif page == "Analytics":
    from pages import analytics
    analytics.render()
elif page == "Visualizations":
    from pages import visualization
    visualization.render()
elif page == "AI Insights":
    from pages import insights
    insights.render()
elif page == "Machine Learning":
    from pages import ml
    ml.render()
elif page == "Reports":
    from pages import reports
    reports.render()