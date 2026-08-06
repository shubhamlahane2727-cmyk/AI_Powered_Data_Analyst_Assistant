"""
visualization.py
-----------------
Visualizations page: manual chart builder across all chart types, the
Smart Chart Recommendation Engine, and a lightweight Natural Language
Query system that maps simple questions to charts.
"""

import re

import streamlit as st

from utils.ui_helpers import require_data, page_header
from utils import profiler, visualizer


CHART_BUILDERS = {
    "Bar Chart": "bar",
    "Line Chart": "line",
    "Pie Chart": "pie",
    "Histogram": "histogram",
    "Box Plot": "box",
    "Scatter Plot": "scatter",
    "Heatmap / Correlation Matrix": "heatmap",
}


def _render_chart(chart_type, df, x=None, y=None, color=None):
    if chart_type == "bar":
        return visualizer.bar_chart(df, x, y, color=color)
    if chart_type == "line":
        return visualizer.line_chart(df, x, y, color=color)
    if chart_type == "pie":
        return visualizer.pie_chart(df, x, y)
    if chart_type == "histogram":
        return visualizer.histogram(df, x)
    if chart_type == "box":
        return visualizer.box_plot(df, y, x)
    if chart_type == "scatter":
        return visualizer.scatter_plot(df, x, y, color=color)
    if chart_type == "heatmap":
        return visualizer.heatmap(df)
    return None


def render():
    page_header("Visualizations", "Build interactive charts and get smart chart recommendations.", "📊")
    df = require_data()
    roles = profiler.detect_column_roles(df)

    tab_builder, tab_smart, tab_nlq = st.tabs(
        ["🛠️ Chart Builder", "✨ Smart Recommendations", "💬 Ask a Question"]
    )

    # -------------------------------------------------------------------
    # MANUAL CHART BUILDER
    # -------------------------------------------------------------------
    with tab_builder:
        chart_label = st.selectbox("Chart type", list(CHART_BUILDERS.keys()))
        chart_type = CHART_BUILDERS[chart_label]
        all_cols = df.columns.tolist()

        x_col = y_col = color_col = None

        if chart_type == "heatmap":
            st.caption("Heatmap uses all numeric columns automatically.")
        elif chart_type in ["pie", "histogram"]:
            x_col = st.selectbox("Column", all_cols, key="single_col")
            if chart_type == "pie":
                use_values = st.checkbox("Aggregate a numeric column instead of counting rows")
                if use_values and roles["numeric"]:
                    y_col = st.selectbox("Numeric column to sum", roles["numeric"], key="pie_values")
        else:
            c1, c2 = st.columns(2)
            with c1:
                x_default_idx = 0
                if chart_type in ["bar", "line"]:
                    # Prefer a sensible default: a datetime column for line
                    # charts, or a low-cardinality categorical for bar charts,
                    # rather than always defaulting to the very first column
                    # (which is often an ID/index column with one unique
                    # value per row — that produced the unreadable
                    # wall-of-bars chart seen previously).
                    preferred = roles["date_like"] if chart_type == "line" else roles["categorical"]
                    if preferred:
                        try:
                            x_default_idx = all_cols.index(preferred[0])
                        except ValueError:
                            x_default_idx = 0
                x_col = st.selectbox("X-axis", all_cols, index=x_default_idx, key="x_col")
            with c2:
                y_options = roles["numeric"] if chart_type in ["bar", "line", "box"] else all_cols
                y_col = st.selectbox("Y-axis", y_options, key="y_col") if y_options else None
            if chart_type in ["bar", "scatter", "line"]:
                color_choice = st.selectbox("Color by (optional)", ["None"] + all_cols, key="color_col")
                color_col = None if color_choice == "None" else color_choice

        if st.button("Generate Chart", type="primary"):
            # Guard against charts that would render as an unreadable wall
            # of bars/lines because the X-axis column is almost all unique
            # values (e.g. an ID column) rather than a real category.
            if chart_type in ["bar", "line"] and x_col is not None:
                n_unique = df[x_col].nunique()
                if n_unique > 40:
                    st.warning(
                        f"⚠️ '{x_col}' has {n_unique:,} unique values — a {chart_type} chart "
                        f"with this many categories will be unreadable. Consider picking a "
                        f"lower-cardinality column (like a category, region, or status field), "
                        f"or use a Histogram/Scatter Plot instead for high-cardinality numeric data."
                    )
            try:
                fig = _render_chart(chart_type, df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't build this chart with the selected columns: {e}")

    # -------------------------------------------------------------------
    # SMART CHART RECOMMENDATION ENGINE
    # -------------------------------------------------------------------
    with tab_smart:
        st.write("Pick one or two columns and get the recommended chart type with a reason.")
        all_cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            col_x = st.selectbox("Column 1", all_cols, key="smart_x")
        with c2:
            col_y_choice = st.selectbox("Column 2 (optional)", ["None"] + all_cols, key="smart_y")
        col_y = None if col_y_choice == "None" else col_y_choice

        if st.button("✨ Get Recommendation", type="primary"):
            chart_type, reason = visualizer.recommend_chart(df, col_x, col_y)
            st.info(f"**Recommended: {chart_type.title()} Chart** — {reason}")
            try:
                fig = _render_chart(chart_type, df, col_x, col_y)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't render the recommended chart: {e}")

        st.write("")
        st.markdown("##### Suggested Visualizations for This Dataset")
        suggestions = visualizer.get_dataset_chart_suggestions(df, roles)
        if not suggestions:
            st.write("Not enough column variety to generate suggestions.")
        for s in suggestions:
            with st.expander(f"{s['chart'].title()} — {s['reason']}"):
                try:
                    fig = _render_chart(s["chart"], df, s["x"], s["y"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write(f"Could not render: {e}")

    # -------------------------------------------------------------------
    # NATURAL LANGUAGE QUERY SYSTEM
    # -------------------------------------------------------------------
    with tab_nlq:
        st.write(
            "Type a simple question and the app will convert it into a chart. "
            "Examples: *show sales by region*, *top 10 products*, *revenue trend*, "
            "*highest profit category*, *monthly growth*."
        )
        query = st.text_input("Ask a question about your data")

        if query:
            fig, explanation = _handle_nl_query(query, df, roles)
            if fig is not None:
                st.success(explanation)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(explanation)


def _find_column(query, candidates):
    """Find the best-matching column name mentioned in the query text."""
    query_lower = query.lower()
    for col in candidates:
        if col.lower().replace("_", " ") in query_lower or col.lower() in query_lower:
            return col
    return None


def _handle_nl_query(query, df, roles):
    q = query.lower().strip()
    all_cols = df.columns.tolist()
    num_default = roles["revenue_like"][0] if roles["revenue_like"] else (roles["numeric"][0] if roles["numeric"] else None)

    # "top N <something>"
    top_n_match = re.search(r"top\s+(\d+)", q)
    if top_n_match:
        n = int(top_n_match.group(1))
        cat_col = _find_column(q, roles["categorical"]) or (roles["product_like"][0] if roles["product_like"] else (roles["categorical"][0] if roles["categorical"] else None))
        num_col = _find_column(q, roles["numeric"]) or num_default
        if cat_col and num_col:
            grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(n).reset_index()
            fig = visualizer.bar_chart(grouped, cat_col, num_col, title=f"Top {n} {cat_col} by {num_col}")
            return fig, f"Showing top {n} {cat_col} by total {num_col}."
        return None, "I couldn't find a suitable category and numeric column for a 'top N' query."

    # "trend" / "growth" / "over time"
    if any(k in q for k in ["trend", "growth", "over time", "monthly", "yearly"]):
        date_col = roles["date_like"][0] if roles["date_like"] else None
        num_col = _find_column(q, roles["numeric"]) or num_default
        if date_col and num_col:
            freq = "YE" if "yearly" in q or "year" in q else "ME"
            data = df[[date_col, num_col]].dropna().set_index(date_col).resample(freq)[num_col].sum().reset_index()
            fig = visualizer.line_chart(data, date_col, num_col, title=f"{num_col} Trend")
            return fig, f"Showing {num_col} trend over time."
        return None, "I need a date column and a numeric column to show a trend — none was found."

    # "by <category>" e.g. "sales by region"
    by_match = re.search(r"by\s+([a-zA-Z_ ]+)", q)
    if by_match or "highest" in q or "best" in q:
        cat_col = _find_column(q, roles["categorical"])
        if not cat_col and by_match:
            cat_col = _find_column(by_match.group(1), roles["categorical"])
        num_col = _find_column(q, roles["numeric"]) or num_default
        if cat_col and num_col:
            fig = visualizer.bar_chart(df, cat_col, num_col, title=f"{num_col} by {cat_col}")
            return fig, f"Showing {num_col} by {cat_col}."

    # fallback: distribution of any mentioned column
    mentioned_col = _find_column(q, all_cols)
    if mentioned_col:
        chart_type, reason = visualizer.recommend_chart(df, mentioned_col)
        fig = _render_chart(chart_type, df, mentioned_col)
        return fig, f"Showing {mentioned_col}: {reason}"

    return None, (
        "I couldn't map this question to a chart. Try mentioning a column name directly, "
        "or phrases like 'X by Y', 'top 10 X', or 'X trend'."
    )
