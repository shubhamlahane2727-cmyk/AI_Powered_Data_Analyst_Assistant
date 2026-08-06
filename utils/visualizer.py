"""
visualizer.py
-------------
Plotly chart builders plus the Smart Chart Recommendation Engine.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Bold


def bar_chart(df, x, y=None, color=None, agg="sum", title=None):
    if y is None:
        data = df[x].value_counts().reset_index()
        data.columns = [x, "count"]
        fig = px.bar(data, x=x, y="count", color=x, template=TEMPLATE,
                     color_discrete_sequence=COLOR_SEQUENCE,
                     title=title or f"Count by {x}")
    else:
        grouped = df.groupby(x, as_index=False)[y].agg(agg)
        grouped = grouped.sort_values(y, ascending=False)
        fig = px.bar(grouped, x=x, y=y, color=color, template=TEMPLATE,
                     color_discrete_sequence=COLOR_SEQUENCE,
                     title=title or f"{y.title()} by {x.title()} ({agg})")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def line_chart(df, x, y, color=None, title=None):
    data = df.copy()
    if pd.api.types.is_datetime64_any_dtype(data[x]):
        data = data.sort_values(x)
    fig = px.line(data, x=x, y=y, color=color, template=TEMPLATE,
                  color_discrete_sequence=COLOR_SEQUENCE, markers=True,
                  title=title or f"{y.title()} over {x.title()}")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def pie_chart(df, names, values=None, title=None, top_n=8):
    if values is None:
        data = df[names].value_counts().reset_index()
        data.columns = [names, "count"]
        values_col = "count"
    else:
        data = df.groupby(names, as_index=False)[values].sum()
        values_col = values

    data = data.sort_values(values_col, ascending=False)
    if len(data) > top_n:
        top = data.head(top_n)
        other_sum = data.iloc[top_n:][values_col].sum()
        other_row = pd.DataFrame({names: ["Other"], values_col: [other_sum]})
        data = pd.concat([top, other_row], ignore_index=True)

    fig = px.pie(data, names=names, values=values_col, template=TEMPLATE,
                 color_discrete_sequence=COLOR_SEQUENCE,
                 title=title or f"Distribution of {names.title()}", hole=0.35)
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def histogram(df, x, bins=30, title=None):
    fig = px.histogram(df, x=x, nbins=bins, template=TEMPLATE,
                        color_discrete_sequence=COLOR_SEQUENCE,
                        title=title or f"Distribution of {x.title()}")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def box_plot(df, y, x=None, title=None):
    fig = px.box(df, x=x, y=y, template=TEMPLATE,
                 color=x, color_discrete_sequence=COLOR_SEQUENCE,
                 title=title or (f"{y.title()} by {x.title()}" if x else f"Box Plot of {y.title()}"))
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def scatter_plot(df, x, y, color=None, size=None, title=None, show_trendline=True):
    fig = px.scatter(df, x=x, y=y, color=color, size=size, template=TEMPLATE,
                      color_discrete_sequence=COLOR_SEQUENCE,
                      title=title or f"{y.title()} vs {x.title()}")

    # Manually computed linear trendline (avoids the statsmodels dependency
    # that plotly's built-in trendline="ols" option requires).
    if show_trendline and pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
        valid = df[[x, y]].dropna()
        if len(valid) >= 2:
            coeffs = np.polyfit(valid[x], valid[y], 1)
            x_range = np.linspace(valid[x].min(), valid[x].max(), 100)
            y_fit = np.polyval(coeffs, x_range)
            fig.add_scatter(
                x=x_range, y=y_fit, mode="lines", name="Trend",
                line=dict(color="#EF4444", width=2, dash="dash")
            )

    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


def heatmap(df, title="Correlation Heatmap"):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", template=TEMPLATE,
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title=title, aspect="auto")
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    return fig


@st.cache_data(show_spinner=False)
def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=np.number)
    return numeric_df.corr().round(3)


# ---------------------------------------------------------------------------
# Smart Chart Recommendation Engine
# ---------------------------------------------------------------------------

def recommend_chart(df, col_x, col_y=None):
    """
    Given one or two selected columns, recommend the best chart type and a
    short reason, following standard data-viz heuristics:
      numeric vs numeric    -> scatter plot
      categorical vs numeric-> bar chart
      categorical alone     -> pie / bar (distribution)
      numeric alone          -> histogram
      datetime + numeric     -> line chart
    """
    x_is_numeric = pd.api.types.is_numeric_dtype(df[col_x])
    x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[col_x])

    if col_y is None:
        if x_is_datetime:
            return "line", f"'{col_x}' is a date/time column — a line chart best shows it over time."
        if x_is_numeric:
            return "histogram", f"'{col_x}' is numeric — a histogram shows its distribution."
        nunique = df[col_x].nunique()
        if nunique <= 8:
            return "pie", f"'{col_x}' has few unique categories — a pie chart shows the breakdown clearly."
        return "bar", f"'{col_x}' has many categories — a bar chart shows counts more clearly than a pie."

    y_is_numeric = pd.api.types.is_numeric_dtype(df[col_y])
    y_is_datetime = pd.api.types.is_datetime64_any_dtype(df[col_y])

    if x_is_datetime and y_is_numeric:
        return "line", f"'{col_x}' is a date and '{col_y}' is numeric — a line chart reveals the trend over time."
    if y_is_datetime and x_is_numeric:
        return "line", f"'{col_y}' is a date and '{col_x}' is numeric — a line chart reveals the trend over time."
    if x_is_numeric and y_is_numeric:
        return "scatter", f"Both '{col_x}' and '{col_y}' are numeric — a scatter plot reveals their relationship."
    if x_is_numeric and not y_is_numeric:
        return "bar", f"'{col_y}' is categorical and '{col_x}' is numeric — a bar chart compares totals across categories."
    if y_is_numeric and not x_is_numeric:
        return "bar", f"'{col_x}' is categorical and '{col_y}' is numeric — a bar chart compares totals across categories."
    return "bar", "Both columns are categorical — a grouped bar chart works best for comparison."


def get_dataset_chart_suggestions(df, roles):
    """Generate a curated list of suggested visualizations for the dataset overview."""
    suggestions = []
    numeric = roles["numeric"]
    categorical = roles["categorical"]
    datetime_cols = roles["datetime"]

    if datetime_cols and numeric:
        suggestions.append({
            "chart": "line", "x": datetime_cols[0], "y": numeric[0],
            "reason": f"Track {numeric[0]} trends over {datetime_cols[0]}."
        })
    if categorical and numeric:
        suggestions.append({
            "chart": "bar", "x": categorical[0], "y": numeric[0],
            "reason": f"Compare total {numeric[0]} across {categorical[0]} categories."
        })
    if categorical:
        suggestions.append({
            "chart": "pie", "x": categorical[0], "y": None,
            "reason": f"See the distribution of records across {categorical[0]}."
        })
    if len(numeric) >= 2:
        suggestions.append({
            "chart": "scatter", "x": numeric[0], "y": numeric[1],
            "reason": f"Explore the relationship between {numeric[0]} and {numeric[1]}."
        })
    if numeric:
        suggestions.append({
            "chart": "histogram", "x": numeric[0], "y": None,
            "reason": f"Understand the distribution/spread of {numeric[0]}."
        })
    if len(numeric) >= 2:
        suggestions.append({
            "chart": "heatmap", "x": None, "y": None,
            "reason": "View correlations across all numeric columns at once."
        })
    return suggestions
