"""
insights_engine.py
-------------------
Rule-based "AI-powered" insights engine. No external LLM call is used here
(everything must run offline inside the app) — instead this module applies
a structured set of statistical + business heuristics to the dataframe and
turns the results into natural-language business insights, growth/trend
analysis, correlation analysis, and a full automated data story
(executive summary, key findings, recommendations).

Insight language adapts: if the dataset looks like a business dataset
(detected via utils.profiler.detect_column_roles), insights are framed in
business terms (revenue, region, product). Otherwise, insights are framed
generically in terms of "numeric column X" / "category Y".
"""

import numpy as np
import pandas as pd
import streamlit as st


def _fmt_num(n):
    if pd.isna(n):
        return "N/A"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:,.2f}"


def _pct(n):
    return f"{n:.1f}%"


# ---------------------------------------------------------------------------
# Core insight generators
# ---------------------------------------------------------------------------

def best_performing_category(df, cat_col, num_col, is_business=True):
    grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
    if grouped.empty:
        return None
    top_cat, top_val = grouped.index[0], grouped.iloc[0]
    total = grouped.sum()
    share = (top_val / total * 100) if total else 0

    if is_business:
        text = (
            f"**{top_cat}** is the top-performing {cat_col} segment, contributing "
            f"**{_fmt_num(top_val)}** in total {num_col} — **{_pct(share)}** of the overall total. "
            + ("This single segment significantly outperforms the rest, suggesting concentrated demand or pricing strength worth protecting and replicating elsewhere."
               if share > 35 else
               "Performance is fairly distributed, with this segment holding a moderate lead over others.")
        )
    else:
        text = (
            f"**{top_cat}** has the highest total {num_col} among {cat_col} groups, at "
            f"**{_fmt_num(top_val)}** (**{_pct(share)}** of the combined total). "
            + ("This group accounts for a disproportionately large share, which may be worth investigating further."
               if share > 35 else
               "Values are fairly evenly distributed across groups, with this one slightly ahead.")
        )

    return {
        "category_column": cat_col,
        "numeric_column": num_col,
        "top_category": top_cat,
        "top_value": top_val,
        "share_pct": share,
        "text": text,
    }


def regional_performance_summary(df, region_col, num_col, top_n=3):
    grouped = df.groupby(region_col)[num_col].sum().sort_values(ascending=False)
    if grouped.empty:
        return None
    total = grouped.sum()
    top = grouped.head(top_n)
    lines = [
        f"**{idx}** generates {_fmt_num(val)} ({_pct(val/total*100 if total else 0)} of total {num_col})"
        for idx, val in top.items()
    ]
    weakest = grouped.index[-1]
    weakest_val = grouped.iloc[-1]
    text = (
        f"Top {region_col} performers by {num_col}: " + "; ".join(lines) + ". "
        f"The weakest performer is **{weakest}** at {_fmt_num(weakest_val)} "
        f"({_pct(weakest_val/total*100 if total else 0)} of total), highlighting a potential opportunity "
        f"for targeted improvement or investigation into underlying causes."
    )
    return {"text": text, "ranking": grouped}


def growth_trend_analysis(df, date_col, num_col, freq="ME"):
    """Compute period-over-period growth (month-over-month by default)."""
    data = df[[date_col, num_col]].dropna().copy()
    if data.empty:
        return None
    data = data.set_index(date_col).resample(freq)[num_col].sum().reset_index()
    data = data[data[num_col] != 0]
    if len(data) < 2:
        return None
    data["growth_pct"] = data[num_col].pct_change() * 100

    latest = data.iloc[-1]
    prev = data.iloc[-2]
    overall_growth = ((data[num_col].iloc[-1] - data[num_col].iloc[0]) / abs(data[num_col].iloc[0]) * 100) \
        if data[num_col].iloc[0] != 0 else np.nan

    period_label = "month" if freq == "ME" else ("year" if freq == "YE" else "period")
    direction = "increased" if latest["growth_pct"] > 0 else "decreased"

    text = (
        f"{num_col} {direction} by **{_pct(abs(latest['growth_pct']))}** in the most recent {period_label} "
        f"compared to the previous one ({_fmt_num(prev[num_col])} → {_fmt_num(latest[num_col])}). "
        + (f"Over the full period analyzed, {num_col} has changed by **{_pct(overall_growth)}** overall."
           if not pd.isna(overall_growth) else "")
    )
    return {"text": text, "trend_data": data}


def correlation_insights(df, threshold=0.6):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return []
    corr = numeric_df.corr()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= threshold:
                pairs.append((cols[i], cols[j], val))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    insights = []
    for col1, col2, val in pairs[:6]:
        direction = "positive" if val > 0 else "negative"
        strength = "very strong" if abs(val) > 0.85 else "strong"
        insights.append({
            "col1": col1, "col2": col2, "corr": val,
            "text": (
                f"There is a **{strength} {direction} correlation** ({val:.2f}) between "
                f"**{col1}** and **{col2}**. "
                + ("As one increases, the other tends to increase as well — "
                   if val > 0 else
                   "As one increases, the other tends to decrease — ")
                + "this relationship may be worth investigating for causal drivers or used for forecasting."
            )
        })
    return insights


def outlier_alerts(df, roles):
    from utils.profiler import detect_outliers_iqr
    alerts = []
    for col in roles["numeric"]:
        outliers = detect_outliers_iqr(df[col])
        if len(outliers) and outliers.sum() > 0:
            pct = outliers.mean() * 100
            if pct > 2:
                alerts.append(
                    f"**{col}** shows {outliers.sum()} outlier value(s) ({_pct(pct)} of records), "
                    f"which may represent data errors, exceptional events, or high-value opportunities worth a closer look."
                )
    return alerts


def customer_behavior_patterns(df, roles):
    insights = []
    if roles["customer_like"] and roles["revenue_like"]:
        cust_col = roles["customer_like"][0]
        rev_col = roles["revenue_like"][0]
        grouped = df.groupby(cust_col)[rev_col].sum().sort_values(ascending=False)
        if len(grouped) > 0:
            total = grouped.sum()
            top_20_pct_count = max(1, int(len(grouped) * 0.2))
            top_20_share = grouped.head(top_20_pct_count).sum() / total * 100 if total else 0
            insights.append(
                f"The top 20% of {cust_col} entries account for **{_pct(top_20_share)}** of total {rev_col}, "
                + ("indicating a Pareto-like concentration — retention efforts for these key accounts should be a priority."
                   if top_20_share > 60 else
                   "indicating a relatively even spread of value across customers.")
            )
    return insights


def seasonal_trend_check(df, date_col, num_col):
    data = df[[date_col, num_col]].dropna().copy()
    if data.empty or data[date_col].dt.month.nunique() < 4:
        return None
    data["month"] = data[date_col].dt.month_name()
    monthly = data.groupby("month")[num_col].mean().sort_values(ascending=False)
    if monthly.empty:
        return None
    best_month, worst_month = monthly.index[0], monthly.index[-1]
    return {
        "text": (
            f"**{best_month}** tends to show the highest average {num_col}, while **{worst_month}** "
            f"shows the lowest — this seasonal pattern could inform inventory planning, staffing, "
            f"or marketing timing."
        )
    }


# ---------------------------------------------------------------------------
# Master insight orchestration
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_all_insights(df, roles):
    """
    Orchestrate all the above into one structured insights bundle.
    Adapts vocabulary depending on whether this looks like a business
    dataset, but the underlying statistical engine is fully generic.
    """
    insights = {
        "performance": [],
        "regional": None,
        "growth": None,
        "seasonal": None,
        "correlations": [],
        "outliers": [],
        "customer": [],
        "recommendations": [],
    }

    num_col = roles["revenue_like"][0] if roles["revenue_like"] else (roles["numeric"][0] if roles["numeric"] else None)
    is_business = roles["is_business_dataset"]

    # Best performing category (try product, then region, then any categorical)
    cat_candidates = roles["product_like"] or roles["region_like"] or roles["categorical"]
    if cat_candidates and num_col:
        for cat_col in cat_candidates[:2]:
            res = best_performing_category(df, cat_col, num_col, is_business=is_business)
            if res:
                insights["performance"].append(res)

    # Regional performance
    if roles["region_like"] and num_col:
        insights["regional"] = regional_performance_summary(df, roles["region_like"][0], num_col)

    # Growth trend
    if roles["date_like"] and num_col:
        insights["growth"] = growth_trend_analysis(df, roles["date_like"][0], num_col, freq="ME")
        insights["seasonal"] = seasonal_trend_check(df, roles["date_like"][0], num_col)

    # Correlations
    insights["correlations"] = correlation_insights(df)

    # Outliers
    insights["outliers"] = outlier_alerts(df, roles)

    # Customer behavior
    insights["customer"] = customer_behavior_patterns(df, roles)

    # Recommendations (synthesized from the above)
    recs = []
    if insights["performance"]:
        top = insights["performance"][0]
        if is_business:
            recs.append(
                f"Allocate additional resources or marketing investment toward **{top['top_category']}**, "
                f"which already leads in {top['numeric_column']} performance."
            )
        else:
            recs.append(
                f"**{top['top_category']}** stands out with the highest total {top['numeric_column']} — "
                f"worth investigating what drives this group's values."
            )
    if insights["regional"] is not None:
        recs.append(
            "Investigate the lowest-performing region for root causes (pricing, distribution, demand) "
            "and consider piloting strategies from top-performing regions there."
        )
    if insights["growth"] is not None and "trend_data" in insights["growth"]:
        latest_growth = insights["growth"]["trend_data"]["growth_pct"].iloc[-1]
        if latest_growth < 0:
            recs.append(
                "Recent performance has declined period-over-period — a deeper investigation into "
                "demand, pricing, or operational factors is recommended before the trend compounds."
            )
        else:
            recs.append(
                "Recent growth is positive — consider scaling current strategies while monitoring "
                "for diminishing returns."
            )
    if insights["correlations"]:
        c = insights["correlations"][0]
        recs.append(
            f"Leverage the relationship between **{c['col1']}** and **{c['col2']}** for forecasting "
            f"or as a lever to influence outcomes."
        )
    if insights["outliers"]:
        recs.append(
            "Review flagged outlier records — they may indicate data entry errors that should be "
            "corrected, or genuine high-impact events worth understanding."
        )
    if not recs:
        recs.append(
            "Dataset does not contain enough business-context columns (e.g. revenue, region, date) "
            "for tailored recommendations — consider enriching the dataset with these dimensions for deeper insight."
        )
    insights["recommendations"] = recs

    return insights


def generate_executive_summary(df, roles, quality_score):
    rows, cols = df.shape
    business_note = (
        "This dataset contains clear business dimensions (e.g. revenue, region, or product information), "
        "enabling commercially-relevant insight generation."
        if roles["is_business_dataset"] else
        "This dataset is primarily structured/operational data without explicit business dimensions; "
        "insights below focus on statistical patterns within the data."
    )
    return (
        f"This report analyzes a dataset of **{rows:,} records** across **{cols} columns**, "
        f"achieving an overall data quality score of **{quality_score['overall_score']}/100**. "
        f"{business_note} "
        f"The analysis below covers performance breakdowns, trends, correlations, and "
        f"actionable recommendations derived directly from the underlying data."
    )


def generate_key_findings(insights):
    findings = []
    for p in insights["performance"]:
        findings.append(p["text"])
    if insights["regional"]:
        findings.append(insights["regional"]["text"])
    if insights["growth"]:
        findings.append(insights["growth"]["text"])
    if insights["seasonal"]:
        findings.append(insights["seasonal"]["text"])
    for c in insights["correlations"][:3]:
        findings.append(c["text"])
    for c in insights["customer"]:
        findings.append(c)
    if not findings:
        findings.append("No strong patterns were detected beyond standard distributional statistics — see the EDA and Analytics pages for full detail.")
    return findings
