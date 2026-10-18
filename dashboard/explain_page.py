"""Understanding the Model page: why the model predicts what it does.

1. What the model relies on overall: split gain of the three LightGBM models, grouped into plain categories.
2. One prediction, step by step: hourly price -> smooth curve -> slope -> curvature -> zigzag -> rules -> result.
3. When to trust it: the risk labels of the selected day.

Zone, day and hour are shared with the Model Results page: clicking a point there opens this page with that
point selected. Same frozen September model as Model Results; nothing is trained or written to disk.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import (ASSESSMENT_MODEL, COMPONENTS, FONT, MODEL_ID, MODEL_LABEL, RISK_BADGE, RISK_REASONS, STRETCH,
                             _style, colours, decomposition_chart, explain_reading, explain_risk, hour_label,
                             inject_style, load_assessment, require_assessment_model, train_csv_path, zone_name)
from aet.pipeline import ShapePipeline

GROUPS = {   # short names: shown as chart labels and in the table
    "shape": "Curve shape around the hour",
    "day": "Price level & day's swing",
    "history": "Same hour, recent days",
    "zones": "Other zones, same day",
    "calendar": "Calendar & zone",
}
COMPONENT_WORDS = {"1": "slope", "2": "curvature", "3": "zigzag"}
PLAIN = {
    "slope_m1": "Change from the previous hour", "slope_p1": "Change to the next hour",
    "slope_m2": "Change two hours before", "slope_p2": "Change two hours after",
    "curvature": "Curvature around the hour", "level": "Hourly price",
    "relative_level": "Hourly price vs the day's average", "day_mean": "Day's average price",
    "day_range": "Day's swing (highest − lowest hour)", "day_std": "Day's volatility",
    "day_negative_hours": "Negative hours in the day", "hour": "Hour of the day", "zone": "Zone",
    "weekday": "Day of the week", "weekend": "Weekend", "season_sin": "Season", "season_cos": "Season",
    "n7": "Days of history available (7 d)", "n28": "Days of history available (28 d)",
}


def feature_group(name: str) -> str:
    if name.startswith(("slope_", "curvature", "base_c")):
        return GROUPS["shape"]
    if name in ("level", "relative_level") or name.startswith("day_"):
        return GROUPS["day"]
    if name.startswith(("lag", "mean7", "mean28")) or name in ("n7", "n28"):
        return GROUPS["history"]
    if re.match(r"z\d_", name):
        return GROUPS["zones"]
    return GROUPS["calendar"]


def plain_name(name: str) -> str:
    if name in PLAIN:
        return PLAIN[name]
    if m := re.fullmatch(r"base_c(\d)", name):
        return f"Smooth curve's {COMPONENT_WORDS[m[1]]}"
    if m := re.fullmatch(r"(lag1|lag7|mean7|mean28)_c(\d)", name):
        when = {"lag1": "Yesterday's", "lag7": "Same weekday last week:", "mean7": "Last 7 days' average",
                "mean28": "Last 28 days' average"}[m[1]]
        return f"{when} {COMPONENT_WORDS[m[2]]} at this hour"
    if m := re.fullmatch(r"z(\d)_(spread|slope_m1|slope_p1)", name):
        what = {"spread": "price gap to", "slope_m1": "change from previous hour in",
                "slope_p1": "change to next hour in"}[m[2]]
        return f"{what.capitalize()} Zone {m[1]}"
    return name


@st.cache_data(show_spinner=False)
def feature_importance() -> pd.DataFrame:
    """Share of the total split gain per feature, for each of the three models (slope, curvature, zigzag)."""
    pipe, _ = ShapePipeline.load(ASSESSMENT_MODEL)
    shares = {}
    for name, booster in zip(COMPONENTS, pipe.model.boosters):
        gain = pd.Series(booster.feature_importance(importance_type="gain"), index=booster.feature_name())
        shares[name] = gain / gain.sum() * 100
    table = pd.DataFrame(shares).fillna(0)
    table["average"] = table[list(COMPONENTS)].mean(axis=1)
    table["group"] = [feature_group(f) for f in table.index]
    table["feature"] = [plain_name(f) for f in table.index]
    return table.sort_values("average", ascending=False)


# ------------------------------------------------------------------ sections
def what_it_relies_on(c: dict) -> None:
    st.subheader("What the model relies on")
    st.markdown("<div class='section-sub'>The learned correction is three small decision-tree models (LightGBM), one "
                "per shape: <b>slope</b> (rising or falling inside the hour), <b>curvature</b> (edges vs middle) and "
                "<b>zigzag</b> (alternating detail). This is how much each one leans on each kind of information."
                "</div>", unsafe_allow_html=True)
    table = feature_importance()
    groups = table.groupby("group")[[*COMPONENTS, "average"]].sum().sort_values("average")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        fig = go.Figure()
        for comp in COMPONENTS:
            fig.add_bar(y=groups.index, x=groups[comp], orientation="h", name=f"{comp.capitalize()} model",
                        marker=dict(color=c[comp], cornerradius=4), text=[f"{v:.0f}%" for v in groups[comp]],
                        textposition="outside", textfont=dict(color=c["ink"], size=FONT["small"] - 3),
                        cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% of the decisions' gain<extra></extra>")
        fig.update_layout(barmode="group", xaxis_title="share of the model's total gain (%)", bargap=0.25)
        fig = _style(fig, c, 470)
        fig.update_layout(margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, **STRETCH)
    with right:
        top = table.head(12)
        st.markdown("**The 12 most used inputs**")
        st.dataframe(top[["feature", "group", "average"]].rename(columns={
            "feature": "Input", "group": "Kind", "average": "Share (avg of 3 models)"}),
            hide_index=True, height=465, **STRETCH, column_config={
                "Input": st.column_config.TextColumn(width="medium"),
                "Kind": st.column_config.TextColumn(width="medium"),
                "Share (avg of 3 models)": st.column_config.ProgressColumn(
                    "Share", format="%.1f%%", min_value=0, max_value=float(top.average.max()), width="small")})
    split = " · ".join(f"{name.lower()} <b>{share:.0f}%</b>" for name, share in groups.average[::-1].items())
    st.markdown(f"<div class='takeaway'><b>In short:</b> no single kind of information dominates: {split}. The smooth "
                "curve already follows the slope between hours, so the trees mainly learn <i>how strongly</i> to bend it, "
                "using the price level, the day's swing and how this hour behaved on recent days. Other zones help a "
                "little; the calendar barely matters.</div>", unsafe_allow_html=True)
    st.caption("Share of split gain in the frozen September model. It shows what the model uses, not cause and effect.")


def ensure_hour(rows: pd.DataFrame, dates: list[str]) -> None:
    """Default to the hour with the largest learned correction, unless one is already selected."""
    if st.session_state.get("explain_date") not in dates:
        learned = rows[list(COMPONENTS)].abs().sum(axis=1).groupby([rows.delivery_date, rows.hour_index]).mean()
        best_date, best_hour = learned.idxmax()
        st.session_state.explain_date, st.session_state.explain_hour = str(best_date), int(best_hour)
    st.session_state.setdefault("explain_hour", 1)


def one_prediction(quarters: pd.DataFrame, hours_all: pd.DataFrame, c: dict):
    st.subheader("One prediction, step by step")
    st.markdown("<div class='section-sub'>Pick any zone, day and hour, or click a point on the Model Results page. "
                "The prediction starts from the hourly price, adds the smooth curve shape, then the three learned "
                "shapes; fixed rules then align zones with (nearly) equal hourly prices and keep each quarter on the "
                "same side of zero as its hour.</div>", unsafe_allow_html=True)
    zones = sorted(quarters.market_id.unique())
    st.session_state.setdefault("explain_zone", zones[0])
    with st.container(border=True):
        zone_col, day_col, hour_col = st.columns([1.3, 1, 2], gap="medium", vertical_alignment="bottom")
        zone = zone_col.segmented_control("Zone", zones, key="explain_zone", format_func=zone_name) or zones[0]
        rows, hours = quarters[quarters.market_id == zone], hours_all[hours_all.market_id == zone]
        dates = sorted(rows.delivery_date.unique())
        ensure_hour(rows, dates)
        date = day_col.selectbox("Day", dates, key="explain_date",
                                 format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"))
        hour = hour_col.select_slider("Hour", options=list(range(1, 25)), key="explain_hour", format_func=hour_label)
    hour_rows = rows[(rows.delivery_date == date) & (rows.hour_index == hour)]
    hour_risk = hours[(hours.delivery_date == date) & (hours.hour_index == hour)]
    left, right = st.columns([1, 3], gap="large")
    with left:
        label = hour_risk.risk_label.iloc[0] if len(hour_risk) else "unknown"
        reason = RISK_REASONS.get(hour_risk.dominant_signal.iloc[0], "") if label in ("medium", "high") else ""
        st.markdown(f"**{RISK_BADGE[label]}**" + (f"  \n{reason.capitalize()}." if reason else ""))
        explain_reading(hour_rows)
    with right:
        st.markdown(f"**{zone_name(zone)} · {pd.Timestamp(date):%a %d %b %Y} · {hour_label(hour)}**")
        st.plotly_chart(decomposition_chart(hour_rows, c), key="decomposition", **STRETCH)
    return zone, date, rows, hours


def render() -> None:
    import nav   # imported late: nav imports this module
    c = colours()
    inject_style(c)
    st.markdown(f"<style>.takeaway {{ border-left: 4px solid {c['model']}; padding: 0.7rem 1rem; margin: 0.6rem 0;"
                f" border-radius: 0 10px 10px 0; font-size: 1.02rem; color: {c['ink']}; }}</style>",
                unsafe_allow_html=True)
    st.session_state.last_click = None   # a later click on Model Results always opens this page again
    try:
        path = str(train_csv_path())
    except FileNotFoundError as exc:
        st.error(f"Official training CSV not found: {exc}. Start the app with `-- --train PATH`.")
        st.stop()
    require_assessment_model()
    quarters, hours_all = load_assessment(path)
    st.markdown("<div class='hero-title'>Understanding the model</div>"
                f"<div class='hero-sub'>Why the {MODEL_LABEL.lower()} looks the way it does.</div>"
                f"<div class='hero-meta'>{MODEL_ID} · same model as Model Results · "
                "explained on held-out September 2026</div>", unsafe_allow_html=True)
    what_it_relies_on(c)
    st.divider()
    zone, date, rows, hours = one_prediction(quarters, hours_all, c)
    st.divider()
    day = pd.Timestamp(date)
    explain_risk(zone_name(zone), day, day, rows[rows.delivery_date == date], hours[hours.delivery_date == date])
    st.divider()
    st.page_link(nav.MODEL_RESULTS, label="Back to Model Results", icon=":material/arrow_back:")
