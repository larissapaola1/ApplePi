"""Quarter-hour price dashboard: the AET reconstruction on held-out September 2026.

Run from the AET-HACKATHON folder:
    python -m pip install -r requirements-explorer.txt
    PYTHONPATH=. python -m streamlit run aet/dashboard.py --server.address localhost
    # official CSV elsewhere:  ... --server.address localhost -- --train path/to/train_prices_15min.csv

Shows the frozen model trained through 31 Aug 2026 (artifacts_september_assessment) on 1-30 Sep 2026.
Each date is replayed with quarter-hours strictly before it. Nothing is trained, tuned or written to disk;
the risk reference is the frozen heuristic of `aet.volatility` (training slice through 31 Jul 2026).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from aet.cube import QuarterCube
from aet.history import HistoryStore
from aet.pipeline import ShapePipeline
from aet.volatility import ShapeRisk, risk_inputs

ASSESSMENT_MODEL = "artifacts_september_assessment"   # trained through 2026-08-31
FIRST_DAY, LAST_DAY = "2026-09-01", "2026-09-30"
RISK_TRAINING_END = "2026-07-31"                      # same slice as `aet.volatility diagnose`
TRAIN_CSV_CANDIDATES = (                              # AET-HACKATHON may sit inside or beside ApplePi
    "../Data/train_prices_15min.csv",
    "../ApplePi/Data/train_prices_15min.csv",
    "data/train_prices_15min.csv",
)

# Basis components of the learned correction, in model order, with plain-language meaning.
COMPONENTS = ("slope", "curvature", "zigzag")
PART_LABELS = {
    "smooth": "Smooth curve shape",
    "slope": "Slope: rising or falling inside the hour",
    "curvature": "Curvature: edges vs middle",
    "zigzag": "Zigzag: alternating detail",
}
SHORT_LABELS = {"smooth": "Smooth", "slope": "Slope", "curvature": "Curve", "zigzag": "Zigzag"}
RISK_REASONS = {   # the strongest of the three risk signals, in plain words
    "hourly_slope": "the hourly price changes sharply around this hour",
    "daily_std": "prices swing a lot during this day",
    "historical_residual_magnitude": "this hour has been hard to predict in recent days",
}

# Validated with the dataviz palette checker (adjacent and all-pairs, light and dark).
# Hourly price red, smooth curve blue, final model green; learned parts yellow / violet / magenta.
PALETTES = {
    "light": dict(hourly="#e34948", smooth="#2a78d6", model="#008300", slope="#eda100", curvature="#6250d6",
                  zigzag="#e87ba4", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb"),
    "dark": dict(hourly="#e66767", smooth="#3987e5", model="#008300", slope="#c98500", curvature="#9085e9",
                 zigzag="#d55181", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a", axis="#383835", surface="#1a1a19"),
}
RISK_COLOURS = {"high": "#fab219"}   # status palette: warning (red is taken by the hourly price)
RISK_BADGE = {"low": "🟢 Low risk", "medium": "🟡 Medium risk", "high": "⚠️ High risk", "unknown": "⚪ Unknown"}
FONT = dict(base=18, small=17)       # chart text sizes (px); page text is scaled in main()


# ------------------------------------------------------------------ compatibility helpers
def _streamlit_version() -> tuple[int, int]:
    major, minor = st.__version__.split(".")[:2]
    return int(major), int(minor)


# `width="stretch"` replaced `use_container_width` in newer Streamlit; requirements allow >= 1.40.
STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


def colours() -> dict:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return PALETTES["dark" if getattr(theme, "type", None) == "dark" else "light"]


# ------------------------------------------------------------------ data
def train_csv_path() -> Path:
    """--train after `--` on the streamlit command line, then $AET_TRAIN_CSV, then the usual locations."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train")
    explicit = parser.parse_known_args()[0].train or os.environ.get("AET_TRAIN_CSV")
    for candidate in ([explicit] if explicit else TRAIN_CSV_CANDIDATES):
        if Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(explicit or " or ".join(TRAIN_CSV_CANDIDATES))


def decompose_day(pipe: ShapePipeline, history: QuarterCube, day, H: np.ndarray):
    """H [zone, 24] -> (store, baseline [zone,24,4], {component: [zone,24,4]}, final [zone,24,4]).

    final = baseline + residual_scale * sum_j coefficient_j * basis_j, followed by the pipeline's
    exact hourly-mean projection; the components are the learned correction split per basis vector.
    """
    H1 = H[None]
    base = pipe.baseline(H1)
    store = HistoryStore.build(history.before(day), pipe.config.baseline, pipe.basis)
    X = pipe._features(np.array([day]), H1, base, store)[pipe.model.feature_names]
    coefficients = np.column_stack([booster.predict(X) for booster in pipe.model.boosters])
    scale = pipe.config.residual_scale
    parts = {name: (scale * coefficients[:, [j]] * pipe.basis[:, j][None, :]).reshape(H.shape + (4,))
             for j, name in enumerate(COMPONENTS)}
    final, _ = pipe.predict_day(day, H, history)
    return store, base[0], parts, final


@st.cache_data(show_spinner="Replaying the 30 September days (about a minute the first time)...")
def load_assessment(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per quarter (prices and parts) and one row per hour (risk), for every zone and date."""
    prices = pd.read_csv(path, dtype={"delivery_date": str})
    pipe, stored = ShapePipeline.load(ASSESSMENT_MODEL)
    cube = QuarterCube.from_long(prices, pipe.zones)
    history = stored.merge(cube)                     # predict_day only reads dates before each target
    risk = ShapeRisk.fit(QuarterCube.from_long(prices[prices.delivery_date <= RISK_TRAINING_END], pipe.zones),
                         pipe.config)
    days = cube.dates[(cube.dates >= np.datetime64(FIRST_DAY)) & (cube.dates <= np.datetime64(LAST_DAY))]
    zone_idx, hour_idx, quarter_idx = np.meshgrid(np.arange(len(pipe.zones)), np.arange(24), np.arange(4), indexing="ij")
    quarter_frames, hour_frames = [], []
    for day in days:
        actual = cube.values[np.searchsorted(cube.dates, day)]
        H = actual.mean(axis=-1)
        store, base, parts, final = decompose_day(pipe, history, day, H)
        start = pd.Timestamp(str(day))
        quarter_frames.append(pd.DataFrame({
            "delivery_date": str(day),
            "market_id": np.asarray(pipe.zones)[zone_idx.ravel()],
            "hour_index": hour_idx.ravel() + 1,
            "quarter_index": quarter_idx.ravel() + 1,
            "timestamp": start + pd.to_timedelta(hour_idx.ravel() * 60 + quarter_idx.ravel() * 15, unit="min"),
            "hourly": np.repeat(H[..., None], 4, axis=-1).ravel(),
            "actual": actual.ravel(),
            "baseline": base.ravel(),
            "model": final.ravel(),
            "smooth": (base - H[..., None]).ravel(),
            **{name: part.ravel() for name, part in parts.items()},
        }))
        hours = risk.predict(risk_inputs(np.array([day]), H[None], store, pipe.config))
        hour_frames.append(hours.assign(delivery_date=str(day))[
            ["delivery_date", "market_id", "hour_index", "risk_label", "dominant_signal", "limited_history"]])
    return pd.concat(quarter_frames, ignore_index=True), pd.concat(hour_frames, ignore_index=True)


def zone_mae(rows: pd.DataFrame, column: str) -> float:
    """Official metric for one zone: MAE per day, then the mean over days."""
    return float((rows[column] - rows.actual).abs().groupby(rows.delivery_date).mean().mean())


def hourly_errors(rows: pd.DataFrame) -> pd.DataFrame:
    """Model MAE per date and hour (mean over the four quarters)."""
    return (rows.assign(error=(rows.model - rows.actual).abs())
            .groupby(["delivery_date", "hour_index"], as_index=False).error.mean())


def hour_label(hour: int) -> str:
    return f"{hour - 1:02d}:00–{hour % 24:02d}:00" if hour < 24 else "23:00–24:00"


# ------------------------------------------------------------------ charts
def _style(fig: go.Figure, c: dict, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=50, b=40), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=c["ink2"], size=FONT["base"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=c["ink"], size=FONT["base"])),
        hoverlabel=dict(font=dict(color=c["ink"], size=FONT["base"])),
    )
    axis = dict(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False, automargin=True,
                tickfont=dict(size=FONT["small"]), title_font=dict(size=FONT["base"]))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def risk_runs(hours: pd.DataFrame, labels=("high",)) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Merge consecutive flagged hours into bands: fewer shapes, same picture."""
    runs, current = [], None
    for row in hours.sort_values(["delivery_date", "hour_index"]).itertuples():
        start = pd.Timestamp(row.delivery_date) + pd.Timedelta(hours=row.hour_index - 1)
        label = row.risk_label if row.risk_label in labels else None
        if current and label == current[2] and start == current[1]:
            current = (current[0], start + pd.Timedelta(hours=1), label)
            continue
        if current:
            runs.append(current)
        current = (start, start + pd.Timedelta(hours=1), label) if label else None
    if current:
        runs.append(current)
    return runs


def price_chart(rows: pd.DataFrame, hours: pd.DataFrame, show: dict, show_risk: bool, c: dict) -> go.Figure:
    fig = go.Figure()
    custom = np.column_stack([rows.delivery_date, rows.hour_index])
    if show["hourly"]:
        fig.add_scatter(x=rows.timestamp, y=rows.hourly, name="Hourly price (input)", line=dict(color=c["hourly"], width=2),
                        line_shape="hv", customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show["baseline"]:
        fig.add_scatter(x=rows.timestamp, y=rows.baseline, name="Smooth curve (baseline)",
                        line=dict(color=c["smooth"], width=2, dash="dash"), customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show["model"]:
        fig.add_scatter(x=rows.timestamp, y=rows.model, name="Final model", line=dict(color=c["model"], width=2),
                        customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show["actual"]:
        fig.add_scatter(x=rows.timestamp, y=rows.actual, name="Actual 15-min price", mode="markers",
                        marker=dict(color=c["ink"], size=6 if rows.delivery_date.nunique() <= 2 else 4,
                                    line=dict(color=c["surface"], width=1)),
                        customdata=custom, hovertemplate="%{y:.2f} €/MWh")
    if show_risk:
        for x0, x1, label in risk_runs(hours):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=RISK_COLOURS[label], opacity=0.2, line_width=0, layer="below")
        # September is more volatile than the reference period, so most hours are at least medium:
        # shading only high-risk hours keeps the chart readable (medium shows in the hour badge).
        fig.add_scatter(x=[None], y=[None], mode="markers", name="High-risk hour", hoverinfo="skip",
                        marker=dict(symbol="square", size=12, color=RISK_COLOURS["high"], opacity=0.5))
    # Fixed axes: computed from every series, so ticking layers on or off never rescales the chart.
    values = rows[["hourly", "baseline", "model", "actual"]].to_numpy()
    pad = 0.05 * (values.max() - values.min()) or 1.0
    first = pd.Timestamp(rows.delivery_date.min())
    last = pd.Timestamp(rows.delivery_date.max()) + pd.Timedelta(days=1)
    fig.update_layout(hovermode="x unified", yaxis_title="€/MWh", clickmode="event+select",
                      yaxis_range=[values.min() - pad, values.max() + pad], xaxis_range=[first, last])
    return _style(fig, c, 500)


def decomposition_chart(hour_rows: pd.DataFrame, c: dict) -> go.Figure:
    """One small waterfall per quarter: hourly price -> + each part -> prediction, with the actual price."""
    steps = ["smooth", *COMPONENTS]
    # Four quarters side by side; step labels are tilted so they stay large in the narrow panels.
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"Q{q} · :{(q - 1) * 15:02d}–:{q * 15 % 60:02d}" for q in range(1, 5)])
    for q, row in enumerate(hour_rows.sort_values("quarter_index").itertuples()):
        cell = dict(row=1, col=q + 1)
        level = row.hourly
        for step in steps:
            delta = getattr(row, step)
            fig.add_bar(x=[SHORT_LABELS[step]], y=[delta], base=[level], name=PART_LABELS[step],
                        marker=dict(color=c[step], cornerradius=4), legendgroup=step, showlegend=q == 0,
                        text=[f"{delta:+.1f}"], textposition="outside", constraintext="none",
                        textfont=dict(color=c["ink"], size=FONT["small"] - 2), cliponaxis=False,
                        hovertemplate=f"{PART_LABELS[step]}: %{{y:+.2f}} €/MWh<extra></extra>", **cell)
            level += delta
        fig.add_scatter(x=["Result"], y=[row.model], mode="markers", name="Final model", legendgroup="model",
                        showlegend=q == 0, marker=dict(symbol="diamond", size=14, color=c["model"]),
                        hovertemplate="Prediction: %{y:.2f} €/MWh<extra></extra>", **cell)
        fig.add_scatter(x=["Result"], y=[row.actual], mode="markers", name="Actual 15-min price", legendgroup="actual",
                        showlegend=q == 0, marker=dict(symbol="circle", size=11, color=c["ink"],
                                                       line=dict(color=c["surface"], width=2)),
                        hovertemplate="Actual: %{y:.2f} €/MWh<extra></extra>", **cell)
    fig.update_annotations(font=dict(color=c["ink2"], size=FONT["base"]))   # quarter titles
    for q, row in enumerate(hour_rows.sort_values("quarter_index").itertuples()):
        # Added after the traces: plotly skips shapes on subplots that are still empty.
        label = dict(annotation_text="hourly price", annotation_position="bottom left",
                     annotation_font=dict(color=c["hourly"], size=FONT["small"])) if q == 0 else {}
        fig.add_hline(y=row.hourly, line=dict(color=c["hourly"], width=2), layer="below", row=1, col=q + 1, **label)
    values = np.r_[hour_rows.hourly, hour_rows.model, hour_rows.actual, hour_rows.baseline]
    pad = max(2.0, 0.25 * (values.max() - values.min()))
    fig.update_yaxes(range=[values.min() - pad, values.max() + pad])
    fig.update_yaxes(title="€/MWh", row=1, col=1)
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig = _style(fig, c, 560)
    fig.update_layout(margin=dict(t=40, b=40), legend=dict(yanchor="top", y=-0.22))
    fig.update_xaxes(tickangle=-40)
    return fig


# ------------------------------------------------------------------ page
def sidebar(quarters: pd.DataFrame):
    first, last = pd.Timestamp(FIRST_DAY).date(), pd.Timestamp(LAST_DAY).date()
    with st.sidebar:
        zone = st.selectbox("Zone", sorted(quarters.market_id.unique()))
        picked = st.date_input("Dates", value=(first, first), min_value=first, max_value=last,
                               help="Pick one day, or a start and an end day. Only September 2026 is held out.")
        picked = picked if isinstance(picked, (tuple, list)) else (picked,)
        start, end = (picked[0], picked[-1]) if picked else (first, first)
        st.subheader("Show")
        show = {
            "hourly": st.checkbox("Hourly price (input)", value=True),
            "actual": st.checkbox("Actual 15-min prices", value=True),
            "baseline": st.checkbox("Smooth curve (baseline)", value=False),
            "model": st.checkbox("Final model", value=True),
        }
        show_risk = st.toggle("Highlight high-risk hours", value=True)
    return zone, pd.Timestamp(start), pd.Timestamp(end), show, show_risk


def score_cards(rows: pd.DataFrame, hours: pd.DataFrame) -> None:
    flat, smooth, model = (zone_mae(rows, col) for col in ("hourly", "baseline", "model"))
    st.markdown("<p style='font-size:1.2rem; margin-bottom:0.2rem'><b>Average error per 15-minute price</b> "
                "(lower is better)</p>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Flat hourly price", f"{flat:.2f} €/MWh",
                   help="The hourly price repeated for all four quarters: the reference to beat (official metric).")
    cols[1].metric("Smooth curve", f"{smooth:.2f} €/MWh", f"{(smooth / flat - 1) * 100:.0f}% vs flat", delta_color="inverse")
    cols[2].metric("Final model", f"{model:.2f} €/MWh", f"{(model / flat - 1) * 100:.0f}% vs flat", delta_color="inverse")
    cols[3].metric("High-risk hours", f"{int((hours.risk_label == 'high').sum())} of {len(hours)}",
                   help="Hours where the model deserves extra caution (heuristic).")


def pick_hour(rows: pd.DataFrame, dates: list[str], event) -> tuple[str, int]:
    """Hour explained below: a clicked chart point wins, otherwise keep or default to the largest correction."""
    points = list(getattr(getattr(event, "selection", None), "points", []) or [])
    clicked = next((p.get("customdata") for p in points if p.get("customdata") is not None), None)
    if clicked is not None and tuple(clicked) != st.session_state.get("last_click"):
        st.session_state.last_click = tuple(clicked)
        st.session_state.explain_date, st.session_state.explain_hour = str(clicked[0]), int(clicked[1])
    if st.session_state.get("explain_date") not in dates:
        learned = rows[list(COMPONENTS)].abs().sum(axis=1).groupby([rows.delivery_date, rows.hour_index]).mean()
        best_date, best_hour = learned.idxmax()
        st.session_state.explain_date, st.session_state.explain_hour = str(best_date), int(best_hour)
    left, right = st.columns([1, 2])
    date = left.selectbox("Day", dates, key="explain_date")
    hour = right.select_slider("Hour", options=list(range(1, 25)), key="explain_hour", format_func=hour_label)
    return date, hour


def explain_reading(hour_rows: pd.DataFrame) -> None:
    error = (hour_rows.model - hour_rows.actual).abs().mean()
    st.markdown("**How to read it.** Each 15-minute price starts from the hourly price (red line). "
                f"The bars add four parts to it, and the result (◆) is compared with the actual price (●). "
                f"Error this hour: {error:.2f} €/MWh.")


def explain_risk(zone, start, end, rows, hours) -> None:
    st.subheader("Risk: how much to trust each hour")
    st.markdown(
        "Every hour of every day gets its own label, for the selected zone, using only what is known before that "
        "day's 15-minute prices: the day's hourly prices and the previous days. Three signals are combined:\n"
        "1. **Hourly ramp:** how fast the hourly price changes around this hour.\n"
        "2. **Day volatility:** how much the 24 hourly prices swing during the day.\n"
        "3. **Recent difficulty:** how far the actual 15-minute prices moved away from the smooth curve at this hour "
        "over the previous 7 and 28 days.")
    period = f"{start:%d %b}" + (f" – {end:%d %b}" if end > start else "")
    st.markdown(f"**High-risk hours · {zone} · {period}**")
    risky = hours[hours.risk_label == "high"]
    if risky.empty:
        st.markdown("🟢 No high-risk hours in this selection.")
        return
    errors = hourly_errors(rows).set_index(["delivery_date", "hour_index"]).error
    st.dataframe(pd.DataFrame({
        "Day": risky.delivery_date, "Hour": risky.hour_index.map(hour_label), "Risk": "⚠️ High",
        "Main reason": risky.dominant_signal.map(RISK_REASONS),
        "Model error (€/MWh)": [round(errors.loc[(d, h)], 2) for d, h in zip(risky.delivery_date, risky.hour_index)],
    }), hide_index=True, **STRETCH)


def main() -> None:
    st.set_option("client.toolbarMode", "minimal")   # no Deploy button or developer menu
    st.set_page_config(page_title="Quarter-hour prices", page_icon="⚡", layout="wide")
    try:
        path = str(train_csv_path())
    except FileNotFoundError as exc:
        st.error(f"Official training CSV not found: {exc}. Start the app with `-- --train PATH`.")
        st.stop()
    quarters, hours_all = load_assessment(path)
    c = colours()

    # Streamlit sizes text in rem, so a larger root font scales the whole page.
    # Also hide the toolbar, menu and "File change: Rerun / Always rerun" prompt (fallback for older Streamlit).
    st.markdown("<style>html { font-size: 18px; }"
                "[data-testid='stMainMenu'], [data-testid='stAppDeployButton'], [data-testid='stToolbarActions'],"
                "[data-testid='stStatusWidget'] { display: none !important; }"
                # Score cards: larger, bolder labels and a larger "% vs flat" indicator.
                "[data-testid='stMetricLabel'], [data-testid='stMetricLabel'] p { font-size: 1.15rem !important;"
                " font-weight: 600 !important; }"
                "[data-testid='stMetricDelta'], [data-testid='stMetricDelta'] div { font-size: 1.2rem !important; }"
                "[data-testid='stMetricDelta'] svg { width: 1.3rem; height: 1.3rem; }</style>", unsafe_allow_html=True)
    st.title("Quarter-hour price reconstruction")
    st.markdown("<p style='font-size:1.5rem; opacity:0.8; margin-top:-0.5rem'>"
                "From 24 hourly prices to 96 quarter-hour prices.</p>", unsafe_allow_html=True)

    zone, start, end, show, show_risk = sidebar(quarters)
    in_range = lambda frame: frame[(frame.market_id == zone) &
                                   frame.delivery_date.between(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}")]
    rows, hours = in_range(quarters), in_range(hours_all)
    dates = sorted(rows.delivery_date.unique())

    score_cards(rows, hours)
    st.subheader(f"{zone} · {start:%d %b}" + (f" – {end:%d %b %Y}" if end > start else f" {start:%Y}"))
    event = st.plotly_chart(price_chart(rows, hours, show, show_risk, c), key="price_chart",
                            on_select="rerun", selection_mode="points", **STRETCH)
    st.caption("Tip: click a point to explain that hour below.")

    st.divider()
    st.subheader("How one prediction is built")
    st.markdown("<p style='font-size:1.15rem; opacity:0.85'>The predicted price is built from the hourly price plus "
                "the smooth curve shape and three learned shape vectors (slope, curvature and zigzag). Each one "
                "adds its own contribution to every 15-minute price.</p>", unsafe_allow_html=True)
    date, hour = pick_hour(rows, dates, event)
    hour_rows = rows[(rows.delivery_date == date) & (rows.hour_index == hour)]
    hour_risk = hours[(hours.delivery_date == date) & (hours.hour_index == hour)]
    label = hour_risk.risk_label.iloc[0] if len(hour_risk) else "unknown"
    reason = RISK_REASONS.get(hour_risk.dominant_signal.iloc[0], "") if label in ("medium", "high") else ""
    st.markdown(f"**{date} · {hour_label(hour)}** · {RISK_BADGE[label]}" + (f" - {reason}" if reason else ""))
    st.plotly_chart(decomposition_chart(hour_rows, c), key="decomposition", **STRETCH)
    explain_reading(hour_rows)

    st.divider()
    explain_risk(zone, start, end, rows, hours)



if __name__ == "__main__":
    main()
