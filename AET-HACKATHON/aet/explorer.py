"""Interactive explorer for the AET quarter-hour price reconstruction.

Run from the project root:
    PYTHONPATH=/path/to/AET-HACKATHON \
    /Users/cristinafernandez-simalbernard/Library/Python/3.9/bin/streamlit run aet/explorer.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aet.baselines import get_baseline
from aet.basis import get_basis
from aet.cube import QuarterCube
from aet.history import HistoryStore
from aet.pipeline import ShapePipeline
from aet.volatility import ShapeRisk, risk_inputs

# ------------------------------------------------------------------ page config
st.set_page_config(
    page_title="Quarter-Hour Reconstruction — AET",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-label { font-size: 0.8rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data loading
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_pipeline():
    pipe, history = ShapePipeline.load("artifacts_official")
    risk = ShapeRisk.load("results/official_volatility/risk_reference.json")
    return pipe, history, risk


@st.cache_data(show_spinner="Loading assessment data...")
def load_assessment():
    pred = pd.read_csv("results/september_assessment/predictions.csv")
    truth = pd.read_csv("results/september_assessment/truth_15min.csv")
    hourly_files = sorted(
        Path("results/september_assessment").glob("*/test_hourly.csv")
    )
    if not hourly_files:
        raise FileNotFoundError(
            "No test_hourly.csv files found in dated assessment folders"
        )
    hourly = pd.concat(
        [pd.read_csv(path) for path in hourly_files],
        ignore_index=True
    )
    return pred, truth, hourly


# ------------------------------------------------------------------ helpers
RISK_COLOUR = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
RISK_EMOJI  = {"low": "🟢", "medium": "🟡", "high": "🔴"}
# Display reliability language (inverse of risk): low risk = high reliability
RISK_DISPLAY = {
    "low":     "HIGH RELIABILITY",
    "medium":  "MEDIUM RELIABILITY",
    "high":    "LOW RELIABILITY",
    "unknown": "UNKNOWN",
}
SIGNAL_LABEL = {
    "hourly_slope": "steep hourly ramp",
    "daily_std": "high daily price volatility",
    "historical_residual_magnitude": "historically volatile hour",
}


def compute_pchip(hourly_values: np.ndarray, hour: int) -> np.ndarray:
    """PCHIP baseline for one hour given a 24-value hourly array."""
    H = hourly_values.reshape(1, 1, 24)
    return get_baseline("pchip")(H)[0, 0, hour - 1]   # [4]


def compute_risk(pipe, history, risk, date_str: str,
                 hourly_df_day: pd.DataFrame, zone: str, hour: int):
    """Risk label, dominant signal and limited-history flag for one hour."""
    day = np.datetime64(date_str, "D")
    store = HistoryStore.build(
        history.before(day),
        pipe.config.baseline,
        get_basis(pipe.config.basis),
    )
    # Build H with all zones in the correct order [1, n_zones, 24]
    n_zones = len(pipe.zones)
    H_all = np.full((1, n_zones, 24), np.nan)
    for zi, z in enumerate(pipe.zones):
        z_rows = hourly_df_day[hourly_df_day["market_id"] == z].sort_values("hour_index")
        if len(z_rows) == 24:
            H_all[0, zi] = z_rows["hourly_price_eur_mwh"].values
    inputs = risk_inputs(np.array([day]), H_all, store, pipe.config)
    # Filter to the selected zone and hour
    mask = (inputs["market_id"] == zone) & (inputs["hour_index"] == hour)
    matches = inputs[mask]
    if matches.empty:
        return "unknown", "unknown", False
    frame = risk.predict(matches)
    if frame.empty:
        return "unknown", "unknown", False
    row = frame.iloc[0]
    return row["risk_label"], row["dominant_signal"], bool(row["limited_history"])


def valuation_error(prices: np.ndarray, actual: np.ndarray,
                    volumes: np.ndarray) -> float:
    return float(abs(np.dot(volumes, prices) - np.dot(volumes, actual)))


# ------------------------------------------------------------------ main app
def main():
    st.title("⚡ Quarter-Hour Price Reconstruction")
    st.caption(
        "AET Neural Wave 2026 · Reconstruct 15-minute profiles from hourly curves "
        "while preserving every hourly average exactly."
    )

    # load resources
    try:
        pipe, history, risk = load_pipeline()
        pred_df, truth_df, hourly_df = load_assessment()
    except FileNotFoundError as exc:
        st.error(
            f"Required file not found: {exc}\n\n"
            "Make sure you run from the project root with artifacts/ and "
            "results/september_assessment/ present."
        )
        st.stop()

    zones = sorted(hourly_df["market_id"].unique())

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Select")
        zone = st.selectbox("Zone", zones)
        dates = sorted(
            hourly_df[hourly_df["market_id"] == zone]["delivery_date"].unique()
        )
        date = st.selectbox("Date", dates)
        hour = st.slider("Hour of day", 1, 24, 12)

        st.divider()
        st.header("Delivery profile")
        st.caption(
            "Select how 1 MWh is delivered across the four quarters. "
            "An even split always shows zero valuation error because the "
            "hourly average is preserved exactly — uneven profiles reveal "
            "where shape reconstruction matters."
        )

        PROFILES = {
            "Early delivery (Q1 only)":  np.array([1.0, 0.0, 0.0, 0.0]),
            "Late delivery (Q4 only)":   np.array([0.0, 0.0, 0.0, 1.0]),
            "Front-heavy (Q1+Q2)":       np.array([0.5, 0.5, 0.0, 0.0]),
            "Back-heavy (Q3+Q4)":        np.array([0.0, 0.0, 0.5, 0.5]),
            "Even (reference)":          np.array([0.25, 0.25, 0.25, 0.25]),
        }

        profile_name = st.selectbox("Profile", list(PROFILES.keys()))
        volumes = PROFILES[profile_name]
        q1, q2, q3, q4 = volumes

        st.caption(
            "Valuation errors are illustrative — they use the selected profile "
            "applied to assessment prices, not AET's actual positions."
        )

    # ---------------------------------------------------------------- data for selection
    day_hourly = hourly_df[
        (hourly_df["market_id"] == zone) &
        (hourly_df["delivery_date"] == date)
    ].sort_values("hour_index")

    if day_hourly.empty:
        st.warning("No hourly data found for this zone and date.")
        st.stop()

    hourly_values = day_hourly["hourly_price_eur_mwh"].values   # [24]
    hourly_price  = float(hourly_values[hour - 1])

    day_pred = pred_df[
        (pred_df["market_id"] == zone) &
        (pred_df["delivery_date"] == date) &
        (pred_df["hour_index"] == hour)
    ].sort_values("quarter_index")

    day_truth = truth_df[
        (truth_df["market_id"] == zone) &
        (truth_df["delivery_date"] == date) &
        (truth_df["hour_index"] == hour)
    ].sort_values("quarter_index")

    if day_pred.empty or day_truth.empty:
        st.warning("Prediction or truth data not found for this selection.")
        st.stop()

    pred_prices  = day_pred["predicted_price_eur_mwh"].values   # [4]
    true_prices  = day_truth["price_eur_mwh"].values            # [4]
    flat_prices  = np.full(4, hourly_price)
    pchip_prices = compute_pchip(hourly_values, hour)           # [4]

    # ---------------------------------------------------------------- risk
    with st.spinner("Computing reliability indicator..."):
        try:
            # Pass full day hourly data for all zones so risk_inputs gets correct shape
            day_hourly_all = hourly_df[hourly_df["delivery_date"] == date]
            risk_label, dominant_signal, limited_history = compute_risk(
                pipe, history, risk, date, day_hourly_all, zone, hour
            )
        except Exception as e:
            st.warning(f"Risk indicator unavailable: {e}")
            risk_label, dominant_signal, limited_history = "unknown", "unknown", False

    # ---------------------------------------------------------------- layout
    # Row 1: risk badge + hourly price
    col_badge, col_price, col_spacer = st.columns([2, 2, 4])
    with col_badge:
        colour = RISK_COLOUR.get(risk_label, "#aaa")
        if risk_label == "low":
            signal_line = "Historical errors on comparable hours are low."
        elif risk_label == "medium":
            signal_line = f"Moderate signal: {SIGNAL_LABEL.get(dominant_signal, dominant_signal)}."
        else:
            signal_line = f"Caution: {SIGNAL_LABEL.get(dominant_signal, dominant_signal)}."

        st.markdown(
            f"<div style='background:{colour}22; border-left: 4px solid {colour}; "
            f"padding: 0.6rem 1rem; border-radius: 4px;'>"
            f"<b style='color:{colour}'>{RISK_EMOJI.get(risk_label,'')} "
            f"{RISK_DISPLAY.get(risk_label, risk_label.upper())}</b><br>"
            f"<span style='font-size:0.85rem; color:#555;'>{signal_line}</span>"
            + ("<br><span style='font-size:0.8rem;color:#e67e22'>⚠ Limited history "
               "(&lt; 3 comparable prior days)</span>" if limited_history else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    with col_price:
        st.metric("Hourly average (supplied)", f"€{hourly_price:.2f}/MWh")

    st.divider()

    # Row 2: chart + valuation
    col_chart, col_val = st.columns([3, 1])

    with col_chart:
        st.subheader("Reconstruction")

        # quarter-hour x positions within the selected hour
        q_times = [hour - 1 + o for o in (0.125, 0.375, 0.625, 0.875)]

        fig = go.Figure()

        # full daily hourly curve (background)
        fig.add_trace(go.Scatter(
            x=list(range(1, 25)),
            y=hourly_values,
            name="Hourly curve",
            line=dict(color="#adb5bd", width=1.5),
            hovertemplate="Hour %{x}: €%{y:.2f}/MWh<extra></extra>",
        ))

        # flat baseline
        fig.add_trace(go.Scatter(
            x=q_times, y=flat_prices,
            name="Flat assumption",
            mode="markers+lines",
            line=dict(color="#f39c12", dash="dot", width=2),
            marker=dict(size=7),
            hovertemplate="Q%{pointNumber+1}: €%{y:.2f}/MWh<extra>Flat</extra>",
        ))

        # PCHIP baseline
        fig.add_trace(go.Scatter(
            x=q_times, y=pchip_prices,
            name="PCHIP baseline",
            mode="markers+lines",
            line=dict(color="#9b59b6", dash="dash", width=2),
            marker=dict(size=7),
            hovertemplate="Q%{pointNumber+1}: €%{y:.2f}/MWh<extra>PCHIP</extra>",
        ))

        # model reconstruction — coloured by risk
        rec_colour = RISK_COLOUR.get(risk_label, "#3498db")
        fig.add_trace(go.Scatter(
            x=q_times, y=pred_prices,
            name=f"Model — {RISK_DISPLAY.get(risk_label, risk_label)}",
            mode="markers+lines",
            line=dict(color=rec_colour, width=3),
            marker=dict(size=10, symbol="circle"),
            hovertemplate="Q%{pointNumber+1}: €%{y:.2f}/MWh<extra>Model</extra>",
        ))

        # actual prices
        fig.add_trace(go.Scatter(
            x=q_times, y=true_prices,
            name="Actual prices",
            mode="markers+lines",
            line=dict(color="#2c3e50", dash="dash", width=2),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="Q%{pointNumber+1}: €%{y:.2f}/MWh<extra>Actual</extra>",
        ))

        # risk shading — coloured rectangle behind the selected hour
        risk_fill = {
            "low":     "rgba(46,204,113,0.10)",
            "medium":  "rgba(243,156,18,0.15)",
            "high":    "rgba(231,76,60,0.18)",
            "unknown": "rgba(150,150,150,0.10)",
        }
        fig.add_shape(
            type="rect",
            x0=hour - 1, x1=hour,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=risk_fill.get(risk_label, "rgba(150,150,150,0.10)"),
            line_width=0,
            layer="below",
        )
        # risk label annotation inside the shaded band
        fig.add_annotation(
            x=hour - 0.5, y=0.97,
            xref="x", yref="paper",
            text=f"{RISK_EMOJI.get(risk_label,'')} {RISK_DISPLAY.get(risk_label, risk_label.upper())}",
            showarrow=False,
            font=dict(size=11, color=RISK_COLOUR.get(risk_label, "#888"), family="monospace"),
            bgcolor="white",
            bordercolor=RISK_COLOUR.get(risk_label, "#888"),
            borderwidth=1,
            borderpad=3,
        )

        # hourly average reference line for the selected hour
        fig.add_shape(
            type="line",
            x0=hour - 1, x1=hour,
            y0=hourly_price, y1=hourly_price,
            line=dict(color="#e74c3c", dash="longdash", width=1),
        )
        fig.add_annotation(
            x=hour - 0.5, y=hourly_price,
            text=f"Hourly avg €{hourly_price:.1f}",
            showarrow=False, font=dict(size=10, color="#e74c3c"),
            yshift=10,
        )

        fig.update_layout(
            xaxis_title="Hour of day",
            yaxis_title="EUR/MWh",
            height=420,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quarter-by-quarter comparison table
        st.markdown("**Quarter-by-quarter prices (EUR/MWh)**")
        qdf = pd.DataFrame({
            "Quarter":        ["Q1", "Q2", "Q3", "Q4", "Average"],
            "Flat":           list(np.round(flat_prices, 2))  + [round(float(flat_prices.mean()), 2)],
            "PCHIP":          list(np.round(pchip_prices, 2)) + [round(float(pchip_prices.mean()), 2)],
            "Model":          list(np.round(pred_prices, 2))  + [round(float(pred_prices.mean()), 2)],
            "Actual":         list(np.round(true_prices, 2))  + [round(float(true_prices.mean()), 2)],
            "Flat error":     list(np.round(np.abs(flat_prices  - true_prices), 2)) + [round(float(np.abs(flat_prices  - true_prices).mean()), 2)],
            "PCHIP error":    list(np.round(np.abs(pchip_prices - true_prices), 2)) + [round(float(np.abs(pchip_prices - true_prices).mean()), 2)],
            "Model error":    list(np.round(np.abs(pred_prices  - true_prices), 2)) + [round(float(np.abs(pred_prices  - true_prices).mean()), 2)],
        })

        def highlight_best(row):
            """Highlight the model error cell green when it is the lowest error."""
            styles = [""] * len(row)
            if row.name < 4:  # skip the average row
                errors = {"Flat error": row["Flat error"],
                          "PCHIP error": row["PCHIP error"],
                          "Model error": row["Model error"]}
                best = min(errors, key=errors.get)
                col_idx = list(qdf.columns).index(best)
                styles[col_idx] = "background-color: #d4edda; font-weight: bold"
            return styles

        st.dataframe(
            qdf.style.apply(highlight_best, axis=1),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Errors are absolute (EUR/MWh). Green cell = lowest error for that quarter. "
            "Average row shows MAE across the four quarters. "
            "The hourly average is preserved exactly in all three models."
        )

    with col_val:
        st.subheader("Valuation")
        st.caption(f"{profile_name} · Q1={q1:.0%} Q2={q2:.0%} Q3={q3:.0%} Q4={q4:.0%}")

        actual_val = float(np.dot(volumes, true_prices))
        flat_err   = valuation_error(flat_prices, true_prices, volumes)
        pchip_err  = valuation_error(pchip_prices, true_prices, volumes)
        model_err  = valuation_error(pred_prices, true_prices, volumes)

        st.metric("Actual value", f"€{actual_val:.2f}/MWh")
        st.metric(
            "Flat error",
            f"€{flat_err:.2f}",
            delta=f"{flat_err:.2f} EUR/MWh off",
            delta_color="inverse",
        )
        st.metric(
            "PCHIP error",
            f"€{pchip_err:.2f}",
            delta=f"−{(flat_err - pchip_err):.2f} vs flat",
            delta_color="normal",
        )
        st.metric(
            "Model error",
            f"€{model_err:.2f}",
            delta=f"−{(flat_err - model_err):.2f} vs flat",
            delta_color="normal",
        )

        if flat_err > 0.01:
            improvement = (flat_err - model_err) / flat_err * 100
            colour = "#2ecc71" if improvement > 0 else "#e74c3c"
            st.markdown(
                f"<div style='background:{colour}22; border-left:4px solid {colour};"
                f"padding:0.5rem 0.8rem; border-radius:4px; margin-top:0.5rem;'>"
                f"<b style='color:{colour}'>{improvement:+.0f}% vs flat</b></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Row 3: overall results summary
    st.subheader("Overall results — September assessment")
    results = pd.DataFrame({
        "Model": ["Flat baseline", "PCHIP baseline", "Final model"],
        "MAE (EUR/MWh)": [10.90, 7.56, 5.04],
        "Improvement vs flat": ["—", "30.6%", "53.7%"],
        "Constraint violations": ["0", "0", "0"],
    })
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.caption(
        "Development MAE (8 monthly folds Jan–Aug 2026): flat 8.13 · PCHIP 5.62 · model 3.95 EUR/MWh. "
        "September MAE (locked assessment): flat 10.90 · PCHIP 7.56 · model 5.04 EUR/MWh. "
        "September was never used during development or model selection."
    )

    st.divider()

    # Row 4: risk indicator summary table
    with st.expander("Risk indicator validation (development folds)"):
        risk_table = pd.DataFrame({
            "Zone":      ["ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3"],
            "Low risk":  [1.52, 2.83, 2.04, 2.30],
            "Medium risk":[2.39, 3.81, 3.45, 4.04],
            "High risk": [5.93, 8.50, 5.75, 7.32],
            "High/Low":  ["3.9×", "3.0×", "2.8×", "3.2×"],
        })
        st.dataframe(risk_table, hide_index=True, use_container_width=True)
        st.caption(
            "MAE in EUR/MWh by risk label, averaged over development folds. "
            "The indicator uses only inputs available before the target date: "
            "hourly slope, daily volatility and historical residual magnitude. "
            "This is a track record indicator, not a calibrated probability."
        )


if __name__ == "__main__":
    main()