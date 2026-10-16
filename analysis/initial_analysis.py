"""Price-only exploratory analysis. September targets stay out of EDA by default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["delivery_date", "market_id", "hour_index", "quarter_index"]
HOUR_KEYS = KEYS[:3]
ZONES = [f"ZONE_{i}" for i in range(4)]
EXCLUDED = pd.to_datetime(["2025-10-26", "2026-03-29"])
COLORS = ["#2864a0", "#e68a2e", "#258a70", "#9258a5"]


def load_and_check(path: Path) -> tuple[pd.DataFrame, dict]:
    """Check the complete official file, without calculating holdout price statistics."""
    df = pd.read_csv(path)
    expected_columns = KEYS + ["price_eur_mwh"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}; got {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("Missing values in the input file")
    df["delivery_date"] = pd.to_datetime(df.delivery_date, format="%Y-%m-%d", errors="raise")
    for col in ["hour_index", "quarter_index", "price_eur_mwh"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values in {col}")
    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate observation keys")
    expected_dates = pd.date_range("2025-10-01", "2026-09-30").difference(EXCLUDED)
    expected = pd.MultiIndex.from_product(
        [expected_dates, ZONES, range(1, 25), range(1, 5)], names=KEYS
    )
    actual = pd.MultiIndex.from_frame(df[KEYS])
    missing, unexpected = expected.difference(actual), actual.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f"Incomplete/invalid grid: {len(missing)} missing, {len(unexpected)} unexpected keys")
    df = df.sort_values(KEYS).reset_index(drop=True)
    checks = {
        "rows": len(df), "zones": ZONES, "days_per_zone": len(expected_dates),
        "start": "2025-10-01", "end": "2026-09-30", "missing_values": 0,
        "duplicate_keys": 0, "finite_prices": True, "complete_expected_grid": True,
        "dst_excluded_dates": [d.strftime("%Y-%m-%d") for d in EXCLUDED],
    }
    return df, checks


def centre_predictions(hourly: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Return four prices per hour with exactly the supplied average (float tolerance)."""
    raw = np.asarray(raw, dtype=float)
    hourly = np.asarray(hourly, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4 or hourly.shape != (len(raw),):
        raise ValueError("Expected hourly shape (n,) and raw shape (n, 4)")
    if not np.isfinite(raw).all() or not np.isfinite(hourly).all():
        raise ValueError("Predictions and hourly values must be finite")
    return hourly[:, None] + raw - raw.mean(axis=1, keepdims=True)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["hourly_price"] = df.groupby(HOUR_KEYS).price_eur_mwh.transform("mean")
    df["deviation"] = df.price_eur_mwh - df.hourly_price
    df["absolute_deviation"] = df.deviation.abs()
    df["weekday"] = df.delivery_date.dt.dayofweek
    df["is_weekend"] = df.weekday.ge(5)
    df["month"] = df.delivery_date.dt.to_period("M").astype(str)
    hourly = df.groupby(HOUR_KEYS).agg(
        hourly_price=("hourly_price", "first"),
        within_hour_range=("price_eur_mwh", lambda s: s.max() - s.min()),
        flat_mae=("absolute_deviation", "mean"),
    ).reset_index()
    daily_group = hourly.groupby(["delivery_date", "market_id"])
    hourly["previous_hour"] = daily_group.hourly_price.shift(1)
    hourly["next_hour"] = daily_group.hourly_price.shift(-1)
    hourly["central_slope"] = (hourly.next_hour - hourly.previous_hour) / 2
    hourly["curvature"] = hourly.next_hour - 2 * hourly.hourly_price + hourly.previous_hour
    wide = df.pivot(index=HOUR_KEYS, columns="quarter_index", values="deviation")
    hourly = hourly.merge((wide[4] - wide[1]).rename("q4_minus_q1").reset_index(), on=HOUR_KEYS)
    return df, hourly


def lag_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Exact calendar-date joins, deliberately avoiding shifts over missing DST days."""
    rows = []
    for lag in [1, 7]:
        previous = df[KEYS + ["deviation"]].copy()
        previous["delivery_date"] += pd.Timedelta(days=lag)
        previous = previous.rename(columns={"deviation": "lag_deviation"})
        joined = df.merge(previous, on=KEYS, how="left", validate="one_to_one")
        for zone, part in joined.groupby("market_id"):
            available = part.dropna(subset=["lag_deviation"])
            rows.append({
                "market_id": zone, "calendar_lag_days": lag,
                "availability_pct": 100 * part.lag_deviation.notna().mean(),
                "deviation_correlation": available.deviation.corr(available.lag_deviation),
                "copy_shape_mae_when_available": (available.deviation - available.lag_deviation).abs().mean(),
                "flat_mae_on_same_available_rows": available.absolute_deviation.mean(),
            })
    return pd.DataFrame(rows)


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(df, hourly, summary, output: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, zone, color in zip(axes.flat, ZONES, COLORS):
        values = df.loc[df.market_id.eq(zone), "price_eur_mwh"]
        ax.hist(values, bins=100, color=color)
        ax.set(title=zone, xlabel="Price (EUR/MWh)", ylabel="Observations", yscale="log")
    fig.suptitle("Price distributions: full tails retained; logarithmic count axis")
    fig.tight_layout()
    save_figure(fig, output / "01_price_distributions.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Descriptive QQ plots: dependence means these are not formal normality tests.
    probs = np.linspace(.001, .999, 501)
    normal = np.array([NormalDist().inv_cdf(float(p)) for p in probs])
    for ax, zone, color in zip(axes.flat, ZONES, COLORS):
        values = df.loc[df.market_id.eq(zone), "deviation"]
        quantiles = values.quantile(probs).to_numpy()
        ax.plot(normal, quantiles, color=color, linewidth=1.5)
        ax.plot(normal, values.mean() + values.std() * normal, "--", color="#777777", label="Matched Gaussian")
        ax.set(title=zone, xlabel="Standard Gaussian quantile", ylabel="Deviation (EUR/MWh)")
        ax.legend(fontsize=8)
    fig.suptitle("Quarter-hour deviation QQ plots (descriptive, not conditional residual tests)")
    fig.tight_layout()
    save_figure(fig, output / "02_deviation_qq.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    profiles = df.groupby(["market_id", "hour_index", "quarter_index"]).deviation.mean()
    limit = float(profiles.abs().max())
    for ax, zone in zip(axes.flat, ZONES):
        matrix = profiles.loc[zone].unstack("quarter_index")
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(title=zone, xlabel="Quarter-hour position", ylabel="Hour index (1–24)", xticks=range(4), xticklabels=range(1, 5), yticks=[0, 5, 11, 17, 23], yticklabels=[1, 6, 12, 18, 24])
    fig.suptitle("Average within-hour shape by zone and hour")
    fig.colorbar(image, ax=list(axes.flat), label="Mean deviation (EUR/MWh)", shrink=.8)
    save_figure(fig, output / "03_shape_heatmaps.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    # Select a representative day by median daily flat error; record selection in titles.
    for ax, zone, color in zip(axes.flat, ZONES, COLORS):
        part = df[df.market_id.eq(zone)]
        daily = part.groupby("delivery_date").absolute_deviation.mean()
        date = (daily - daily.median()).abs().idxmin()
        day = part[part.delivery_date.eq(date)]
        x = day.hour_index - 1 + (day.quarter_index - 1) / 4
        ax.plot(x, day.price_eur_mwh, color=color, label="Observed 15-min")
        ax.step(x, day.hourly_price, where="post", color="#777777", alpha=.8, label="Hourly repetition")
        ax.set(title=f"{zone}: {date:%Y-%m-%d}", xlabel="Hours since midnight", ylabel="EUR/MWh", xticks=[0, 6, 12, 18, 24])
        ax.legend(fontsize=8)
    fig.suptitle("Representative days: closest to each zone's median daily flat-baseline MAE")
    fig.tight_layout()
    save_figure(fig, output / "04_representative_days.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for zone, color in zip(ZONES, COLORS):
        part = df[df.market_id.eq(zone)]
        axes[0].plot(part.groupby("hour_index").absolute_deviation.mean(), label=zone, color=color)
        monthly = part.groupby("month").absolute_deviation.mean()
        axes[1].plot(monthly.index, monthly.values, label=zone, color=color)
    axes[0].set(xlabel="Hour index", ylabel="Flat-baseline MAE (EUR/MWh)", title="Reconstruction difficulty by hour")
    axes[1].set(xlabel="Delivery month", ylabel="Flat-baseline MAE (EUR/MWh)", title="Reconstruction difficulty over time")
    axes[1].tick_params(axis="x", rotation=60)
    axes[0].legend()
    fig.tight_layout()
    save_figure(fig, output / "05_flat_baseline_patterns.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, zone, color in zip(axes.flat, ZONES, COLORS):
        part = hourly[hourly.market_id.eq(zone)].dropna(subset=["central_slope"])
        bins = pd.qcut(part.central_slope, 15, duplicates="drop")
        grouped = part.groupby(bins, observed=True).agg(slope=("central_slope", "mean"), spread=("q4_minus_q1", "mean"))
        ax.scatter(part.central_slope, part.q4_minus_q1, s=3, alpha=.08, color=color, rasterized=True)
        ax.plot(grouped.slope, grouped.spread, color="#202020", marker="o", markersize=3, label="Binned averages")
        ax.axhline(0, color="#aaaaaa", linewidth=.7)
        ax.set(title=zone, xlabel="Central hourly slope (EUR/MWh per hour)", ylabel="Q4 − Q1 price (EUR/MWh)")
        ax.legend(fontsize=8)
    fig.suptitle("Does the supplied hourly curve help identify the within-hour direction?")
    fig.tight_layout()
    save_figure(fig, output / "06_slope_and_shape.png")


def markdown_table(df: pd.DataFrame) -> str:
    """Small Markdown table without an optional tabulate dependency."""
    shown = df.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    return "\n".join([
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)],
    ])


def run_analysis(input_path: Path = ROOT / "Data/train_prices_15min.csv",
                 output_dir: Path = ROOT / "analysis/results",
                 eda_end: str = "2026-08-31") -> dict:
    cutoff = pd.Timestamp(eda_end)
    if cutoff >= pd.Timestamp("2026-09-01"):
        raise ValueError("September is reserved: EDA cutoff must precede 2026-09-01")
    full, checks = load_and_check(Path(input_path))
    selected = full[full.delivery_date.le(cutoff)].copy()
    if selected.empty:
        raise ValueError("EDA date selection is empty")
    df, hourly = prepare(selected)
    output_dir = Path(output_dir)
    tables, figures = output_dir / "tables", output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    checks.update({"eda_end_inclusive": str(cutoff.date()), "eda_rows": len(df), "eda_days_per_zone": int(df.groupby("market_id").delivery_date.nunique().min()), "september_price_analysis": False})
    (output_dir / "quality_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    summary = df.groupby("market_id").agg(
        rows=("price_eur_mwh", "size"), days=("delivery_date", "nunique"),
        mean_price=("price_eur_mwh", "mean"), median_price=("price_eur_mwh", "median"),
        min_price=("price_eur_mwh", "min"), max_price=("price_eur_mwh", "max"),
        negative_pct=("price_eur_mwh", lambda x: 100*x.lt(0).mean()),
        flat_mae=("absolute_deviation", "mean"),
        deviation_skew=("deviation", "skew"),
        deviation_excess_kurtosis=("deviation", lambda x: x.kurt()),
        abs_deviation_p95=("absolute_deviation", lambda x: x.quantile(.95)),
        abs_deviation_p99=("absolute_deviation", lambda x: x.quantile(.99)),
    ).reset_index()
    summary.to_csv(tables / "zone_summary.csv", index=False)
    for grouping, name in [(["market_id", "hour_index", "quarter_index"], "mean_shapes"),
                           (["market_id", "month"], "monthly_patterns"),
                           (["market_id", "hour_index"], "hourly_patterns"),
                           (["market_id", "weekday"], "weekday_patterns")]:
        df.groupby(grouping).agg(mean_deviation=("deviation", "mean"), flat_mae=("absolute_deviation", "mean"), observations=("deviation", "size")).reset_index().to_csv(tables / f"{name}.csv", index=False)
    lags = lag_diagnostics(df)
    lags.to_csv(tables / "calendar_lag_diagnostics.csv", index=False)
    slopes = hourly.groupby("market_id").apply(
        lambda p: p.central_slope.corr(p.q4_minus_q1), include_groups=False
    ).rename("slope_vs_q4_minus_q1_correlation").reset_index()
    slopes.to_csv(tables / "slope_correlations.csv", index=False)
    # First-quarter/last-quarter errors equal EUR valuation errors for a 1 MWh position.
    valuation = df[df.quarter_index.isin([1, 4])].groupby(["market_id", "quarter_index"]).absolute_deviation.mean().unstack()
    valuation.columns = ["early_1MWh_mean_absolute_error_EUR", "late_1MWh_mean_absolute_error_EUR"]
    valuation.reset_index().to_csv(tables / "illustrative_flat_valuation.csv", index=False)
    make_figures(df, hourly, summary, figures)
    score = float(summary.flat_mae.mean())
    slope_min = slopes.slope_vs_q4_minus_q1_correlation.min()
    slope_max = slopes.slope_vs_q4_minus_q1_correlation.max()
    summary_view = summary[["market_id", "days", "min_price", "max_price", "negative_pct", "flat_mae"]]
    report = f"""# Initial AET price analysis

Generated by `analysis/initial_analysis.py` from the official local CSV only.

## Scope and data quality

The complete file contains {checks['rows']:,} observations, four zones and 363 dates per zone.
The exact expected observation grid passes: no missing keys, duplicate keys, missing values or non-finite prices.
The excluded dates are 2025-10-26 and 2026-03-29. Hours use the supplied CET/CEST delivery-date convention.

**All price statistics and plots below cover 2025-10-01 through {cutoff:%Y-%m-%d} only**
({checks['eda_days_per_zone']} days per zone). September prices are excluded from exploratory analysis.
Full-file structural checks do not constitute a model evaluation.

## Main findings

- The hourly central slope and Q4-minus-Q1 price difference have correlations from
  **{slope_min:.3f} to {slope_max:.3f}** across zones. This supports testing slope features.
- Copying yesterday's deviations improves on hourly repetition in all four zones on matched
  available historical rows. Weekly lags are not uniformly better than the flat baseline.
- Mean shapes differ by hour and zone; the heatmaps expose rising and falling within-hour patterns.
- Pooled deviations have heavier tails than the matched Gaussian in the QQ plots.
  This does not determine the likelihood of a future conditional model.

These are exploratory findings over the development period, not claims of held-out model accuracy.

## Price levels and flat baseline

{markdown_table(summary_view)}

Prices and MAE are in EUR/MWh; negative_pct is a percentage. The equal-zone flat-baseline MAE
over this exploratory period is **{score:.4f} EUR/MWh**. This is a descriptive baseline, not a
held-out result for a fitted model. No predictive model has been trained by this script.

![Price distributions](figures/01_price_distributions.png)

## Within-hour shapes and tails

The reconstruction target is `deviation = quarter_hour_price - hourly_average`.
Its four values sum to zero within every hour. Inspect patterns by hour rather than relying
on a single daily-average shape: rising and falling hours can cancel.

![Mean shapes](figures/03_shape_heatmaps.png)
![Representative days](figures/04_representative_days.png)

Representative dates are selected automatically as the date closest to each zone's median daily
flat-baseline MAE. They are not selected for visual impact or model performance.

![Deviation QQ plots](figures/02_deviation_qq.png)

The QQ plots and skew/kurtosis in `tables/zone_summary.csv` describe pooled deviations.
They do not establish the distribution of errors conditional on future model inputs.
No independent-observation normality-test p-values are reported because these time series are dependent.
Keep negative prices and extremes; do not infer market price caps from observed extrema.

## Hourly curve and calendar lags

![Slope and shape](figures/06_slope_and_shape.png)

{markdown_table(slopes)}

The central slope uses previous and next hours of the supplied same-day hourly curve.
Boundary hours are excluded from this diagnostic, not from the dataset. Correlation is descriptive,
not proof of predictive improvement or a causal explanation.

{markdown_table(lags)}

Lag availability includes start-of-file gaps and excluded DST dates. Joins use exact calendar dates,
zone, hour and quarter. Copy-shape and flat MAEs are compared on the same available rows;
these are historical diagnostics, not selected-model test results.

![Difficulty by hour and month](figures/05_flat_baseline_patterns.png)

## Illustrative business consequence

{markdown_table(valuation.reset_index())}

Each example delivers 1 MWh in the first or fourth quarter of an hour and zero elsewhere.
Values are mean absolute per-hour valuation errors in EUR under hourly repetition.
They are an interpretation of quarter-specific price errors, not trading profit or actual AET exposure.
Equal energy in every quarter would have zero valuation error by hourly-average consistency.

## Next implementation steps

1. Train hourly shape templates on October–June, then evaluate on July–August.
2. Compare a compact CatBoost model using zone/calendar, hourly slopes and exact historical lags.
3. Centre each four-quarter prediction; score only the final constrained output.
4. Keep September for one final chronological assessment; do not tune on it.
5. Freeze missing-lag fallbacks and all inference rules before submission.

No external data, zone identification or evaluation-period lookups are used. Do not round hourly
averages or predictions to two decimals. `centre_predictions` is a reusable constraint helper;
an eventual submission must additionally validate keys, finiteness and averages after CSV round-trip.
"""
    (output_dir / "initial_analysis.md").write_text(report)
    print(json.dumps(checks, indent=2))
    print(summary_view.to_string(index=False))
    print(f"Exploratory equal-zone flat MAE: {score:.6f} EUR/MWh")
    print(f"Report: {output_dir / 'initial_analysis.md'}")
    return {"checks": checks, "summary": summary, "lags": lags, "slopes": slopes, "output_dir": output_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Data/train_prices_15min.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/results")
    parser.add_argument("--eda-end", default="2026-08-31", help="Inclusive cutoff; must precede September")
    args = parser.parse_args()
    run_analysis(args.input, args.output_dir, args.eda_end)
