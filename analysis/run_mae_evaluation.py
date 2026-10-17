"""Run the saved August-cutoff model on September 19–30, without fitting."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AET-HACKATHON"))

import numpy as np
import pandas as pd
import scipy
import lightgbm

from aet.baselines import cumulative, central_slope
from aet.cube import HourlyCube, QuarterCube
from aet.infer import run
from aet.schema import HOUR_KEYS, QH_KEYS, read_quarter_hourly, validate_predictions
from tools.score import score


def table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    return "\n".join([
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join(["---"] * len(shown.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |"
          for row in shown.itertuples(index=False, name=None)],
    ])


def main():
    model = ROOT / "AET-HACKATHON/artifacts_september_assessment"
    manifest = json.loads((model / "manifest.json").read_text())
    assert manifest["training_range"][1] == "2026-08-31"
    artifact_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in model.iterdir() if p.is_file()}
    output = ROOT / "analysis/results" / datetime.now().strftime("mae_evaluation_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    raw = read_quarter_hourly(ROOT / "Data/train_prices_15min.csv")
    truth = raw[raw.delivery_date.between("2026-09-19", "2026-09-30")].copy()
    assert truth.delivery_date.nunique() == 12 and len(truth) == 12 * 384
    history = raw[raw.delivery_date < truth.delivery_date.max()].copy()
    hourly = truth.groupby(HOUR_KEYS, as_index=False).price_eur_mwh.mean().rename(
        columns={"price_eur_mwh": "hourly_price_eur_mwh"})
    truth.to_csv(output / "truth_15min.csv", index=False)
    history.to_csv(output / "history_15min.csv", index=False)
    hourly.to_csv(output / "test_hourly.csv", index=False, float_format="%.17g")
    diagnostics = run(output / "test_hourly.csv", output / "history_15min.csv",
                      model, output / "predictions.csv")
    assert not diagnostics.get("emergency_flat", False), diagnostics
    assert diagnostics["fallback_hours"] == 0 and diagnostics["flat_hours"] == 0, diagnostics
    pred = pd.read_csv(output / "predictions.csv", parse_dates=["delivery_date"])
    checks = validate_predictions(pred, hourly)
    zones, quarters, overall = score(pred, truth)
    zones.to_csv(output / "mae_by_zone.csv")
    quarters.to_csv(output / "mae_by_quarter.csv")

    # Construct all comparison methods on the same supplied hourly inputs.
    supplied = pd.read_csv(output / "test_hourly.csv", parse_dates=["delivery_date"])
    cube = HourlyCube.from_long(supplied, tuple(manifest["config"]["zones"]))
    H = cube.values
    offsets = np.array([-.375, -.125, .125, .375])
    values = {
        "Flat": np.repeat(H[..., None], 4, axis=-1),
        "PCHIP": cumulative(H, "pchip"),
        "Slope x 2 (fixed)": H[..., None] + 2 * central_slope(H)[..., None] * offsets,
    }
    comparisons = []
    for name, array in values.items():
        frame = QuarterCube(cube.dates, cube.zones, array).to_long("predicted_price_eur_mwh")
        z, _, result = score(frame, truth)
        comparisons.append({"method": name, "MAE_EUR_MWh": result["score"],
                            "improvement_vs_flat_pct": result["gain_pct"],
                            **z.model.to_dict()})
    comparisons.append({"method": "Saved final model", "MAE_EUR_MWh": overall["score"],
                        "improvement_vs_flat_pct": overall["gain_pct"], **zones.model.to_dict()})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "method_comparison.csv", index=False)
    details = truth.merge(pred, on=QH_KEYS, validate="one_to_one").merge(
        hourly, on=HOUR_KEYS, validate="many_to_one")
    details["absolute_error"] = (details.predicted_price_eur_mwh - details.price_eur_mwh).abs()
    details["flat_absolute_error"] = (details.hourly_price_eur_mwh - details.price_eur_mwh).abs()
    details.to_csv(output / "prediction_details.csv", index=False)
    daily_zone = details.groupby(["delivery_date", "market_id"])[
        ["absolute_error", "flat_absolute_error"]].mean()
    daily_zone.to_csv(output / "daily_zone_mae.csv")
    daily = daily_zone.groupby("delivery_date").mean()
    daily.to_csv(output / "daily_mae.csv")
    assert np.isclose(overall["score"], daily_zone.groupby("market_id").absolute_error.mean().mean())
    assert all(hashlib.sha256((model / name).read_bytes()).hexdigest() == value
               for name, value in artifact_hashes.items())
    metadata = {
        "model": str(model), "model_training_range": manifest["training_range"],
        "evaluation_start": "2026-09-19", "evaluation_end": "2026-09-30",
        "evaluation_dates": 12, "predictions": len(pred), "overall": overall,
        "checks": checks, "inference_diagnostics": diagnostics,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "lightgbm": lightgbm.__version__},
        "artifact_hashes_unchanged": artifact_hashes,
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    examples = details.sort_values(QH_KEYS).head(8)[
        QH_KEYS + ["price_eur_mwh", "predicted_price_eur_mwh", "absolute_error"]]
    report = f"""# Model MAE: actual saved-model run

The saved assessment model was trained through **31 August 2026**. This run reconstructs
**19–30 September 2026**, covering 12 dates, four zones, and {len(pred):,} quarter-price predictions.
No models were fitted, tuned, or overwritten. Both saved-model directories remain separate;
the full-year submission model was not used to score September.

## What MAE means

MAE = mean(abs(predicted price - observed price)). It is measured in **EUR/MWh**, not percent.
For example, predicting 105 when the observed price is 100 gives an absolute error of 5 EUR/MWh.
Overprediction and underprediction both count positively. Lower is better; zero is perfect.
The official score averages quarter errors per date and zone, then dates within each zone,
then the four zones equally. A mean does not guarantee that every prediction is within that error.

This model's MAE is **{overall['score']:.4f} EUR/MWh** versus **{overall['flat']:.4f}** for
repeating the hourly average four times: **{overall['gain_pct']:.2f}% less absolute error**.
That reduction is not a percentage accuracy or a return on an investment.

## Same-date comparison

{table(comparison)}

Flat uses the supplied hourly average for every quarter. PCHIP interpolates a cumulative
hourly curve while preserving each hourly mean. Slope x 2 is the EDA rule with a fixed multiplier,
not a multiplier selected on these September targets. The final model adds 0.8 times learned
zero-sum LightGBM corrections to PCHIP. The corrections describe within-hour slope, curvature,
and zigzag; inputs include hourly geometry, calendar, other zones' hourly curves, and past shapes.

## MAE by zone

{table(zones.reset_index())}

`model` and `flat` are EUR/MWh; `gain_pct` is the relative error reduction.

## MAE by quarter position

{table(quarters.reset_index())}

Q1 = :00–:15, Q2 = :15–:30, Q3 = :30–:45, Q4 = :45–:60 within each hour.

## MAE by date (four zones equally weighted)

{table(daily.reset_index())}

## Example predictions: first eight quarters of ZONE_0 on 19 September

{table(examples)}

## Checks and interpretation

- The weights are fixed at the August cutoff. For each target date, the pipeline slices
  quarter-price history to strictly earlier dates; earlier evaluation dates can become history.
  The full target-day hourly curve is a permitted input. Observed target-day quarters are used
  to construct the supplied hourly reference and for scoring, never as shape features.
- Output validation passed; inference used no emergency or per-hour fallback.
- Original assessment artifact hashes were unchanged. Detailed validation and runtime versions
  are in `summary.json`. The runtime differs from the original Python 3.12 environment, so use
  the results from this run rather than assuming bit-identical reproduction of published scores.
- September was already explored during EDA. This reproduces a historical assessment, not a new
  independent test, a forecast of unknown hourly prices, or proof of performance on every day.
- To rerun from the ApplePi folder: `.venv-mae/bin/python analysis/run_mae_evaluation.py`.
  Each run creates its own results directory.
"""
    (output / "MAE_results.md").write_text(report)
    print(comparison.to_string(index=False))
    print("\nBY ZONE\n", zones.to_string())
    print("\nBY QUARTER\n", quarters.to_string())
    print("\nOUTPUT:", output)
    print("VALIDATION:", checks)
    print("VERSIONS:", metadata["versions"])


if __name__ == "__main__":
    main()
