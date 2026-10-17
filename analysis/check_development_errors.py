"""Replay the fixed final recipe on Jan–Aug development folds; no selection."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/applepi-error-analysis-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AET-HACKATHON"))
from aet.baselines import central_slope
from aet.config import load_config, final_config, folds_from_config
from aet.cube import QuarterCube
from aet.pipeline import ShapePipeline, runtime_versions
from aet.schema import read_quarter_hourly, validate_predictions, HOUR_KEYS, QH_KEYS


def table(frame):
    f = frame.copy()
    for c in f.select_dtypes(include="number"):
        f[c] = f[c].map(lambda x: f"{x:.3f}")
    return "\n".join(["| " + " | ".join(f.columns) + " |",
                      "| " + " | ".join(["---"] * len(f.columns)) + " |",
                      *["| " + " | ".join(map(str, r)) + " |" for r in f.itertuples(index=False, name=None)]])


def main():
    started = time.perf_counter()
    out = ROOT / "analysis/results" / datetime.now().strftime("development_error_check_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    artifacts = [p for name in ["artifacts_official", "artifacts_september_assessment"]
                 for p in (ROOT / "AET-HACKATHON" / name).iterdir() if p.is_file()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    raw = load_config(ROOT / "AET-HACKATHON/configs/final.yaml")
    config = final_config(raw)
    folds, _ = folds_from_config(raw, "monthly")
    assert len(folds) == 8 and all(f.score_end < np.datetime64("2026-09-01") for f in folds)
    # Cut off September BEFORE constructing any analysis statistics or model inputs.
    data = read_quarter_hourly(ROOT / "Data/train_prices_15min.csv")
    data = data[data.delivery_date < pd.Timestamp("2026-09-01")].copy()
    cube = QuarterCube.from_long(data, config.zones)
    H_all = cube.hourly()
    records, limits, fold_checks = [], [], []
    for fold in folds:
        tick = time.perf_counter()
        training = cube.until(fold.train_end)
        pipe = ShapePipeline(config).fit(training)
        assert training.dates.max() < fold.score_start
        # Availability-safe regime thresholds estimated on this fold's training data.
        thresholds = np.quantile(np.abs(central_slope(training.hourly())), .75, axis=(0, 2))
        for z, zone in enumerate(config.zones):
            limits.append({"fold": fold.name, "zone": zone, "train_end": str(fold.train_end),
                           "steep_slope_p75": thresholds[z]})
        max_mean_error = 0.
        for i in np.flatnonzero((cube.dates >= fold.score_start) & (cube.dates <= fold.score_end)):
            day, H, actual = cube.dates[i], H_all[i], cube.values[i]
            prediction, info = pipe.predict_day(day, H, cube)
            assert info["fallback_hours"].sum() == info["flat_hours"].sum() == 0
            pchip = pipe.baseline(H)
            slope = central_slope(H)
            slope2 = H[..., None] + 2 * slope[..., None] * np.array([-.375, -.125, .125, .375])
            error = np.abs(prediction - actual)
            consistency = float(np.max(np.abs(prediction.mean(axis=-1) - H)))
            max_mean_error = max(max_mean_error, consistency)
            assert consistency <= .0001 and np.isfinite(prediction).all()
            frame = QuarterCube(np.array([day]), config.zones, actual[None]).to_long()
            frame["fold"] = fold.name
            frame["train_end"] = str(fold.train_end)
            frame["predicted_price_eur_mwh"] = prediction.reshape(-1)
            frame["flat_price"] = np.repeat(H[..., None], 4, axis=-1).reshape(-1)
            frame["pchip_price"] = pchip.reshape(-1)
            frame["slope2_price"] = slope2.reshape(-1)
            frame["model_error"] = error.reshape(-1)
            frame["signed_error"] = (prediction-actual).reshape(-1)
            for method in ["flat", "pchip", "slope2"]:
                frame[method + "_error"] = (frame[method + "_price"] - frame.price_eur_mwh).abs()
            backward = np.concatenate([np.full((4, 1), np.nan), np.diff(H, axis=-1)], axis=-1)
            forward = np.concatenate([np.diff(H, axis=-1), np.full((4, 1), np.nan)], axis=-1)
            turning = backward * forward < 0
            geometry = np.where(turning, "turning point", "ramp/plateau transition")
            geometry[:, [0, 23]] = "day edge"
            for name, value in {"hourly_slope": slope, "steep_ramp": np.abs(slope) > thresholds[:, None],
                                "geometry": geometry, "actual_range": np.ptp(actual, axis=-1),
                                "predicted_range": np.ptp(prediction, axis=-1),
                                "actual_tilt": actual[..., 3] - actual[..., 0],
                                "predicted_tilt": prediction[..., 3] - prediction[..., 0]}.items():
                frame[name] = np.repeat(value[..., None], 4, axis=-1).reshape(-1)
            records.append(frame)
        fold_checks.append({"fold": fold.name, "train_end": str(fold.train_end),
                            "score_start": str(fold.score_start), "score_end": str(fold.score_end),
                            "max_hourly_mean_error": max_mean_error,
                            "seconds": time.perf_counter()-tick})
        print(f"Completed {fold.name}: {fold_checks[-1]['seconds']:.1f}s", flush=True)
    d = pd.concat(records, ignore_index=True)
    assert not d.duplicated(QH_KEYS).any() and d.delivery_date.max() < pd.Timestamp("2026-09-01")
    d["hour_start"] = d.hour_index-1
    d["time"] = ((d.hour_index-1)*60 + (d.quarter_index-1)*15).map(lambda n: f"{n//60:02d}:{n%60:02d}")
    d["afternoon_16_20"] = d.hour_start.between(16, 19)
    d["outer_quarter"] = d.quarter_index.isin([1, 4])
    d["range_bias"] = d.predicted_range - d.actual_range
    d["oriented_tilt_error"] = (d.predicted_tilt - d.actual_tilt)*np.sign(d.hourly_slope)
    d.to_csv(out / "all_development_predictions.csv", index=False)
    reloaded = pd.read_csv(out / "all_development_predictions.csv", parse_dates=["delivery_date"])
    validation = validate_predictions(reloaded[QH_KEYS + ["predicted_price_eur_mwh"]],
        reloaded[HOUR_KEYS + ["flat_price"]].drop_duplicates().rename(
            columns={"flat_price": "hourly_price_eur_mwh"}))
    total = d.model_error.sum()

    def summarize(keys):
        s = d.groupby(keys).agg(predictions=("model_error", "size"), dates=("delivery_date", "nunique"),
            model_MAE=("model_error", "mean"), flat_MAE=("flat_error", "mean"),
            pchip_MAE=("pchip_error", "mean"), slope2_MAE=("slope2_error", "mean"),
            signed_bias=("signed_error", "mean"), range_bias=("range_bias", "mean"),
            oriented_tilt_error=("oriented_tilt_error", "mean"), error_sum=("model_error", "sum"))
        s["error_share_pct"] = 100*s.pop("error_sum")/total
        s["observation_share_pct"] = 100*s.predictions/len(d)
        return s.reset_index()

    tables = {}
    for name, keys in {"zone": ["market_id"], "quarter": ["quarter_index"], "hour": ["hour_start"],
                       "fold": ["fold"], "geometry": ["geometry"], "steep_ramp": ["steep_ramp"],
                       "fold_hour": ["fold", "hour_start"], "zone_quarter": ["market_id", "quarter_index"],
                       "zone_steep_ramp": ["market_id", "steep_ramp"],
                       "fold_afternoon": ["fold", "afternoon_16_20"],
                       "fold_outer_quarter": ["fold", "outer_quarter"],
                       "fold_steep_ramp": ["fold", "steep_ramp"]}.items():
        tables[name] = summarize(keys)
        tables[name].to_csv(out / f"errors_by_{name}.csv", index=False)
    methods = ["model_error", "flat_error", "pchip_error", "slope2_error"]
    daily_zone = d.groupby(["delivery_date", "market_id"])[methods].mean()
    official = daily_zone.groupby("market_id").mean().mean()
    assert np.isclose(official.model_error, d.model_error.mean())
    daily_zone.to_csv(out / "daily_zone_mae.csv")
    hour_shapes = d.drop_duplicates(HOUR_KEYS)
    hour_shapes[HOUR_KEYS + ["fold", "hourly_slope", "steep_ramp", "geometry",
        "actual_range", "predicted_range", "range_bias", "actual_tilt", "predicted_tilt",
        "oriented_tilt_error"]].to_csv(out / "hourly_shape_diagnostics.csv", index=False)
    fold_results = []
    for fold, part in d.groupby("fold"):
        afternoon = part.afternoon_16_20
        outer = part.outer_quarter
        steep = part.steep_ramp
        fold_results.append({"fold": fold, "MAE": part.model_error.mean(),
            "afternoon_MAE": part.loc[afternoon, "model_error"].mean(),
            "other_hours_MAE": part.loc[~afternoon, "model_error"].mean(),
            "afternoon_error_share_pct": 100*part.loc[afternoon, "model_error"].sum()/part.model_error.sum(),
            "outer_MAE": part.loc[outer, "model_error"].mean(),
            "middle_MAE": part.loc[~outer, "model_error"].mean(),
            "outer_error_share_pct": 100*part.loc[outer, "model_error"].sum()/part.model_error.sum(),
            "steep_range_bias": part.loc[steep, "range_bias"].mean(),
            "other_range_bias": part.loc[~steep, "range_bias"].mean(),
            "steep_oriented_tilt_error": part.loc[steep, "oriented_tilt_error"].mean()})
    f = pd.DataFrame(fold_results)
    f.to_csv(out / "fold_pattern_checks.csv", index=False)
    pd.DataFrame(limits).to_csv(out / "training_regime_thresholds.csv", index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    findings = {"predictions": len(d), "dates": int(d.delivery_date.nunique()),
        "official_MAE": float(official.model_error), "flat_MAE": float(official.flat_error),
        "afternoon_error_share_pct": float(100*d.loc[d.afternoon_16_20,"model_error"].sum()/total),
        "outer_error_share_pct": float(100*d.loc[d.outer_quarter,"model_error"].sum()/total),
        "folds_afternoon_worse_than_other_hours": int((f.afternoon_MAE>f.other_hours_MAE).sum()),
        "folds_outer_worse_than_middle": int((f.outer_MAE>f.middle_MAE).sum()),
        "folds_steep_range_underestimated": int((f.steep_range_bias<0).sum()),
        "folds_steep_tilt_underestimated": int((f.steep_oriented_tilt_error<0).sum()),
        "seconds": time.perf_counter()-started, "versions": runtime_versions(),
        "fold_validation": fold_checks, "artifact_hashes_unchanged": hashes,
        "config": config.to_dict(), "serialized_output_validation": validation}
    (out / "summary.json").write_text(json.dumps(findings, indent=2, default=str)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for field, label in [("model_MAE", "Final recipe"), ("flat_MAE", "Flat"), ("slope2_MAE", "Slope x 2")]:
        axes[0].plot(tables["hour"].hour_start, tables["hour"][field], label=label)
    axes[0].set(xlabel="Clock hour start", ylabel="MAE (EUR/MWh)", title="Development errors by hour")
    axes[0].legend()
    x = np.arange(len(f))
    axes[1].bar(x-.18, f.outer_MAE, width=.36, label="Q1/Q4")
    axes[1].bar(x+.18, f.middle_MAE, width=.36, label="Q2/Q3")
    axes[1].set(xticks=x, xticklabels=[v[-2:] for v in f.fold], xlabel="Month in 2026", ylabel="MAE (EUR/MWh)", title="Outer-quarter error repeats across months")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "development_error_patterns.png", dpi=160)
    plt.close(fig)
    conclusion = ("Yes: steep-ramp amplitude underestimation repeats in every fold."
                  if findings["folds_steep_range_underestimated"]==8 and findings["folds_steep_tilt_underestimated"]==8
                  else "The amplitude diagnosis is mixed: inspect the fold and zone differences before choosing an amplitude correction.")
    cols = ["predictions", "model_MAE", "flat_MAE", "pchip_MAE", "slope2_MAE", "error_share_pct"]
    report = f"""# Do the September error patterns repeat in development?

**Afternoon errors are higher than other hours in {findings['folds_afternoon_worse_than_other_hours']}/8 folds;
Q1/Q4 errors are higher than Q2/Q3 in {findings['folds_outer_worse_than_middle']}/8 folds.**
{conclusion}

This run replays the unchanged final recipe on eight rolling-origin January–August 2026 folds,
scoring day 19 through each month-end: **{findings['dates']} dates, {len(d):,} predictions**.
Each temporary fold model is fitted only through the previous month-end. Weights are fixed
within the fold; historical quarter prices are sliced strictly before each predicted date.
No September or October observations enter the computations. Both original artifact directories
were checked by SHA-256 and remain unchanged. Fold models were used in memory, not saved over artifacts.

## Score and repeatability

Official-style final-recipe MAE: **{official.model_error:.4f} EUR/MWh**, versus flat
**{official.flat_error:.4f}**, PCHIP **{official.pchip_error:.4f}**, and fixed slope × 2
**{official.slope2_error:.4f}**. Runtime versions are recorded in `summary.json`; they differ from
the original training environment, so small discrepancies from published scores are possible.

16:00–20:00 contributes **{findings['afternoon_error_share_pct']:.1f}%** of total error from 16.7% of observations.
Q1/Q4 contribute **{findings['outer_error_share_pct']:.1f}%** from 50% of observations.
The table below checks each month rather than relying only on the pooled result.

{table(f[['fold','MAE','afternoon_MAE','other_hours_MAE','afternoon_error_share_pct','outer_MAE','middle_MAE','outer_error_share_pct']])}

![Development patterns](development_error_patterns.png)

## Does the model underestimate ramp amplitude?

“Steep” means absolute central hourly slope exceeds that zone's training-fold 75th percentile;
the day edges use a one-sided slope. The cutoff is fitted on training inputs only, not chosen
from evaluation outcomes. Its exact values are in `training_regime_thresholds.csv`.

- Range bias = predicted quarter max-minus-min minus observed quarter max-minus-min.
  Negative means too narrow. This diagnostic is meaningful but expected uncertainty can make
  a conditional-median prediction narrower than realized extremes, even for a well-chosen MAE model.
- Oriented tilt error = `(predicted Q4-Q1 - observed Q4-Q1) * sign(hourly slope)`.
  Negative means too little net tilt in the hourly ramp direction on average. Unlike an absolute
  range, this tests a signed direction. It still is not proof that a global amplitude increase lowers MAE.

{table(f[['fold','steep_range_bias','other_range_bias','steep_oriented_tilt_error']])}

Range is underestimated on steep ramps in **{findings['folds_steep_range_underestimated']}/8 folds**;
directional tilt is underestimated in **{findings['folds_steep_tilt_underestimated']}/8**.
Each statistic is averaged over hours (repeating it for four quarters leaves the mean unchanged).

{table(tables['steep_ramp'][['steep_ramp',*cols,'range_bias','oriented_tilt_error']])}

## Zone differences

{table(tables['zone'][['market_id',*cols]])}

{table(tables['zone_steep_ramp'][['market_id','steep_ramp','model_MAE','range_bias','oriented_tilt_error']])}

**The pooled pattern is not a reason to amplify every zone.** In this run, ZONE_1's steep-ramp
oriented tilt error is about **−6.19 EUR/MWh**, whereas ZONE_2's is about **+0.88 EUR/MWh**.
ZONE_2 can have a slightly underestimated total range while its net Q4-minus-Q1 tilt is too
large: those are different shape measures. ZONE_1, followed by ZONE_3, is the stronger starting
point for an amplitude diagnosis; ZONE_2 merits a shape-specific investigation instead.

Use this table to distinguish a shared amplitude issue from a zone-specific one. Pooled data can
hide opposite signs. A high absolute error is also partly a consequence of a more variable target.

## Turning points versus ramps

{table(tables['geometry'][['geometry',*cols]])}

Turning points use opposite-signed backward/forward hourly differences; day edges are separate.
The remainder includes ramps and plateau transitions. These coarse groups mix hours and zones,
so conditional comparisons do not establish causality. Check `errors_by_zone_quarter.csv` and
`errors_by_fold_hour.csv` for more detail.

## Recommended next experiment, without claiming an improvement yet

1. Prioritize the weaknesses that repeat across months, using these summaries rather than a single
   September extreme. Inspect zone-specific signs before assuming every ramp needs more amplitude.
2. Test one small feature family: explicit local absolute movement and ramp asymmetry/reversal flags.
   Existing slopes and curvature already encode much of this; require incremental price-MAE gain.
   Focus the diagnostic review on ZONE_1/ZONE_3 steep ramps; do not apply an automatic upward
   amplitude adjustment to ZONE_2.
3. If testing coefficient scales, estimate a small, regularized set on earlier development predictions
   and confirm on later months. Do not tune a scale on the same errors used to report its performance.
   The incumbent's 0.8 applies only to its learned correction, not to its full intra-hour shape.
4. Preserve the existing promotion gate: >=0.5% lower MAE, improvement on >=6/8 folds, no zone >2% worse.
   No candidate feature set or amplitude change was trained or selected in this diagnostic run.

These development folds have already supported model selection, and this investigation follows
September findings. Repetition is evidence of a recurring failure pattern, not independent final-test
confirmation. The attainable improvement remains unknown until a separately evaluated candidate wins.

## Files and reproduction

- `all_development_predictions.csv`: every prediction and observed value with fold cutoff and baseline errors.
- `fold_pattern_checks.csv`: the monthly comparisons above; `hourly_shape_diagnostics.csv`: per-hour amplitudes.
- `errors_by_*.csv`: zone, quarter, hour, geometry, steepness, and fold-specific summaries.
- `summary.json`: scores, runtime versions, configuration, cutoff checks, and unchanged-artifact hashes.
- Run from ApplePi: `.venv-mae/bin/python analysis/check_development_errors.py`.
"""
    (out / "development_error_check.md").write_text(report)
    print("\nFOLD CHECKS\n" + f.to_string(index=False), flush=True)
    print("\nSTEEPNESS\n" + tables["steep_ramp"].to_string(index=False), flush=True)
    print("\nZONES\n" + tables["zone_steep_ramp"].to_string(index=False), flush=True)
    print("\nOFFICIAL MAE", official.to_dict(), flush=True)
    print("\nREPORT", out / "development_error_check.md", flush=True)


if __name__ == "__main__":
    main()
