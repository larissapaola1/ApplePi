"""Offline training, inference, and a fixed eight-fold TGNN comparison."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from . import REPO
from .temporal_graph import Config, GraphReconstructor, ZONES
from aet.baselines import cumulative, central_slope
from aet.config import load_config, final_config, folds_from_config
from aet.cube import QuarterCube
from aet.pipeline import ShapePipeline
from aet.schema import (HOUR_KEYS, QH_KEYS, read_quarter_hourly, read_hourly,
                        check_training_window, validate_predictions)


def markdown_table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    return "\n".join(["| " + " | ".join(shown.columns) + " |",
                      "| " + " | ".join(["---"]*len(shown.columns)) + " |",
                      *["| " + " | ".join(map(str,row)) + " |"
                        for row in shown.itertuples(index=False,name=None)]])


def training_cube(path, end):
    if not np.datetime64("2025-10-01") <= np.datetime64(end) <= np.datetime64("2026-09-30"):
        raise ValueError("invalid official training cutoff")
    frame = read_quarter_hourly(path)
    check_training_window(frame)
    frame = frame[frame.delivery_date <= pd.Timestamp(end)]
    cube = QuarterCube.from_long(frame, ZONES)
    if not len(cube.dates) or not np.isfinite(cube.values).all():
        raise ValueError("training requires complete four-zone days")
    return cube


def train(args):
    if args.out.exists():
        raise FileExistsError("choose a new output directory; artifacts are never overwritten")
    cube = training_cube(args.train, args.train_end)
    model = GraphReconstructor(Config(epochs=args.epochs, graph=not args.no_graph)).fit(cube)
    model.save(args.out)
    print(json.dumps({"saved": str(args.out), "training_range": model.training_range,
                      "parameters": sum(p.numel() for p in model.network.parameters()),
                      "final_training_MAE": model.losses[-1]}, indent=2))


def infer(args):
    hourly = read_hourly(args.hourly)
    if len(hourly) == 0:
        raise ValueError("empty hourly input")
    model = GraphReconstructor.load(args.model)
    history = model.training_history
    if args.history:
        supplied = QuarterCube.from_long(read_quarter_hourly(args.history, strict=False), ZONES)
        history = history.merge(supplied)
    pred = model.predict_frame(hourly, history)
    validate_predictions(pred, hourly)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("choose a new output file")
    pred.to_csv(args.out, index=False, float_format="%.10f")
    checks = validate_predictions(pd.read_csv(args.out), hourly)
    print(json.dumps({"output": str(args.out), **checks}, indent=2))


def backtest(args):
    if args.out.exists():
        raise FileExistsError("choose a new output directory")
    args.out.mkdir(parents=True)
    # No September targets enter training, scaling, scoring, or hyperparameter choice.
    cube = training_cube(args.train, "2026-08-31")
    raw = load_config(REPO/"AET-HACKATHON/configs/final.yaml")
    folds, _ = folds_from_config(raw, "monthly")
    incumbent = final_config(raw)
    cfg = Config(epochs=args.epochs, graph=not args.no_graph)
    records, predictions, timings, validations = [], [], [], []
    artifacts = [p for name in ["artifacts_official", "artifacts_september_assessment"]
                 for p in (REPO/"AET-HACKATHON"/name).iterdir() if p.is_file()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    for fold in folds:
        tick = time.perf_counter()
        training = cube.until(fold.train_end)
        graph = GraphReconstructor(cfg).fit(training)
        reference = ShapePipeline(incumbent).fit(training)
        dates = cube.dates[(cube.dates>=fold.score_start)&(cube.dates<=fold.score_end)]
        for day in dates:
            day_cube = cube._select(cube.dates == day)
            truth, H = day_cube.values[0], day_cube.hourly()[0]
            neural = graph.predict_day(day, H, cube)
            lgbm, diagnostics = reference.predict_day(day, H, cube)
            assert diagnostics["fallback_hours"].sum() == 0
            methods = {"TGNN" if cfg.graph else "Temporal_no_graph": neural,
                       "LightGBM_incumbent": lgbm, "Flat": np.repeat(H[..., None], 4, axis=-1),
                       "PCHIP": cumulative(H), "Slope_x2": H[..., None] +
                       2*central_slope(H)[..., None]*np.array([-.375,-.125,.125,.375])}
            for name, value in methods.items():
                drift = float(np.max(np.abs(value.mean(axis=-1)-H)))
                if not np.isfinite(value).all() or drift > .0001:
                    raise ValueError(f"non-compliant {name} prediction")
                for z, zone in enumerate(ZONES):
                    error = np.abs(value[z]-truth[z])
                    records.append({"method": name, "fold": fold.name, "delivery_date": str(day),
                        "zone": zone, "MAE": error.mean(), "hourly_mean_error": drift,
                        **{f"Q{q+1}_MAE": error[:,q].mean() for q in range(4)}})
            frame = day_cube.to_long()
            frame["fold"] = fold.name
            frame["predicted_price_eur_mwh"] = neural.reshape(-1)
            frame["absolute_error"] = np.abs(neural-truth).reshape(-1)
            predictions.append(frame)
        timings.append({"fold": fold.name, "seconds": time.perf_counter()-tick})
        print(f"{fold.name}: {timings[-1]['seconds']:.1f}s", flush=True)
        if fold == folds[-1]:
            graph.save(args.out/"last_fold_model")
    scores = pd.DataFrame(records)
    scores.to_csv(args.out/"daily_zone_scores.csv", index=False)
    detail = pd.concat(predictions, ignore_index=True)
    detail.to_csv(args.out/"predictions_vs_actual.csv", index=False)
    for day, part in detail.groupby("delivery_date"):
        hourly = part.groupby(HOUR_KEYS, as_index=False).price_eur_mwh.mean().rename(
            columns={"price_eur_mwh":"hourly_price_eur_mwh"})
        validations.append(validate_predictions(part[QH_KEYS+["predicted_price_eur_mwh"]], hourly))
    zone = scores.groupby(["method","zone"]).MAE.mean().unstack()
    summary = zone.copy()
    summary.insert(0, "MAE", zone.mean(axis=1))
    foldscores = scores.groupby(["method","fold","zone"]).MAE.mean().groupby(["method","fold"]).mean().unstack(0)
    target = "TGNN" if cfg.graph else "Temporal_no_graph"
    improvement = 1-summary.loc[target,"MAE"]/summary.loc["LightGBM_incumbent","MAE"]
    wins = int((foldscores[target]<foldscores.LightGBM_incumbent).sum())
    worstzone = float((zone.loc[target]/zone.loc["LightGBM_incumbent"]-1).max())
    passed = bool(improvement>=.005 and wins>=6 and worstzone<=.02)
    summary.to_csv(args.out/"method_comparison.csv")
    foldscores.to_csv(args.out/"fold_scores.csv")
    scores.groupby("method")[[f"Q{q}_MAE" for q in range(1,5)]].mean().to_csv(args.out/"quarter_scores.csv")
    for p, value in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==value
    metadata = {"config": asdict(cfg), "torch": str(torch.__version__),
        "numpy": np.__version__, "pandas": pd.__version__, "timings": timings,
        "relative_improvement_vs_incumbent": float(improvement), "folds_won": wins,
        "worst_zone_degradation": worstzone, "promotion_gate_passed": passed,
        "artifact_hashes_unchanged": hashes, "validation_max_mean_error": max(v['max_abs_mean_error'] for v in validations)}
    (args.out/"run.json").write_text(json.dumps(metadata, indent=2)+"\n")
    state = "passes" if passed else "does not pass"
    report = f"""# Temporal graph model: fixed development experiment

{markdown_table(summary.reset_index())}

All MAEs are EUR/MWh, evaluated on the same Jan–Aug 2026 rolling-origin folds,
days 19 to month-end, with strictly earlier training and quarter-price history.
The graph candidate {state} the existing promotion gate: relative MAE change versus
LightGBM is {-100*improvement:+.2f}%, wins {wins}/8 folds, worst-zone degradation
{100*worstzone:+.2f}%. No incumbent artifact or configuration was changed.

The recipe is fixed in advance at {cfg.epochs} epochs, hidden width {cfg.hidden},
{cfg.blocks} temporal/graph blocks; no scoring-target early stopping or hyperparameter search.
This is a single experimental run, not a claim that graph networks are optimal.
Development folds have already been used for earlier model selection. September and October
were excluded. Results do not resolve historical repository eligibility/provenance issues.

`last_fold_model/` is trained through July, solely for Aug replay and inference smoke checks;
it is not a full-year submission model. All row predictions are in `predictions_vs_actual.csv`.
Every day's output was checked for finite values, complete keys, and hourly mean consistency.
See `run.json` for settings, runtime versions, timing, promotion checks, and artifact hashes.
"""
    (args.out/"results.md").write_text(report)
    print("\n"+report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["train", "backtest"]:
        q = sub.add_parser(name)
        q.add_argument("--train", type=Path, default=REPO/"Data/train_prices_15min.csv")
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--epochs", type=int, default=80)
        q.add_argument("--no-graph", action="store_true", help="temporal-only comparison, same features")
        if name=="train":
            q.add_argument("--train-end", default="2026-08-31")
        q.set_defaults(func=train if name=="train" else backtest)
    q = sub.add_parser("infer")
    q.add_argument("--hourly", type=Path, required=True)
    q.add_argument("--history", type=Path)
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.set_defaults(func=infer)
    args=p.parse_args()
    args.func(args)


if __name__=="__main__":
    main()
