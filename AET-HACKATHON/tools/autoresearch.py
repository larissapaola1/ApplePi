"""Fixed evaluation harness for the autoresearch loop (AUTORESEARCH.md). Never edit it during a run.

    python -m tools.autoresearch --state results/autoresearch/<tag> --init      # store the incumbent
    python -m tools.autoresearch --state results/autoresearch/<tag>             # evaluate HEAD
    python -m tools.autoresearch --state results/autoresearch/<tag> --promote   # last run -> incumbent

The `exp` candidate of configs/autoresearch.yaml is trained on every monthly fold (train to the end
of Dec 2025 .. Jul 2026) and scores every day of the following month with history replay:
- SEARCH set: days >= 19 (the development folds); the promotion gate is applied here.
- CONFIRM set: days 1-18 of the same months. Different dates, same fold models, never gated:
  a kept change must not lose there, which guards against fitting the search days' noise.
Results are paired with the stored per-day results of the incumbent, so a code change that also
alters the old candidate cannot fool the comparison. The locked September block and October 2026
are never read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from aet.backtest import evaluate, summarize
from aet.config import candidate_configs, gate_from_config, load_config, monthly_folds
from aet.cube import QuarterCube
from aet.schema import check_training_window, read_quarter_hourly

CANDIDATE = "exp"
SIMPLER_TOLERANCE_PCT = -0.1   # a simplification may cost at most this much, on both sets
MAX_MEAN_ERROR = 1e-6          # far inside the official 1e-4 tolerance
INCUMBENT, LAST = "incumbent_per_day.csv", "last_per_day.csv"


def split(per_day: pd.DataFrame, first_search_day: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = pd.to_datetime(per_day["delivery_date"]).dt.day
    return per_day[day >= first_search_day], per_day[day < first_search_day]


def compare(new: pd.DataFrame, incumbent: pd.DataFrame, gate: dict) -> pd.Series:
    """Official score of `new`, its gain over `incumbent` and the promotion gate, paired by day."""
    both = pd.concat([incumbent.assign(candidate="incumbent"), new.assign(candidate=CANDIDATE)],
                     ignore_index=True)
    timings = pd.DataFrame({"candidate": ["incumbent", CANDIDATE], "train_s": 0.0, "infer_s": 0.0})
    return summarize(both, timings, "incumbent", gate).set_index("candidate").loc[CANDIDATE]


def verdict(search: pd.Series, confirm: pd.Series) -> dict:
    compliant = search["max_mean_err"] <= MAX_MEAN_ERROR and confirm["max_mean_err"] <= MAX_MEAN_ERROR
    return {
        "improves": bool(compliant and search["gate"] == "PASS" and confirm["vs_ref_pct"] >= 0.0),
        "not_worse": bool(compliant and search["vs_ref_pct"] >= SIMPLER_TOLERANCE_PCT
                          and confirm["vs_ref_pct"] >= SIMPLER_TOLERANCE_PCT),
    }


def _commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", required=True, help="results/autoresearch/<tag>")
    parser.add_argument("--train", default="data/train_prices_15min.csv")
    parser.add_argument("--config", default="configs/autoresearch.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true", help="evaluate HEAD and store it as the incumbent")
    mode.add_argument("--promote", action="store_true", help="the last evaluated run becomes the incumbent")
    args = parser.parse_args(argv)
    state = Path(args.state)

    if args.promote:
        shutil.copyfile(state / LAST, state / INCUMBENT)
        shutil.copyfile(state / "last_run.json", state / "incumbent.json")
        print("promoted:", json.loads((state / "incumbent.json").read_text())["commit"])
        return 0
    if args.init and (state / INCUMBENT).exists():
        parser.error(f"{state / INCUMBENT} exists; use a new --state tag")
    if not args.init and not (state / INCUMBENT).exists():
        parser.error("no incumbent yet: run with --init first")

    started = time.perf_counter()
    raw = load_config(args.config)
    config = candidate_configs(raw, [CANDIDATE])[0]
    spec = raw.get("folds", {})
    first_search_day = int(spec.get("score_from_day", 19))
    folds = monthly_folds(spec.get("first_train_end", "2025-12-31"), spec.get("last_train_end", "2026-07-31"), 1)
    df = read_quarter_hourly(args.train, strict=True)
    check_training_window(df)
    cube = QuarterCube.from_long(df, config.zones)
    per_day, timings = evaluate(cube, config, folds)
    state.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(state / LAST, index=False)
    run = {"commit": _commit(), "config_sha256": raw["_sha256"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (state / "last_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    if args.init:
        shutil.copyfile(state / LAST, state / INCUMBENT)
        shutil.copyfile(state / "last_run.json", state / "incumbent.json")

    gate = gate_from_config(raw)
    # Both sides go through the same exact CSV round trip, so identical runs tie exactly.
    per_day = pd.read_csv(state / LAST, float_precision="round_trip")
    incumbent = pd.read_csv(state / INCUMBENT, float_precision="round_trip")
    s_new, c_new = split(per_day, first_search_day)
    s_inc, c_inc = split(incumbent, first_search_day)
    search, confirm = compare(s_new, s_inc, gate), compare(c_new, c_inc, gate)
    lines = {
        "commit": run["commit"],
        "search_score": f"{search['score']:.6f}",
        "search_vs_incumbent_pct": f"{search['vs_ref_pct']:+.3f}",
        "folds_better": search["folds_better"],
        "worst_zone_pct": f"{search['worst_zone_pct']:+.3f}",
        "day_win_rate": f"{search['day_win_rate']:.3f}",
        "gate": search["gate"] if not args.init else "ref",
        "confirm_score": f"{confirm['score']:.6f}",
        "confirm_vs_incumbent_pct": f"{confirm['vs_ref_pct']:+.3f}",
        "max_mean_err": f"{max(search['max_mean_err'], confirm['max_mean_err']):.2e}",
        "fallback_hours": int(search["fallback_hours"] + confirm["fallback_hours"]),
        "train_s": f"{timings['train_s'].sum():.1f}",
        "infer_s": f"{timings['infer_s'].sum():.1f}",
        **{k: ("yes" if v else "no") for k, v in verdict(search, confirm).items()},
        "seconds": f"{time.perf_counter() - started:.1f}",
    }
    print("---")
    for key, value in lines.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
