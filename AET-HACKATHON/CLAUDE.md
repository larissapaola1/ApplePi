# CLAUDE.md — AET Neural Wave 2026, public replica

## Context
- David is exploring the AET Neural Wave 2026 challenge on his own; **he is not a participant**.
  Reply in Spanish unless he writes in English. He prefers clean, testable, composable code and
  explicit design patterns.
- Task: turn 24 hourly day-ahead prices per zone into 96 quarter-hour prices whose hourly mean
  equals the input (|error| <= 1e-4 EUR/MWh). Metric: daily MAE -> mean per zone -> mean over the
  four zones. Reference: flat baseline (hourly price repeated four times).
- Sources: `docs/source/challenge_brief.pdf` (official rules) and `docs/source/weekend_roadmap.pdf`
  (David's plan). Decisions taken in the planning chat: `docs/HANDOFF.md` — read it before changing
  the design.
- There is no official data. We use a public replica built from Energy-Charts
  (`tools/fetch_energy_charts.py`); results are not comparable with the hackathon leaderboard.

## Rules we keep anyway (they make the replica meaningful)
- Train, validate and select only on 2025-10-01..2026-09-30 (the code refuses later training rows).
- Features for date D read only quarter-hours strictly before D (`QuarterCube.before`,
  `HistoryStore`). Never weaken this; the sentinel tests guard it.
- Inference never refits or recalibrates; it only recomputes fixed summaries over supplied history.
- Every output hour has four finite prices with mean = hourly price (validated before and after
  writing the CSV).

## Architecture
`baseline (mean-preserving) + residual_scale * residual (zero-sum basis)` -> fallback chain
learned -> baseline -> flat -> exact float64 projection. Arrays are `[date, zone, hour, quarter]`;
rows are ordered `(date, zone, hour)`. Baseline, basis, residual model, features, residual scale and
recency weighting are strategies selected in YAML (`configs/`). See the layout table in README.md.

## Commands
```bash
pip install -r requirements.txt && pytest -q          # 45 tests, synthetic data only
python -m tools.fetch_energy_charts --out data --eval-date 2026-10-16
python -m aet.eda --train data/train_prices_15min.csv
python -m aet.backtest --train data/train_prices_15min.csv --config configs/core.yaml --folds monthly
python -m aet.backtest --train data/train_prices_15min.csv --config configs/ablations.yaml --reference E3_coef_hist
python -m aet.train --train data/train_prices_15min.csv --config configs/final.yaml --out artifacts
python -m aet.infer --hourly data/eval/2026-10-16/test_hourly.csv \
  --history data/eval/2026-10-16/history_15min.csv --model artifacts --out predictions.csv
python -m tools.score --pred predictions.csv --truth data/eval/2026-10-16/truth_15min.csv
```

## Status (17 Oct 2026, after the overnight run)
- Real replica fetched (Energy-Charts; zone code is `IT-North`), 363 training dates, alignment OK.
- All roadmap steps done on 8 monthly folds: EDA, ladder E0-E3, E5 ablations, E4 analogue memory
  (`AnalogueMemory` in `history.py`), E6 graph shrinkage and `lgbm_quantile` (common-tau), round 4.
- Frozen final = I2: PCHIP + 0.8 x `lgbm_coef`, history + cross-zone, `prev_day_edge: false`.
  E4/E6 not promoted (kept in code with tests). Search stopped after two rounds without promotion.
- Scores: dev 3.943 (-51.5 % vs flat), locked Sep 5.004 (-54.1 %), 1-17 Oct 4.939 (-51.5 %).
  Every number and table: `docs/METHODOLOGY.md`. Run logs: `results/` (git-ignored).

## Next steps
1. Score 19 Oct from Sunday 18 Oct ~13:00 CET with the frozen `artifacts/` (commands in
   `docs/METHODOLOGY.md` section 10) and add it to section 7.7. Do not retrain or retune.
2. Further improvement only as a new development cycle (re-freeze afterwards): the tuning loop in
   `AUTORESEARCH.md` or the literature-driven loop in `RESEARCH.md` (`scripts/research_loop.sh`).
   Both use the fixed harness `tools/autoresearch.py` and the candidate `configs/autoresearch.yaml`.

## Working agreements
- Ablate one change at a time; when results are close keep the simpler method.
- `tests/synthetic.py` is for tests only, never for choosing anything.
- Keep it a small package, not a framework. Run `pytest -q` after every change.
