# AET Neural Wave 2026 — hourly → quarter-hourly shape reconstruction

Reconstructs four 15-minute day-ahead prices per hour whose average equals the
supplied hourly price (tolerance 0.0001 EUR/MWh), using only the official
training set and the jury-supplied history of earlier dates.

## Method

```
P(h, q) = baseline(h, q) + s · residual(h, q)
```

- **Baseline** — the cumulative hourly curve is interpolated with PCHIP and differenced on the
  quarter grid. The four quarters telescope back to the hourly price, so the mean is preserved
  algebraically. Flat and linear-ramp baselines are kept for comparison and as fallbacks.
- **Residual** — zero-sum by construction, predicted either as 3 coefficients on an orthonormal
  basis (slope, curvature, zigzag) with one L1 LightGBM per coefficient (`lgbm_coef`), or as the
  4 quarters with one pooled L1 model followed by centring (`lgbm_quarter`, loss in price space),
  or with five quantile models and the constrained common-τ decision (`lgbm_quantile`, E6).
- **Features** — the zone's hourly curve (levels, ±1/±2 slopes, curvature; optionally hour 1 uses
  the previous day's last hours from history), day statistics, calendar, the same-day curves of all
  zones, the baseline's own shape, and lagged residual shapes (1 and 7 days, 7- and 28-day means)
  from strictly earlier dates. Optional E4 analogue memory (curve-conditioned nearest earlier days).
- **Optional E6 shrinkage** — cross-zone graph smoothing of each hour's shape coefficients.

**Final configuration** (`configs/final.yaml`, chosen on 8 monthly folds of the public replica):
PCHIP + 0.8 × `lgbm_coef` residual, history and cross-zone features, no previous-day edge; E4 and
E6 were evaluated and not promoted. Replica scores (EUR/MWh): development 3.943 (−51.5 % vs flat),
locked September 5.004 (−54.1 %), real days 1–17 Oct 2026 after the freeze 4.939 (−51.5 %).
Details and every rejected idea: `docs/METHODOLOGY.md`.
- **Safety net** — fallback chain learned → baseline → flat per hour, then an exact float64
  projection onto the hourly mean.

## Install and run

### Volatility diagnostics and optional magnitude features

See `docs/VOLATILITY.md` for an official-data-only development experiment, retrospective extreme-hour
evaluation, and a separate heuristic shape-risk CSV. The `volatility_history` features default off;
the existing final configuration and artifacts remain unchanged. Risk labels are not calibrated
probabilities. No synthetic training augmentation is implemented.

### Report exported September assessment predictions

The optional reporting tool reads existing held-out predictions and observations. It does not
train or tune models. From the activated project environment:

```bash
python -m pip install -r requirements-report.txt
python -m tools.report_assessment --assessment results/september_assessment --out results/september_report
```

Open `results/september_report/report.md`. The output folder also contains five PNG figures,
model/valuation/error CSV tables, detailed joined predictions and a validation summary.
Optionally pass `--backtest PATH_TO_LOCKED_PER_DAY_CSV` to compare exported errors against
the original backtest. Representative and difficult example dates are chosen automatically
by median and maximum daily model MAE, respectively. Illustrative valuation errors are not
actual AET positions or trading profits.

### Modelling commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 45 tests, ~12 s, synthetic data only

# 1. Describe the data (baselines, component energy, shape atlas, monthly drift)
python -m aet.eda --train data/train_prices_15min.csv --out results/eda

# 2. Experiment ladder E0-E3; add --folds monthly for 8 development folds (Jan-Aug)
python -m aet.backtest --train data/train_prices_15min.csv --config configs/core.yaml

# 3. One-change-at-a-time ablations against the incumbent
python -m aet.backtest --train data/train_prices_15min.csv --config configs/ablations.yaml --reference E3_coef_hist

# 4. Freeze: point configs/final.yaml at the winner, report the locked block once, train on everything
python -m aet.backtest --train data/train_prices_15min.csv --config configs/final.yaml --include-locked
python -m aet.train --train data/train_prices_15min.csv --config configs/final.yaml --out artifacts

# 5. Jury entry point (offline, no refitting)
python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts --out predictions.csv
```

Every backtest writes `per_day.csv`, `summary.csv` and `run.json` to a timestamped folder and
appends one line per candidate to `results/ledger.jsonl`. The summary reports the official score
(daily MAE → mean per zone → mean over zones), the gain over flat and over the reference,
folds improved, worst zone, paired daily win rate, per-quarter error, fallbacks and runtimes.
The promotion gate from the roadmap (≥0.5 % better, ≥2/3 of folds, no zone >2 % worse) is
applied automatically.

## Layout

| Module | Responsibility |
|---|---|
| `schema.py` | Column contracts, cleaning, training-window guard, output validator (mirror of brief §9) |
| `cube.py` | Dense `[date, zone, hour, quarter]` arrays, strict `before(day)` slicing, history merge |
| `basis.py` | Orthonormal zero-sum bases (`poly`, `haar`), encode/decode |
| `baselines.py` | Flat, linear ramp, cumulative PCHIP / Akima / natural spline |
| `history.py` | Calendar-indexed residual history: lags and window means strictly before each date; E4 analogue memory |
| `features.py` | One feature builder for training, backtesting and inference |
| `models.py` | `ResidualModel` strategies: `none`, `lgbm_coef`, `lgbm_quarter`, `lgbm_quantile` (common-τ) |
| `pipeline.py` | `ShapePipeline`: composition, optional E6 shrinkage, fallback chain, projection, save/load |
| `config.py` | YAML candidates with `extends`, explicit or monthly folds, locked block, gate |
| `backtest.py` / `train.py` / `infer.py` / `eda.py` | Entry points |
| `tools/` | Public-replica downloader (Energy-Charts), official-metric scorer and the fixed autoresearch harness; outside the submission |

## How the rules are enforced in code

- **Training window** — training, backtests and EDA refuse rows dated after 2026-09-30.
- **Causality** — every lookup goes through `QuarterCube.before(day)` and `HistoryStore`, which
  only read calendar slots strictly before the target date. Tests poison all future rows and
  check that features and predictions do not change.
- **Calendar lags** — lags count calendar days, not rows: the day after a DST-excluded date gets a
  missing lag rather than the wrong day.
- **Frozen inference** — inference loads the saved boosters and only recomputes fixed arithmetic
  summaries; nothing is fitted, scaled or tuned on jury data.
- **Short jury history** — the artifacts carry the official training prices as a history store;
  the jury's file is merged over it (jury values win), so lag features keep their training-time
  meaning even if the supplied history is short.
- **Output** — validated before writing and again after reloading the CSV (`%.10f`); any failure
  in reconstruction writes the flat baseline instead, so the jury always receives a compliant file.
- **Reproducibility** — fixed seeds, `deterministic` + `force_row_wise` LightGBM, pinned
  dependencies, manifest with config hash, training CSV hash, date range and library versions.

## Public replica without the official package

`tools/` is **not part of a submission** (the brief forbids external data for participants). It
rebuilds challenge-shaped files from the public Energy-Charts API (Fraunhofer ISE, no key;
the default zones are CC BY 4.0 from Bundesnetzagentur | SMARD.de) so the pipeline can be exercised by anyone:

```bash
python -m tools.fetch_energy_charts --out data --eval-date 2026-10-16      # ~55 API calls, cached
python -m aet.eda --train data/train_prices_15min.csv
python -m aet.backtest --train data/train_prices_15min.csv --config configs/core.yaml --folds monthly
python -m aet.train --train data/train_prices_15min.csv --config configs/final.yaml --out artifacts
python -m aet.infer --hourly data/eval/2026-10-16/test_hourly.csv --history data/eval/2026-10-16/history_15min.csv \
                    --model artifacts --out predictions.csv
python -m tools.score --pred predictions.csv --truth data/eval/2026-10-16/truth_15min.csv
```

Defaults: DE-LU, FR, AT and IT-North (`--zones` to change), market time CET/CEST, train window
1 Oct 2025 – 30 Sep 2026, DST days and any day that is not 96 complete quarters excluded, and only
days complete in every zone kept. Timestamps are read as interval starts; the tool prints an
alignment check (the step across the hour boundary should usually be the largest). Day D is
auctioned on D-1 around 12:45 CET, so `--eval-date 2026-10-19` works from Sunday afternoon.

## Synthetic data

`tests/synthetic.py` generates fake prices to test invariants and wiring. It is never used to
train, validate, calibrate or select anything in the submitted model.

## Experiment configs

| File | Question |
|---|---|
| `configs/core.yaml` | Ladder E0–E3 (+E2b/E3b) |
| `configs/ablations.yaml` | E5 one-change ablations against E3 |
| `configs/memory.yaml` | E4 analogue memory against E3 + scale 0.8 |
| `configs/e6.yaml` | E6 graph shrinkage and constrained quantiles against I2 |
| `configs/round4.yaml` | Narrow ablations against I2 (stopping round) |
| `configs/final.yaml` | Frozen submission (I2) plus references for the locked report |

## Autonomous research

`AUTORESEARCH.md` is a Karpathy-style autoresearch program: an agent edits the `exp` candidate in
`configs/autoresearch.yaml` and the modelling modules, `tools/autoresearch.py` scores each commit
on a search set (days 19..end, gated) and a confirm set (days 1–18, must not lose), paired against the
stored incumbent, and the agent keeps or reverts it, logging every run in `results/autoresearch/<tag>/`.

`RESEARCH.md` extends it into a literature-driven loop: each cycle searches papers on a topic map,
logs what it read (`research/papers.md`), writes idea cards (`research/ideas.md`), implements and
evaluates the best one with the same harness and keep rule, and records the outcome
(`research/journal.md`). Run it unattended with `scripts/research_loop.sh <tag> [cycles]`
(one fresh Claude Code session per cycle, on branch `research/<tag>`).
