# AET Neural Wave: official-data reconstruction

Reconstruct 96 quarter-hour prices from 24 supplied hourly prices per zone, preserving every
hourly average within 0.0001 EUR/MWh. This is constrained reconstruction, not hourly forecasting.

The only data source is the official participant package; place it in data/ (see [data/README.md](data/README.md)).

## Team ApplePi

- Cristina Fernández-Simal Bernard
- Paola Hernandez Llamas
- Andrea Mero Plaza
- Laura Jimena Tagle Muñoz

## Current model and evidence

The current configuration (I3) and supplied `artifacts_official/` use:
- cumulative PCHIP interpolation;
- three L1 LightGBM regressors (1000 rounds) for zero-sum slope, curvature and zigzag residual coefficients;
- residual scale 0.8;
- historical and cross-zone features; **prev_day_edge: true**;
- fixed mean-preserving post-processing: zones within 0.1 EUR/MWh share one quarter-hour shape,
  zones within 5 EUR/MWh move 40% toward their cluster shape, and quarters keep the sign of their
  hour (floor -0.01 for positive hours, ceiling 0 for negative hours);
- no analogue memory, graph smoothing, quantile decision or magnitude-feature extension.

| Method | Official-data development MAE | September backtest MAE |
|---|---:|---:|
| Flat | 8.1276 | 10.8988 |
| Linear ramp ×1 | 5.7580 | — |
| PCHIP | 5.6185 | 6.9270 |
| Linear ramp ×2 | 4.8322 | — |
| Unscaled coefficient model with history | 3.9838 | 5.4270 |
| I2, previous final (no post-processing, 500 rounds) | 3.9479 | 5.0444 |
| **I3, current final** | **3.8883** | **5.0032** |

EUR/MWh. Development: eight Jan–Aug monthly folds, days 19–month-end. September: train
through August, assess 19–30 September. September aggregate EDA had already been inspected,
and September was scored a second time when switching from I2 to I3, so it is not an untouched
holdout for I3 (docs/METHODOLOGY.md section 4). Exported inference scores 5.0035 rather than
5.0032 due to tiny hourly-input rounding differences affecting tree branches and tie tolerances;
both paths improve over flat by approximately 54.1%.

On these development folds, PCHIP improves over flat by 30.87%, slope ×2 by 40.55%,
and the final model by 52.16%. Final improves over slope ×2 by 19.53%.
The full-year descriptive EDA uses a different sample (PCHIP approximately 29%);
do not mix its percentages with this table. Dashes mean no assessment reported here.
New local-geometry and extrema-distance features did not pass the development promotion
gate and remain disabled; see [experiment details](docs/experiments/GEOMETRY.md).


## Install and test

Use Python 3.12, matching the recorded artifact environment. If using uv:

```bash
uv python install 3.12
uv venv --python 3.12 .venv312
source .venv312/bin/activate
uv pip install -r requirements-dev.txt
python -m pytest -q
```

Dependencies must already be installed for offline execution.

## Run the supplied model

From this repo root, with organiser-supplied input files:

```bash
python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts_official --out predictions.csv
```

No training is needed. All required model files are in artifacts_official/. See
[portable execution instructions](docs/SUBMISSION.md).

## Retrain or reproduce development

Place the official participant dataset in data/ (see [data/README.md](data/README.md)).

```bash
export TRAIN_CSV="data/train_prices_15min.csv"
python -m aet.train --train "$TRAIN_CSV" --config configs/final.yaml --out artifacts_retrained
python -m aet.backtest --train "$TRAIN_CSV" --config configs/final.yaml --folds monthly --out results/reproduced_development
```

Training writes to artifacts_retrained and does not overwrite artifacts_official.
Reopening September is for audit/reproduction only, never another tuning round.
The inference output uses the official schema and is checked before writing and after reloading.

## Safeguards and limits

The mean-preserving baseline plus zero-sum correction enforces consistency algebraically.
A final float64 projection removes numerical drift. Calendar-based history excludes the
target date and all future dates. Model weights are never refitted at inference.

With valid hourly input and installed dependencies, artifact/history/prediction failures fall
back to hourly repetition and emit an emergency diagnostic. Invalid hourly references cannot
be repaired safely; malformed inputs, missing runtime dependencies and write failures remain
errors. Do not claim the pipeline always writes a compliant file under every failure.

## Optional analysis

- EDA: `python -m pip install -r requirements-analysis.txt`; see [analysis/README.md](analysis/README.md).
- Report figures: `python -m tools.report_assessment --assessment results/september_assessment --out results/september_report`.
- Volatility: [docs/experiments/VOLATILITY.md](docs/experiments/VOLATILITY.md). The risk indicator is a separate
  heuristic, not calibrated confidence. Added magnitude features were not promoted.
- Dashboard: see [dashboard/README.md](dashboard/README.md). Not part of the inference path.
- Tests: `python -m pip install -r requirements-dev.txt && python -m pytest -q`.

## Repository layout

| Path | Content |
|---|---|
| `aet/` | Model package: training (`aet.train`), inference (`aet.infer`), backtest, features, history |
| `configs/` | `core.yaml` (folds, candidates), `final.yaml` (frozen submission), `experiments/` (not promoted) |
| `artifacts_official/` | Trained model used for submission (full training year) |
| `data/` | Place the official CSV here; not committed |
| `analysis/` | EDA notebook, reports and figures |
| `dashboard/` | Optional Streamlit dashboard |
| `docs/` | Methodology, submission contract, provenance, AI usage, references, experiment notes |
| `tests/` | Unit and end-to-end tests (synthetic data, no network) |
| `tools/` | Official-metric scorer and report figures |
| `results/` | Curated results table and reproduction commands |
| `report/`, `presentation/`, `video/` | Deliverables |
