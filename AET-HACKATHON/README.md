# AET Neural Wave: official-data reconstruction

Reconstruct 96 quarter-hour prices from 24 supplied hourly prices per zone, preserving every
hourly average within 0.0001 EUR/MWh. This is constrained reconstruction, not hourly forecasting.

The active data source is the official participant package in data/.
Development-history disclosure is retained in [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Current model and evidence

The current configuration and supplied `artifacts_official/` use:
- cumulative PCHIP interpolation;
- three L1 LightGBM regressors for zero-sum slope, curvature and zigzag residual coefficients;
- residual scale 0.8;
- historical and cross-zone features; **prev_day_edge: true**;
- no analogue memory, graph smoothing, quantile decision or magnitude-feature extension.

| Method | Official-data development MAE | September backtest MAE |
|---|---:|---:|
| Flat | 8.1276 | 10.8988 |
| Linear ramp ×1 | 5.7580 | — |
| PCHIP | 5.6185 | 6.9270 |
| Linear ramp ×2 | 4.8322 | — |
| Unscaled coefficient model with history | 3.9838 | 5.4270 |
| Current final | **3.9479** | **5.0444** |

EUR/MWh. Development: eight Jan–Aug monthly folds, days 19–month-end. September: train
through August, assess 19–30 September. September aggregate EDA had already been inspected;
the block was held out from the subsequent official model comparisons. Exported inference
scores 5.0442 rather than 5.0444 due to tiny hourly-input rounding differences affecting tree
branches; both paths improve over flat by approximately 53.7%. See docs/METHODOLOGY.md.

On these development folds, PCHIP improves over flat by 30.87%, slope ×2 by 40.55%,
and the final model by 51.43%. Final improves over slope ×2 by 18.30%.
The full-year descriptive EDA uses a different sample (PCHIP approximately 29%);
do not mix its percentages with this table. Dashes mean no assessment reported here.
New local-geometry and extrema-distance features did not pass the development promotion
gate and remain disabled; see [experiment details](docs/GEOMETRY.md).


## Install and test

Use Python 3.12, matching the recorded artifact environment. If using uv:

```bash
uv python install 3.12
uv venv --python 3.12 .venv312
source .venv312/bin/activate
uv pip install -r requirements.txt
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

The official participant dataset and its original README are in data/. Set TRAIN_CSV to that local file.

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

- Report: `python -m pip install -r requirements-report.txt`, then
  `python -m tools.report_assessment --assessment results/september_assessment --out results/september_report`.
- Volatility: docs/VOLATILITY.md. The risk indicator is a separate heuristic, not calibrated
  confidence. Added magnitude features were not promoted.
- Explorer: `python -m pip install -r requirements-explorer.txt`, then
  `python -m streamlit run aet/explorer.py`. Requires local assessment results and risk reference;
  it is not part of the inference bundle.

## Packaging

`python -m tools.package_submission --out dist/aet_review_bundle.zip` builds an allowlisted
**review bundle**, including the provenance disclosure. It excludes tools/, downloader,
October inputs, archives, explorer and extra experiment configs. It does not certify eligibility.
The complete historical repository remains available for audit; no historical evidence is deleted.
