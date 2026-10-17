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
| PCHIP | 5.6185 | 6.9270 |
| Unscaled coefficient model with history | 3.9838 | 5.4270 |
| Current final | **3.9479** | **5.0444** |

EUR/MWh. Development: eight Jan–Aug monthly folds, days 19–month-end. September: train
through August, assess 19–30 September. September aggregate EDA had already been inspected;
the block was held out from the subsequent official model comparisons. Exported inference
scores 5.0442 rather than 5.0444 due to tiny hourly-input rounding differences affecting tree
branches; both paths improve over flat by approximately 53.7%. See docs/METHODOLOGY.md.


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

## Dashboard

An interactive view of the frozen September model (`artifacts_september_assessment`) on
held-out 1–30 September 2026: score tiles, actual vs predicted prices, a per-hour breakdown of
each prediction, and risk flags. Nothing is trained or written to disk.

One-time setup (separate environment, so the pinned versions don't change your base Python):

```bash
cd AET-HACKATHON
conda create -n aet python=3.12 -y
conda activate aet
pip install -r requirements.txt -r requirements-explorer.txt
```

Open it (every time):

```bash
cd AET-HACKATHON
conda activate aet
PYTHONPATH=. python -m streamlit run aet/app.py --server.address localhost
```

`aet/app.py` has three pages in the top bar, in pitch order:
- **Data Story** (`aet/eda_page.py`): what we found, the EDA chapter by chapter, computed from the training CSV.
- **Model Results** (`aet/dashboard_2.py`): how well it works, per zone or all zones, with uncertainty bands.
  Clicking a point opens Understanding the Model with that zone, day and hour selected.
- **Understanding the Model** (`aet/explain_page.py`): why it works: what the three LightGBM models rely on (split gain
  by kind of input), one prediction step by step, and the risk labels.

The original single-page dashboard is still available as `aet/dashboard.py`.

It opens at http://localhost:8501; the first load takes about a minute. Stop it with Ctrl+C.
After editing `aet/dashboard.py`, stop and restart it to see the changes. The training CSV is
found automatically at `../Data/train_prices_15min.csv`; elsewhere, add `-- --train PATH` to the
command.

**Uncertainty (display only).** The green fan shows a 50% band (dark) and a 90% band (light), and
the tiles give the chance that the actual price lands within ±5 €/MWh of the prediction. All come
from the distribution (percentiles 1–99) of honest out-of-sample errors per zone × hour × quarter,
from the Jan–Aug monthly development folds (no September data). The table is stored in
`intervals/september_80.json`; regenerate it (about 20 s) with:

```bash
python -m aet.intervals --train ../Data/train_prices_15min.csv --config configs/final.yaml \
    --out intervals/september_80.json
```

80% band coverage in development (leave one month out) is 77.9%. September 2026 was more volatile
than the calibration months, so the statements are optimistic there: the 90% band held 80.0%, the
50% band 37.0%, and "within ±5 €/MWh" happened 66.7% of the time against 76.7% stated, although
the ranking is right (quarters stated more certain were more often right). The dashboard shows the
stated and measured values side by side for the current selection. Scaling the range by the day's spread or
by recent 7-day difficulty did not improve development coverage, so the simple fixed range is
used. Ranges are not part of the submission file and do not affect the hourly-mean constraint.

## Packaging

`python -m tools.package_submission --out dist/aet_review_bundle.zip` builds an allowlisted
**review bundle**, including the provenance disclosure. It excludes tools/, downloader,
October inputs, archives, explorer and extra experiment configs. It does not certify eligibility.
The complete historical repository remains available for audit; no historical evidence is deleted.
