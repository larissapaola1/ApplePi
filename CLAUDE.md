# CLAUDE.md — AET Hackathon Challenge (Neural Wave 2026)

## The challenge in one sentence
For each of 4 anonymized European day-ahead market zones, take a day's **24 hourly prices** and reconstruct the **96 quarter-hourly (15-min) prices** (4 per hour), so that **the 4 quarters of every hour average exactly to the given hourly price**.

This is NOT price forecasting. The hourly level is given. We only predict the **intra-hour shape**.

Proponent: Azienda Elettrica Ticinese (AET), Ticino electricity producer/wholesaler/trader.

## Why it matters (business context)
- The European day-ahead market (SDAC) switched from hourly to 15-min prices on 1 Oct 2025.
- Most history, forecasts and long-term forward curves exist only at hourly resolution.
- Repeating the hourly price 4 times (flat) hides intra-hour dynamics (e.g. sawtooth / ramp patterns), which matter for valuing flexible assets (hydro), trading 15-min products, pricing contracts and training 15-min models.
- Because the solution must also work on forward curves and long horizons, **no load/generation/weather data is allowed**. Only prices.

## Data
Training file: `train_prices_15min.csv`
- Columns: `delivery_date, market_id, hour_index, quarter_index, price_eur_mwh`
- `market_id` ∈ {ZONE_0, ZONE_1, ZONE_2, ZONE_3}; `hour_index` 1–24; `quarter_index` 1–4
- Period: 2025-10-01 to 2026-09-30, **363 valid days per zone** (the two DST days are removed). Every day has exactly 24 hours / 96 quarters.
- Times are CET/CEST market time, grouped by delivery date.
- Unit: EUR/MWh. Source: ENTSO-E Transparency Platform, processed by AET. Prices can be negative and spiky.
- Hourly price = arithmetic mean of the 4 quarters (compute it ourselves for training).

At inference the jury provides, per evaluated date:
- `test_hourly.csv`: `delivery_date, market_id, hour_index, hourly_price_eur_mwh` (the 24 hourly prices of the day)
- `history_15min.csv`: same schema as the training file, **only dates before the evaluated date**. May include dates after the training period (e.g. 1–18 Oct 2026). Exact length of history is not specified, so code must handle short/missing history gracefully (fallback to artifacts derived from training data).

## Required output
One CSV with one row per `delivery_date, market_id, hour_index, quarter_index`, column `predicted_price_eur_mwh`.
- 4 zones × 24 hours × 4 quarters = **384 rows per day**, hours in chronological order.
- **Hard constraint:** for every (date, zone, hour), |mean(4 predictions) − hourly_price| ≤ **0.0001 EUR/MWh**. All values finite and numeric, no duplicates, no missing rows.
- Any violation → submission is non-compliant and not scored. No automatic correction by the jury.
- Always finish with a re-centering step and an explicit validation check before writing.

## Evaluation
- **Quantitative:** MAE on 15-min prices. Computed per day and zone → averaged over days per zone → simple mean over the 4 zones (25% each). Lower is better.
- Diagnostics they publish: error by zone and by quarter position, error distribution, worst cases, constraint compliance, comparison with the **flat baseline** (hourly price repeated 4×).
- **Qualitative:** simplicity, robustness, interpretability, methodological clarity, reproducibility. Complexity must be justified by results; unnecessary complexity can be penalized.
- Confirmed secret test date: **2026-10-19**, plus possibly other undisclosed dates. Don't overfit to one date.

## Rules (critical)
Development (before freeze):
- Use ONLY `train_prices_15min.csv` for training, validation, calibration, hyperparameter selection.
- **NEVER consult or use prices from 1–18 October 2026**, even indirectly (no websites, charts, news).
- No external data of any kind (weather, load, generation, other prices, online archives).
- **No pretrained models, external weights, transfer learning or embeddings.** Standard libraries, statistical methods, classical ML and own code are fine. Neural nets trained from scratch are technically allowed but discouraged (little data, jury prefers simplicity).

Submission / freeze:
- Code, config, dependencies (`requirements.txt`), fixed seeds, regenerable artifacts.
- Separate entry points for training and inference.
- Must run **offline, without code modifications, without undocumented manual steps**. The jury may retrain from scratch.
- Nothing changes after submission.

Inference (run by the jury):
- May use the 24 hourly prices of the evaluated day (the whole curve, all hours) and `history_15min.csv` as **inputs to rules fixed in code** (e.g. lag features).
- Must NOT use 15-min prices of the evaluated day, retrain, recalibrate, re-tune, or access the network.

## Modeling approach
Work with **residuals**: `residual_q = price_q − hourly_price`. The 4 residuals of an hour sum to zero. Predict residuals, re-center (subtract their mean), add the hourly price back.

Feature ideas (only from the day's hourly curve and earlier days):
- Curve: hourly price, previous/next hour (and ±2), jump in/out, slope = (H_next − H_prev)/2, curvature = H_next − 2H + H_prev, position vs daily mean/min/max/std.
- Calendar: hour of day, weekday, weekend flag, month / day-of-year (sin/cos), zone.
- Lags from history: residuals at the same hour yesterday, mean/median of last 7/28 days at the same hour.
- Edges: hour 1 has no previous hour within the day and hour 24 has no next hour (the next day is NOT available). Lags must only use strictly earlier dates.

Model ladder (report MAE of each on the same validation split):
1. Flat baseline (residuals = 0) — official benchmark.
2. Slope interpolation: residual_q = slope × offset_q, offsets = (−0.375, −0.125, +0.125, +0.375). No training, automatically mean-preserving.
3. Historical average shape per zone × hour × quarter (× weekday type), optionally blended with recent history.
4. Ridge regression (+ quantile/median regression for comparison, since the metric is MAE).
5. Gradient boosting (CatBoost or LightGBM) with MAE/L1 loss.
6. Optional: small ensemble of the best, re-centered.

Validation: **time-based split only**, e.g. train Oct 2025–Jul 2026, validate Aug–Sep 2026. Never random splits. Lag features must be computed without peeking at the validated day.

## Roadmap
0. Setup: repo structure, requirements, validation split agreed, ask organizers about history length and bundling the training CSV.
1. EDA: residual shapes by hour, zone, weekday/weekend, season; spikes and negative prices.
2. Baselines (flat, slope) + pipeline skeleton (`predict.py` working end to end in the jury's format with constraint check).
3. Feature engineering.
4. Models: ridge/quantile regression, CatBoost/LightGBM (MAE loss), always re-centered.
5. Evaluation: MAE per zone and per quarter position, worst days, feature importance; pick final model weighing score vs simplicity.
6. Final training on ALL data with fixed seeds; simulate the jury (fresh env, offline, fake test files e.g. 2026-09-30 as test day, one command, 384 rows/day, tolerance check).
7. Add-ons only if 1–6 are done: uncertainty flag per hour, dashboard / small web app.
8. Deliverables: README with exact commands, methodology note (features, model, validation, post-processing, MAE table), presentation, video script.

## Suggested repo layout
```
data/                 # train_prices_15min.csv (not committed if large)
src/
  data.py             # loading, hourly averages, residuals
  features.py         # feature engineering (shared by train and predict)
  models.py           # baselines + ML models
  postprocess.py      # re-centering + constraint check
  evaluate.py         # official-style MAE (day&zone → zone → mean)
artifacts/            # saved model + saved shape statistics (regenerable)
train.py              # entry point: retrain from scratch
predict.py            # entry point: --hourly test_hourly.csv --history history_15min.csv --output predictions.csv
requirements.txt
README.md
METHODOLOGY.md
```

## Conventions for Claude Code
- Features must be computed by the SAME code in training and inference.
- Never introduce external data, internet calls, or pretrained weights.
- Never look up or use October 2026 prices.
- Keep solutions simple and explainable; justify any added complexity with validation MAE.
- Always run the constraint check before writing predictions.
