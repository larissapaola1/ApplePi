# Official-data methodology and results

This note describes the configuration in configs/final.yaml and artifacts_official/manifest.json.
It supersedes the replica narrative for the current artifacts. The historical note is retained
verbatim at archive/PUBLIC_REPLICA_METHODOLOGY.md, including its October evaluation record.

## 1. Eligibility and provenance

Read PROVENANCE.md. Earlier external-replica selection and October 2026 evaluation preceded the
official-data work. The current offline pipeline and official training cutoff do not establish
that the complete development process complies with the brief. An organiser interpretation is
needed before presenting this as an eligible submission. An internal freeze is not automatically
the submission freeze. Nothing in this note conceals or retrospectively cures earlier exposure.

## 2. Target and representation

For each supplied hourly value H, predict four quarter-hour prices whose mean is H.
Cumulative interpolation uses C[0]=0, C[h+1]=C[h]+H[h]. A PCHIP interpolator F through these
knots gives quarter q as 4*(F(h+(q+1)/4)-F(h+q/4)). The increments telescope, preserving H.

Observed-minus-PCHIP residuals are encoded in an orthonormal zero-sum basis:
slope [-3,-1,1,3]/sqrt(20), curvature [1,-1,-1,1]/2, zigzag [-1,3,-3,1]/sqrt(20).
Three pooled LightGBM regressors predict the coefficients. Each uses regression_l1,
500 rounds, learning rate 0.05, 15 leaves, minimum leaf size 100 and seed 42.
Coefficient L1 is a surrogate; model selection scores reconstructed prices.

Final prices are PCHIP + 0.8*decoded_residual, followed by a float64 mean projection.
This uses fixed basis vectors, not learned external embeddings.

## 3. Current features and history

The current official model has **prev_day_edge: true**, unlike the historical replica.
Features include zone, hour, weekday/weekend, seasonal sin/cos, local hourly levels,
backward/forward slopes and curvature, daily statistics, cross-zone spreads/slopes,
baseline coefficients, prior residual coefficients at calendar lags 1/7 and 7-/28-day
means with counts. The first hours may use the previous calendar day's known last hours.
The last hours never read next-day prices. Missing history stays missing (NaN).

All target-day hourly curves are supplied together. Quarter-hour lookups are strictly
earlier than the target date. Supplied history is merged over saved official training history,
then sliced before each prediction date. Weights and calibration are not updated at inference.
Optional magnitude features default off and are not in the shipped model.

## 4. Official-data validation

Eight rolling-origin development folds cover January–August 2026. Each model trains through
the preceding month-end and scores days 19 through month-end with earlier-history replay.
The predefined promotion gate requires at least 0.5% lower MAE, improvement on at least
two-thirds of folds, and no zone more than 2% worse. This is an engineering rule, not a
statistical significance guarantee; repeated development comparisons can overfit those folds.

Score = mean over zones of mean daily MAE. Each zone has 25% weight.

| Model | Development MAE | September backtest MAE |
|---|---:|---:|
| Flat | 8.1276 | 10.8988 |
| PCHIP | 5.6185 | 6.9270 |
| Unscaled coefficients + history | 3.9838 | 5.4270 |
| Final, scale 0.8, previous-day edge on | 3.9479 | 5.0444 |

September assessment trained through 31 August and scored 19–30 September only (12 dates).
September aggregate EDA had already been inspected. The official-data ledger records a
single locked scoring run after the core/ablation/memory/E6 comparisons. It was not a wholly
unseen month in every respect. Do not restart model selection based on that result.

Final September backtest zone MAEs: ZONE_0 3.8588, ZONE_1 6.2573, ZONE_2 5.5072, ZONE_3 4.5544.
The final model reduces September MAE by 53.7% versus flat and 27.2% versus PCHIP.

## 5. Official-data experiment decisions

- Scale 0.8 improved 0.90% over unscaled, 6/8 folds: promoted.
- Removing previous-day edge features after scaling worsened 0.16%: retained.
- Removing cross-zone features from the unscaled model worsened 1.43%: retained.
- Haar, Akima and spline substitutions did not improve the complete learned pipeline.
- Analogue memory K=8 improved 0.33%, only 5/8 folds: below gate, off.
- Graph smoothing lambda=0.1 improved 0.19%: below gate; lambda=0.3 worsened 0.75%.
- Quantile alternatives worsened 1.67–2.70%: off.
- Later magnitude-feature extension scored 3.9590 versus 3.9479: off.
- Risk diagnostics are separate development research, not part of the required prediction CSV.

Current configs/e6.yaml describes the official experiment with the edge on. It is not the
historical replica config. Historical core/final/E6/round4 configs from commit 111b489 are
preserved under archive/replica_111b489_*.yaml; use those to understand historical claims.

## 6. Backtest versus exported inference

The exported September assessment predictions score 5.044184 EUR/MWh versus
5.044430 in the original backtest. A diagnostic using identical training arrays and identical
saved boosters reproduced the difference by changing only the supplied hourly inputs:
NumPy means versus CSV-input means differ by at most 1.14e-13 EUR/MWh and can cross tree
thresholds. The overall MAE difference is about 0.000246; the maximum individual prediction
difference observed was about 0.633 EUR/MWh. Do not claim bit-identical backtest and CSV paths.
The official inference smoke run is repeatable for identical files.

Representative/difficult plots and valuation results use exported predictions from a model
trained through August, never the full-year model. Illustrative 1 MWh Q1/Q4 positions have
mean absolute errors about EUR 5.96/6.21 versus flat EUR 15.60/16.13. These are numerical
translations of quarter-specific errors, not actual AET positions, profits or savings.

## 7. Artifacts and reproducibility

artifacts_official/ contains three boosters, history_store.npz and a manifest recording
training dates 2025-10-01..2026-09-30, data/config hashes and versions. The separate
artifacts_september_assessment/ ends 2026-08-31. Current source defaults new optional
volatility_history to false, consistent with the old manifest's absent flag.
No weights were changed by the documentation/fallback correction.

Use Python 3.12 and requirements.txt; run python -m pytest -q in the full repository.
See SUBMISSION.md for portable inference and retraining commands. The official dataset is
provided separately; no code fetches it during inference. Replication is conditional on runtime
versions and valid inputs, not a guarantee across all hardware or environments.

## 8. Robustness and limitations

Outputs require four finite prices per supplied hour, unique quarter keys, no extra hours
and mean error <=0.0001. Checks run before and after CSV writing. With valid hourly input,
model-loading/history/reconstruction failures produce flat predictions and an explicit diagnostic.
Malformed hourly input is rejected rather than silently inventing missing prices or choosing
among duplicates. Missing dependencies and filesystem failures can prevent output.

A strict September training cutoff cannot detect earlier external consultation or selection.
Tests establish software properties, not procedural eligibility. Rare events remain harder,
and 12 dates cannot establish universal generalisation. No calibrated uncertainty is claimed.
