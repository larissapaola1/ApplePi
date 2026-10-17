# Frozen Bayesian graph: September 19–30 assessment

The new standalone Bayesian graph scores **5.11650 EUR/MWh**, a **3.88%** improvement over the
previous optimized graph's 5.32305, but **1.43% higher error than AET**. Its frozen 10% blend
scores **5.03028**, improving AET by only 0.28%. The former 25% blend remains the lowest-scoring
reported model at **4.99794**. The new graph and new blend do not reach MAE below 5.

Calibrated nominal 90% intervals cover **94.12%**, with average width **30.28 EUR/MWh**;
raw Bayesian intervals cover **86.63%**. These September bands are conservative on average
and wider than the June–August bands. Coverage alone does not establish a better uncertainty model.

| method | MAE_EUR_MWh | change_vs_AET_pct |
| --- | --- | --- |
| AET | 5.0443 | 0.0000 |
| Previous optimized graph | 5.3230 | 5.5265 |
| Previous 25% graph blend | 4.9979 | -0.9186 |
| New Bayesian graph | 5.1165 | 1.4320 |
| New frozen 10% Bayesian blend | 5.0303 | -0.2773 |

MAE is EUR/MWh; lower is better. These are the same 12 dates, four equally weighted zones,
and 4,608 quarter-hour predictions as the previous assessment. The new standalone model is
the adaptive clustered Bayesian graph. Its 10% graph / 90% AET blend was selected on January–May,
before June–August confirmation. The previous 25% blend uses the former optimized graph.
No September fitting, calibration, model selection or blend-weight search occurred.

## Zone MAEs

| market_id | AET | Previous optimized graph | Previous 25% graph blend | New Bayesian graph | New frozen 10% Bayesian blend |
| --- | --- | --- | --- | --- | --- |
| ZONE_0 | 3.8593 | 4.0318 | 3.7776 | 3.9265 | 3.8452 |
| ZONE_1 | 6.2577 | 6.3049 | 6.1435 | 6.2536 | 6.2292 |
| ZONE_2 | 5.5064 | 5.9685 | 5.5153 | 5.5138 | 5.4902 |
| ZONE_3 | 4.5537 | 4.9869 | 4.5554 | 4.7720 | 4.5565 |

## Bayesian graph uncertainty

| market_id | raw90_inside | calibrated90_inside | interval_width | interval_score90 | CRPS | whole_zone_day_inside |
| --- | --- | --- | --- | --- | --- | --- |
| ZONE_0 | 0.8793 | 0.9236 | 20.3462 | 23.9654 | 2.8163 | 0.0833 |
| ZONE_1 | 0.8411 | 0.9392 | 37.3930 | 44.9223 | 4.6980 | 0.0833 |
| ZONE_2 | 0.8646 | 0.9635 | 37.4380 | 40.7011 | 4.0581 | 0.1667 |
| ZONE_3 | 0.8802 | 0.9384 | 25.9537 | 32.0523 | 3.5725 | 0.0000 |

Nominal marginal coverage is 90%. Raw intervals come from Bayesian predictive samples;
calibrated bands use the calibration frozen on August 18–31, with network weights trained
only through August 17. AET's assessment weights end August 31. Calibrated coverage should
be considered together with width and interval score. CRPS evaluates the raw scenarios.
These marginal intervals are not simultaneous whole-day guarantees; raw negative-price
probabilities are not separately calibrated. Uncertainty applies to the graph, not either blend.

## Interpretation and audit

This is the explicitly requested historical assessment. September was already EDA-exposed
and assessed for earlier recipes; the result is not a new independent test. It must not be
used to retune the model or select a new September-optimal blend.

The supplied target-day hourly curves are valid inputs. Quarter-price history is sliced
strictly before each target date; earlier assessment actuals can enter later-date history.
All saved point outputs and raw scenarios preserve the supplied hourly means. No October
data was read. Protected artifacts and the original input files match their recorded hashes.
All files are isolated under Paola_models on branch Paola; no commits or pushes were made.

Validation: **75 tests passed**. All five point outputs passed complete-key, finite-value and
hourly-mean checks; maximum mean error after serialization was 5.01e-11 EUR/MWh, below 0.0001.
All raw sampled scenarios also passed the mean constraint. Protected artifact hashes are unchanged.

`predictions_vs_actual.csv` contains every model prediction, actual, signed error and absolute
error. Zone/quarter/date summaries, uncertainty comparisons and frozen graph/scenario exports
are saved alongside this report. Settings, hashes and validation details are in `summary.json`.

Reproduce from ApplePi using a fresh output directory:

```bash
.venv-mae/bin/python -m Paola_models.assess_probabilistic_september --out Paola_models/results/NEW_frozen_bayesian_assessment
```
