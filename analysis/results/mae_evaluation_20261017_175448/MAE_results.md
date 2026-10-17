# Model MAE: actual saved-model run

The saved assessment model was trained through **31 August 2026**. This run reconstructs
**19–30 September 2026**, covering 12 dates, four zones, and 4,608 quarter-price predictions.
No models were fitted, tuned, or overwritten. Both saved-model directories remain separate;
the full-year submission model was not used to score September.

## What MAE means

MAE = mean(abs(predicted price - observed price)). It is measured in **EUR/MWh**, not percent.
For example, predicting 105 when the observed price is 100 gives an absolute error of 5 EUR/MWh.
Overprediction and underprediction both count positively. Lower is better; zero is perfect.
The official score averages quarter errors per date and zone, then dates within each zone,
then the four zones equally. A mean does not guarantee that every prediction is within that error.

This model's MAE is **5.0443 EUR/MWh** versus **10.8988** for
repeating the hourly average four times: **53.72% less absolute error**.
That reduction is not a percentage accuracy or a return on an investment.

## Same-date comparison

| method | MAE_EUR_MWh | improvement_vs_flat_pct | ZONE_0 | ZONE_1 | ZONE_2 | ZONE_3 |
| --- | --- | --- | --- | --- | --- | --- |
| Flat | 10.8988 | 0.0000 | 10.7839 | 13.3021 | 9.7233 | 9.7860 |
| PCHIP | 6.9270 | 36.4425 | 5.7282 | 8.5809 | 7.4874 | 5.9116 |
| Slope x 2 (fixed) | 6.1087 | 43.9507 | 4.7519 | 7.5154 | 6.3696 | 5.7979 |
| Saved final model | 5.0443 | 53.7171 | 3.8593 | 6.2577 | 5.5064 | 4.5537 |

Flat uses the supplied hourly average for every quarter. PCHIP interpolates a cumulative
hourly curve while preserving each hourly mean. Slope x 2 is the EDA rule with a fixed multiplier,
not a multiplier selected on these September targets. The final model adds 0.8 times learned
zero-sum LightGBM corrections to PCHIP. The corrections describe within-hour slope, curvature,
and zigzag; inputs include hourly geometry, calendar, other zones' hourly curves, and past shapes.

## MAE by zone

| market_id | model | flat | gain_pct |
| --- | --- | --- | --- |
| ZONE_0 | 3.8593 | 10.7839 | 64.2126 |
| ZONE_1 | 6.2577 | 13.3021 | 52.9569 |
| ZONE_2 | 5.5064 | 9.7233 | 43.3690 |
| ZONE_3 | 4.5537 | 9.7860 | 53.4667 |

`model` and `flat` are EUR/MWh; `gain_pct` is the relative error reduction.

## MAE by quarter position

| quarter_index | model | flat |
| --- | --- | --- |
| 1.0000 | 5.9598 | 15.6013 |
| 2.0000 | 3.9670 | 6.0461 |
| 3.0000 | 4.0423 | 5.8131 |
| 4.0000 | 6.2079 | 16.1346 |

Q1 = :00–:15, Q2 = :15–:30, Q3 = :30–:45, Q4 = :45–:60 within each hour.

## MAE by date (four zones equally weighted)

| delivery_date | absolute_error | flat_absolute_error |
| --- | --- | --- |
| 2026-09-19 00:00:00 | 4.6286 | 9.2075 |
| 2026-09-20 00:00:00 | 4.6768 | 8.4596 |
| 2026-09-21 00:00:00 | 6.1440 | 13.1040 |
| 2026-09-22 00:00:00 | 5.7306 | 13.5167 |
| 2026-09-23 00:00:00 | 5.6256 | 12.6064 |
| 2026-09-24 00:00:00 | 4.3104 | 10.2098 |
| 2026-09-25 00:00:00 | 4.5255 | 10.6038 |
| 2026-09-26 00:00:00 | 4.5811 | 9.6846 |
| 2026-09-27 00:00:00 | 4.3498 | 9.4764 |
| 2026-09-28 00:00:00 | 5.2919 | 12.6238 |
| 2026-09-29 00:00:00 | 4.6628 | 9.3161 |
| 2026-09-30 00:00:00 | 6.0041 | 11.9768 |

## Example predictions: first eight quarters of ZONE_0 on 19 September

| delivery_date | market_id | hour_index | quarter_index | price_eur_mwh | predicted_price_eur_mwh | absolute_error |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-19 00:00:00 | ZONE_0 | 1.0000 | 1.0000 | 132.5200 | 124.0903 | 8.4297 |
| 2026-09-19 00:00:00 | ZONE_0 | 1.0000 | 2.0000 | 117.7600 | 108.2991 | 9.4609 |
| 2026-09-19 00:00:00 | ZONE_0 | 1.0000 | 3.0000 | 97.4300 | 96.1429 | 1.2871 |
| 2026-09-19 00:00:00 | ZONE_0 | 1.0000 | 4.0000 | 65.8300 | 85.0077 | 19.1777 |
| 2026-09-19 00:00:00 | ZONE_0 | 2.0000 | 1.0000 | 96.0500 | 92.9764 | 3.0736 |
| 2026-09-19 00:00:00 | ZONE_0 | 2.0000 | 2.0000 | 82.4400 | 82.5204 | 0.0804 |
| 2026-09-19 00:00:00 | ZONE_0 | 2.0000 | 3.0000 | 79.1800 | 76.4952 | 2.6848 |
| 2026-09-19 00:00:00 | ZONE_0 | 2.0000 | 4.0000 | 60.4300 | 66.1080 | 5.6780 |

## Checks and interpretation

- The weights are fixed at the August cutoff. For each target date, the pipeline slices
  quarter-price history to strictly earlier dates; earlier evaluation dates can become history.
  The full target-day hourly curve is a permitted input. Observed target-day quarters are used
  to construct the supplied hourly reference and for scoring, never as shape features.
- Output validation passed; inference used no emergency or per-hour fallback.
- Original assessment artifact hashes were unchanged. Detailed validation and runtime versions
  are in `summary.json`. The runtime differs from the original Python 3.12 environment, so use
  the results from this run rather than assuming bit-identical reproduction of published scores.
- September was already explored during EDA. This reproduces a historical assessment, not a new
  independent test, a forecast of unknown hourly prices, or proof of performance on every day.
- To rerun from the ApplePi folder: `.venv-mae/bin/python analysis/run_mae_evaluation.py`.
  Each run creates its own results directory.
