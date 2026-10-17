# Original temporal graph model: September 19–30 assessment

**TGNN MAE: 5.7418 EUR/MWh. AET LightGBM: 5.0443 EUR/MWh.**
The graph's error is **13.83% higher** than AET on these same observations.
Both models were trained through 31 August; neither was retrained or tuned for this assessment.
This is the original 80-epoch graph associated with 4.2611 January–August validation MAE,
not the later optimized graph or blend retained separately in the backup.

## Same-date, four-zone comparison

| method | MAE_EUR_MWh |
| --- | --- |
| Flat hourly repetition | 10.8988 |
| AET LightGBM | 5.0443 |
| Original TGNN | 5.7418 |

All scores use September 19–30, 12 dates, four zones, **4,608 quarter-price predictions**.
MAE is calculated per date/zone, averaged over dates within each zone, and then equally over
the four zones (25% each). Units are EUR/MWh, not a percentage accuracy. Lower is better.

## Each zone

| market_id | TGNN_MAE | flat_MAE | AET_MAE | TGNN_minus_AET |
| --- | --- | --- | --- | --- |
| ZONE_0 | 4.7710 | 10.7839 | 3.8593 | 0.9118 |
| ZONE_1 | 6.8796 | 13.3021 | 6.2577 | 0.6219 |
| ZONE_2 | 5.9431 | 9.7233 | 5.5064 | 0.4367 |
| ZONE_3 | 5.3736 | 9.7860 | 4.5537 | 0.8199 |

`TGNN_minus_AET` is positive where graph error is higher.

## Each quarter

| quarter_index | TGNN_MAE | flat_MAE | AET_MAE |
| --- | --- | --- | --- |
| 1.0000 | 7.1835 | 15.6013 | 5.9598 |
| 2.0000 | 4.2075 | 6.0461 | 3.9670 |
| 3.0000 | 4.2934 | 5.8131 | 4.0423 |
| 4.0000 | 7.2830 | 16.1346 | 6.2079 |

## Each date (equal-zone averages)

| delivery_date | TGNN_abs_error | AET_abs_error |
| --- | --- | --- |
| 2026-09-19 00:00:00 | 4.5614 | 4.6286 |
| 2026-09-20 00:00:00 | 5.2881 | 4.6768 |
| 2026-09-21 00:00:00 | 6.3988 | 6.1440 |
| 2026-09-22 00:00:00 | 6.4185 | 5.7306 |
| 2026-09-23 00:00:00 | 6.8787 | 5.6256 |
| 2026-09-24 00:00:00 | 5.0761 | 4.3104 |
| 2026-09-25 00:00:00 | 5.6354 | 4.5255 |
| 2026-09-26 00:00:00 | 5.0034 | 4.5811 |
| 2026-09-27 00:00:00 | 4.1583 | 4.3498 |
| 2026-09-28 00:00:00 | 5.4636 | 5.2919 |
| 2026-09-29 00:00:00 | 6.3810 | 4.6628 |
| 2026-09-30 00:00:00 | 7.6388 | 6.0041 |

## Interpretation and checks

- The 3.9479 AET / 4.2611 original-graph figures refer to January–August development folds.
  Compare the two September figures above with each other, not with the development figures.
- All 24 hourly prices are supplied inputs. For each target date, quarter-price history is sliced
  strictly before that date. Earlier assessment dates may become history for later dates.
- No October data, optimized-model restoration, hyperparameter selection, or model fitting occurred.
- Both prediction files passed finite-value, complete-key, and hourly-mean checks. TGNN maximum
  hourly-mean error after CSV reload: 2.5e-11 EUR/MWh (limit 0.0001).
- Both saved August-cutoff models and the unused full-year incumbent have unchanged hashes.
- September was already EDA-exposed. This is the requested historical assessment, not a new
  independent test or evidence for choosing a new configuration based on September.

Files: `tgnn_predictions.csv`, `aet_predictions.csv`, `predictions_vs_actual.csv`, zone/quarter/date
summaries, `method_comparison.csv`, and `summary.json` for provenance and checks.

Reproduce from ApplePi with a new output directory:

```bash
.venv-mae/bin/python -m Paola_models.assess_september --out Paola_models/results/NEW_september_assessment
```
