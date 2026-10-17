# Frozen hybrid: September MAE below 5

**The frozen 25% optimized graph / 75% AET blend scores 4.997939 EUR/MWh MAE.**
This is 0.92% lower error than AET's 5.044274, and only **0.002061 EUR/MWh below 5**.
The pure optimized graph does not reach 5. This result is a **hybrid containing a neural network
and LightGBM**, not a pure-neural score or a guarantee for other dates.

## Fair same-date comparison

| method | MAE_EUR_MWh | reduction_vs_flat_pct | change_vs_AET_pct |
| --- | --- | --- | --- |
| AET LightGBM | 5.0443 | 53.7171 | 0.0000 |
| Original TGNN | 5.7418 | 47.3167 | 13.8289 |
| Optimized TGNN | 5.3230 | 51.1593 | 5.5265 |
| Frozen 25% graph / 75% AET blend | 4.9979 | 54.1423 | -0.9186 |

All rows use Sep 19–30, four equally weighted zones, 12 dates, and 4,608 predictions. Both
component models were trained through August 31. The official score averages daily MAE within
each zone, then the four zones equally. MAE units are EUR/MWh, not percent accuracy.

## Each zone

| market_id | AET LightGBM | Original TGNN | Optimized TGNN | Frozen 25% graph / 75% AET blend |
| --- | --- | --- | --- | --- |
| ZONE_0 | 3.8593 | 4.7710 | 4.0318 | 3.7776 |
| ZONE_1 | 6.2577 | 6.8796 | 6.3049 | 6.1435 |
| ZONE_2 | 5.5064 | 5.9431 | 5.9685 | 5.5153 |
| ZONE_3 | 4.5537 | 5.3736 | 4.9869 | 4.5554 |

The blend's pooled improvement need not imply improvement in every zone; inspect all four.

## Each quarter

| quarter_index | AET LightGBM | Original TGNN | Optimized TGNN | Frozen 25% graph / 75% AET blend |
| --- | --- | --- | --- | --- |
| 1.0000 | 5.9598 | 7.1835 | 6.3749 | 5.8902 |
| 2.0000 | 3.9670 | 4.2075 | 4.1233 | 3.9472 |
| 3.0000 | 4.0423 | 4.2934 | 4.0296 | 3.9950 |
| 4.0000 | 6.2079 | 7.2830 | 6.7644 | 6.1593 |

## What was fixed before this assessment

The optimized architecture and .25 graph weight were selected using January–May development
predictions and recorded in `aet_informed_optimization/selection_before_confirmation.json`.
They were then checked on June–August: blend 4.2057 versus AET 4.2246. That earlier 0.45% gain
failed the 0.5% promotion threshold. Nothing was retuned to turn the September score into a pass.

For this request, the saved optimization was restored from its verified backup directly into
`Paola_models/` on the **Paola branch**. No temporary checkout was recreated. The original README's
September results and all original artifacts were preserved. Both frozen optimized checkpoints
were evaluated exactly as saved. Every model's result is reported, including those above 5.

September had already been explored in EDA and assessed for the original models. This is a
historical assessment of previously fixed recipes, not a wholly untouched test. Repeatedly trying
new settings until this September number falls would make it a tuning score; no such search was
performed for this result. Do not treat the small margin below 5 as a robust performance guarantee.

## Run the hybrid

From the ApplePi root, on Paola:

```bash
.venv-mae/bin/python -m Paola_models.cli infer \
  --model Paola_models/results/aet_informed_optimization/selected_blend_through_august \
  --hourly test_hourly.csv --history history_15min.csv \
  --out Paola_models/NEW_predictions.csv
```

The input dates must follow the August cutoff. Inference does not train, recalibrate, or update
the .25 blend coefficient. The history is sliced strictly before each target date. All four
methods passed complete-key, finite-price, and hourly-mean checks after CSV reload.

Files here include `method_comparison.csv`, `mae_by_zone.csv`, `mae_by_quarter.csv`, `daily_mae.csv`,
`all_predictions_vs_actual.csv`, and `summary.json`. Raw prediction files remain in
`../september_19_30_assessment/`. `summary.json` records the fixed selection, input/weight hashes,
and validation. No AET weights were replaced, no commits were made, and nothing was pushed.
