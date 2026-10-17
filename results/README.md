# Results

[summary.csv](summary.csv) is the curated results table (MAE in EUR/MWh, official metric:
MAE per day and zone → mean over days per zone → mean over the four zones).

| Window | Definition |
|---|---|
| `dev_jan_aug_monthly_folds` | Eight monthly folds Jan–Aug 2026: train on everything before the month, score days 19–month-end. Used for model selection. |
| `sep_19_30_holdout` | Train through 2026-08-31, score 19–30 Sep 2026 with exported inference. Held out from the official model comparisons. |

The September figure for `final` is the exported-inference score (5.0442); the backtest path gives
5.0444 because of tiny hourly-input rounding differences (see [docs/METHODOLOGY.md](../docs/METHODOLOGY.md)).

## Reproduce

```bash
export TRAIN_CSV=data/train_prices_15min.csv
# Development folds (flat, PCHIP and model ladder are candidates in the config)
python -m aet.backtest --train "$TRAIN_CSV" --config configs/core.yaml --folds monthly --out results/development
python -m aet.backtest --train "$TRAIN_CSV" --config configs/final.yaml --folds monthly --out results/final
# Experiments that were not promoted
python -m aet.backtest --train "$TRAIN_CSV" --config configs/experiments/ablations.yaml --reference E3_coef_hist --out results/ablations
```

Everything under `results/` except this README and `summary.csv` is generated and gitignored.
