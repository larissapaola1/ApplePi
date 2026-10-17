# AET-informed graph optimization

The graph improved, but the existing AET model remains the stronger standalone model.
The best blend has a small positive development result that does not meet the promotion gate.

**Subsequent fixed-recipe September assessment:** hybrid 4.99794, optimized graph 5.32305,
AET 5.04427 EUR/MWh. No settings were changed for that assessment; the model and .25 weight
were reused as already selected. [September report](results/frozen_hybrid_september/below_5_results.md).

| Model | June–August MAE (EUR/MWh) |
|---|---:|
| Original graph | 4.5369 |
| Optimized graph | 4.3440 |
| AET LightGBM | 4.2246 |
| 25% optimized graph + 75% AET | 4.2057 |

The optimized graph reduces its original error by **4.25%** on these same confirmation dates.
The blend reduces AET error by **0.45%**, improves all three months and all four zones, but falls
below the required **0.5%** gain. No automatic promotion is justified. These confirmation months
were already inspected in earlier development work; they are not an independent final test.

## What was tried

Three choices were declared before training:

1. Add AET's explicit cross-zone hourly spreads and slopes to the original graph's features.
2. Add those features and copy AET's coefficient-L1 objective and fixed 0.8 correction scale.
3. Use those settings in a smaller one-block graph with a direct local linear output path.

The third recipe won selection on January–May: **3.9695** versus original graph **4.0864**.
Adding features alone scored 4.1998; features plus loss/shrinkage scored 4.0964. Neither was a
pure improvement in that block. The winning recipe has **7,032 parameters**, compared with
17,347 originally, hidden width 24, one block, and fixed 120 epochs. Its local path preserves
a direct connection from normalized input magnitudes to coefficient outputs alongside graph
and temporal messages. Because multiple settings change together in this candidate, the result
does not isolate the causal contribution of the local path or any individual setting.

The 0.8 multiplier scales learned corrections to PCHIP, not the hourly level or the complete
within-hour shape. Coefficient L1 is a surrogate; all decisions use reconstructed-price MAE.

## Selection and confirmation

January–May predictions selected the graph recipe and a graph blend weight from
`[0, 0.1, 0.25, 0.5, 1]`. The exact choice was written to
`selection_before_confirmation.json` before June–August scoring. No retuning followed.
Each monthly fold trains only through the preceding month-end; input quarter-price history
stops strictly before each target date. September and October were excluded.

The blend combines **predicted prices**, not MAE values. This can help when models make
different errors: a graph worse on its own may still add a small amount of useful information.
The gain remains small and may not generalize. Selection results are optimistic after comparing
candidates; the confirmation block has also been seen in earlier project work.

The existing gate was adapted to the three confirmation folds: >=0.5% improvement, wins on
>=2/3 folds, and no zone >2% worse. The blend met the fold/zone checks but failed the gain check.

## Files

- [Complete experiment report and charts](results/aet_informed_optimization/optimization_results.md)
- [Declared candidate settings](results/aet_informed_optimization/plan.json)
- [Selection locked before confirmation](results/aet_informed_optimization/selection_before_confirmation.json)
- [All selected graph, AET, and blend predictions versus actual](results/aet_informed_optimization/selected_predictions.csv)
- [Confirmation results by zone](results/aet_informed_optimization/confirmation_zone_scores.csv)
- `results/aet_informed_optimization/selected_graph_through_august/`: separate optimized graph artifact.
- `results/aet_informed_optimization/selected_blend_through_august/`: graph plus a copy of the August-cutoff AET artifact.

Both artifacts are frozen through August. Neither was scored on September to choose the model.
The two original AET artifact directories were verified unchanged using SHA-256. Existing graph
checkpoints remain compatible; all new configuration fields default to the original behavior.

## Run it

This work was originally created in a separate checkout, then archived and that checkout removed
at the user's request. It is now restored directly into `Paola_models/` on **Paola**.
Run from the ApplePi root. No commit, merge, or push was performed.

Use supplied hourly inputs and earlier history for dates after the artifact cutoff:

```bash
.venv-mae/bin/python -m Paola_models.cli infer \
  --model Paola_models/results/aet_informed_optimization/selected_blend_through_august \
  --hourly test_hourly.csv --history history_15min.csv \
  --out Paola_models/NEW_predictions.csv
```

To use the pure graph instead, set `--model` to `selected_graph_through_august`.
Inference does not update weights, scalers, or the blend coefficient. All hourly means are
preserved and the output is validated before writing and after CSV reload.

To fit the selected graph recipe on a new allowed cutoff, without repeating model selection:

```bash
.venv-mae/bin/python -m Paola_models.cli train \
  --recipe Paola_models/results/aet_informed_optimization/selection_before_confirmation.json \
  --train Data/train_prices_15min.csv --train-end 2026-08-31 \
  --out Paola_models/artifacts/NEW_selected_graph
```

The `train` command creates the pure graph, not a new blend. A blend requires matching-cutoff
AET and graph components; the supplied `save_hybrid` validates those cutoffs. Output directories
must be new, so prior results are never overwritten. Keep the existing dependency file.

To reproduce this bounded experiment, choose a new result directory:

```bash
.venv-mae/bin/python -m Paola_models.optimize \
  --out Paola_models/results/NEW_aet_informed_optimization
```

It reuses preserved out-of-fold AET predictions in the earlier development error report, verifies
their official score and their targets against the supplied CSV, and records the source hash.

## Verification

**13 Paola-model tests pass**, including the new feature/loss/shrinkage variant, backward-compatible
checkpoint loading, blend serialization, no current/future-history leakage, and recipe-based CLI
training/inference. **All 54 existing AET tests pass.** Both pure graph and blend outputs passed
complete-key, finite-value, and hourly-mean validation on the development predictions after CSV reload.
The final saved hybrid also passed a CLI smoke check using explicitly synthetic constant hourly
values at a future dummy date. Those values were only an interface test, never training data or a
reported performance score. No actual September/October evaluation prices were used.
