# What the adaptive graph experiment tells us

The new graph provides probabilistic and relationship diagnostics, but it does **not** replace the
best point model. On June 19–30, July 19–31 and August 19–31 (38 dates, four equally weighted zones):

| Method | MAE, EUR/MWh |
| --- | ---: |
| Previous fixed 25% graph / 75% AET blend | **4.20575** |
| New selected 10% Bayesian graph / 90% AET blend | 4.21954 |
| AET | 4.22464 |
| New clustered Bayesian graph | 4.33585 |
| Previous optimized graph | 4.34401 |

The new graph is only 0.19% better than the former graph and 2.63% worse than AET. Its new blend
improves AET by just 0.12%, below the existing 0.5% improvement requirement; the previous blend
remains better. Every zone and every confirmation month has higher standalone graph MAE than AET.
Selection used January–May and was frozen before confirmation. These previously inspected
development dates are not a new independent test. No September score is claimed for the new model.

Subsequent assessment: the user later requested a frozen September replay. That separate run gives
new graph **5.11650** and new 10% blend **5.03028**, with no tuning or calibration changes.
See [the September report](../adaptive_probabilistic_september/assessment_results.md); the
development comparisons and conclusions below remain those recorded before that assessment.

## Neighbor insights

The model's zone clusters are stable across the three confirmation folds: ZONE_0, ZONE_1 and
ZONE_2 group together, with ZONE_3 separate. This reflects residual quarter-price shape similarity,
not identified geography or congestion.

ZONE_1 → ZONE_0 has the largest average helpful removal delta: masking it increases ZONE_0 MAE
by **0.01091 EUR/MWh**. Even that small effect changes sign by month: +0.01377 in June, +0.02283
in July, −0.00366 in August. Thus there is no strong, stable causal conclusion to draw.

Attention alone can be misleading. ZONE_0 → ZONE_1 receives average attention around **0.446**,
yet removing it slightly improves ZONE_1 MAE (−0.00650). On the selection folds, the matched
Bayesian no-message control also slightly outperforms the graph (3.95545 versus 3.96106).
The control retains a shared cross-zone regime gate, so it is specifically a graph-message ablation.
Neither observation supports adding graph complexity purely because zones trade together.

The interactive explorer separates attention, error-based edge removal, and hypothetical +10 EUR/MWh
source-curve sensitivities. Arrows are source → receiver. Changes concern reconstructed intrahour
shape; supplied hourly means are fixed. These are model dependencies, not observed trade effects.

## Uncertainty and business interpretation

Raw nominal 90% predictive intervals cover **79.69%** of quarter observations. Training-window
calibration raises that to **89.28%**, with average width **18.48 EUR/MWh**. Zone coverages are
90.90%, 87.69%, 88.24% and 90.30%; ZONE_1 and ZONE_2 still under-cover. Average raw predictive
CRPS is **3.27505**. Calibrated interval score is **27.78954**. These two scores evaluate different
outputs (raw scenarios versus calibrated marginal bands); they do not establish superiority over
a simple probabilistic baseline, which was not tested in this study.

Only 1 of 152 zone-days has all 96 actuals inside the marginal bands. These bands were not designed
as simultaneous whole-day sets. Do not read “90%” as a 90% chance that an entire daily path fits.
All raw sampled scenarios respect the known hourly mean, but the likelihood does not learn a full
cross-zone/hour noise covariance. Negative-price probabilities are uncalibrated model frequencies.

For a dashboard, show MAE alongside coverage and width, flag difficult intrahour periods, and let
users inspect which peer connections help versus merely receive attention. ZONE_1 has the largest
graph MAE (**5.03192**); Q4 and Q1 are harder than Q2/Q3 (5.34778 and 5.26143 versus 3.40989 and
3.32432). This suggests focusing future error analysis on hour-boundary shape changes rather than
adding more generic graph layers. These are reconstruction diagnostics, not demonstrated trading profits.

## Next experiment worth running

1. Keep the former blend/AET as accuracy references. Train a small graph to correct **out-of-fold
   AET errors**, rather than relearning the whole PCHIP correction. Teacher predictions for graph
   training must themselves come from earlier-date fits, otherwise the residual task leaks training fit.
2. Compare sparse graph messages with a matched no-message correction, and select shrinkage using
   inner rolling folds. Preserve zero-sum corrections and require stable improvements across zones
   and months. Use a predeclared outer evaluation; do not tune this model to September's known MAE.
3. Benchmark the Bayesian intervals against a simple rolling residual-quantile baseline on identical
   calibration windows. Then assess regime-specific calibration, negative-price reliability and
   day-level dependence before adding more probabilistic complexity.

The current model, artifacts and explorer are on branch **Paola**, inside `Paola_models/`.
Original AET and former graph artifacts were hash-checked and remain unchanged. Nothing was pushed.
See [the full trial ledger and report](advanced_results.md) and
[architecture and reproducible commands](../../PROBABILISTIC_GRAPH.md).

Validation: **75 tests passed**, including target/future poisoning, calibration isolation, directed
edge masking, coherent uncertainty scenarios, save/load and export checks. Both inference CLIs
passed a synthetic-input smoke check. The static diagnostic plot was visually checked; JavaScript
syntax and explorer control logic were checked separately. Live browser interaction could not be
verified because the session exposes no browser.
