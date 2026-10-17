# Paola's model experiments and possible improvements

The strongest measured candidate is the **25% optimized graph / 75% AET blend**. It scores
**4.20575 EUR/MWh** on June–August confirmation and **4.99794** on September 19–30.
The standalone graph models have not beaten AET on those comparisons. The newest AET error
corrections improve June–August slightly, but both worsen September MAE: **5.06125 with graph
messages** and **5.05213 without**, versus AET **5.04427**. The existing development promotion
threshold has not been met, and the previous fixed blend remains the accuracy reference.

The main lessons from Paola's experiments are:

- **Smaller networks and shrinkage helped.** The compact graph improved substantially on the
  original graph, and a small contribution from it improved the AET blend.
- **More graph complexity did not produce better point accuracy.** Adaptive attention, clustering
  and partial Bayesian inference supplied useful diagnostics but did not beat AET alone.
- **Error correction needs stronger generalization evidence.** Its no-message version performed
  better than its graph version, and neither carried its small development gain into September.
- **Anomaly prediction is a proposed next experiment.** The existing dashboard risk indicator is
  heuristic; a calibrated classifier for unusually large AET errors has not been trained yet.

This summary covers the models developed for Paola in `Paola_models/`, with AET as the reference.
It summarizes saved results; no new fitting, assessment or model selection was performed to write it.
All MAEs are **EUR/MWh**, rounded from saved scores to five decimals, with lower values better
and each of the four zones weighted equally.

**The main comparison: identical dates within each column**

| Model | What it does | June–August confirmation MAE | September 19–30 MAE |
| --- | --- | ---: | ---: |
| AET reference | PCHIP plus three LightGBM shape coefficients, price-history and cross-zone features | 4.22464 | 5.04427 |
| Original temporal graph | Two temporal/graph blocks reconstruct the correction to PCHIP | 4.53688 | 5.74184 |
| Optimized compact graph | Smaller one-block graph, direct local path, AET-style features and coefficient loss | 4.34401 | 5.32305 |
| 25% optimized graph / 75% AET | Frozen convex blend of the two complete predictions | **4.20575** | **4.99794** |
| Clustered Bayesian graph | Adaptive directed attention, regime clusters, partial Bayesian weights/edges and Student-t uncertainty | 4.33585 | 5.11651 |
| 10% Bayesian graph / 90% AET | Frozen convex blend of Bayesian graph point predictions with AET | 4.21954 | 5.03028 |
| AET + graph error correction | AET plus 0.25 times a graph trained on chronological out-of-fold AET errors | 4.21623 | 5.06125 |
| AET + no-message error correction | Same correction task, using local temporal/features without graph messages; strength 0.25 | 4.20680 | 5.05213 |

June–August means days 19 through month-end in each month: **38 dates and 14,592 quarter prices**,
with each fold trained through the preceding month-end. September means **12 dates and 4,608
quarter prices**, using saved August-cutoff artifacts. The Bayesian graph reserves August 18–31
for interval calibration, so its network weights end August 17; the other September components
fit weights through August 31. These differing training recipes are part of the frozen candidates.

Architecture/blend/correction-strength choices used January–May, then were frozen before
June–August confirmation. September was assessed afterward without fitting or tuning. These
development dates and September EDA had been inspected previously; they are not a wholly
independent final test. Do not average the two score columns or compare them as equal-difficulty periods.

The below-5 result belongs to the **hybrid**, not a standalone GNN. Its margin below 5 is only
0.00206 EUR/MWh on September. This is not a guarantee of sub-5 performance on other dates.

Sources: [original graph](results/tgnn_80epochs/results.md),
[compact graph/blend study](results/aet_informed_optimization/optimization_results.md),
[Bayesian study](results/adaptive_probabilistic_study/advanced_results.md),
[frozen September comparison](results/adaptive_probabilistic_september/assessment_results.md),
[AET error-correction study](results/aet_residual_graph_study/residual_results.md), and its
[subsequent frozen September assessment](results/aet_residual_graph_september/assessment_results.md).

**How the model families differ**

The original and compact GNNs learn a full correction to the smooth PCHIP reconstruction.
The compact model improved the original graph through a smaller architecture, a direct local
prediction path, explicit cross-zone features, coefficient L1 loss and 0.8 shrinkage. It still
underperforms AET alone. Combining its different errors with AET provides the best measured blend.

The Bayesian graph adds probabilistic outputs and more flexible zone relationships. It improves
on the compact standalone graph on September by **3.88%**, but remains **1.43% worse than AET**.
Its main additional outputs are uncertainty scenarios, intervals, regime clusters and graph
diagnostics. These benefits do not establish better point-prediction MAE.

The newest residual model learns **actual quarters minus an earlier-fit AET prediction**. Monthly
out-of-fold teachers prevent AET from fitting its own correction-training target dates. At inference,
the formula is **AET + 0.25 × predicted AET error**. This is an additive error model rather than
the earlier blend of two independently reconstructed curves. The final correction uses 303 OOF
dates and a 3,710-parameter network.

On June–August, graph error correction improves AET by **0.199%**; its no-message control improves
AET by **0.422%**. The latter improves all four zone averages and wins two of three months,
but misses the 0.5% promotion requirement and remains slightly worse than the previous blend.
At full correction strength, both worsen MAE: graph **4.33846**, no-message **4.27548**.
Applying a noisy estimated correction at full strength is not automatically beneficial.

The subsequent requested September assessment does not reproduce these small development gains:
graph correction scores **5.06125** (**0.336% worse than AET**) and no-message correction
**5.05213** (**0.156% worse**). Both improve ZONE_2 but worsen the other three zone averages.
Both kept their frozen 0.25 strength; this was assessment only. The earlier fixed blend remains best.

AET already uses other zones' hourly spreads and slopes. Therefore, the no-message correction
still receives AET predictions containing cross-zone information. Its stronger performance means
the *extra graph message path* did not help this correction recipe; it does not prove that
neighboring-zone information is irrelevant. The four anonymous nodes cannot be identified as
countries or physical transmission links from this dataset.

**Other variants tried during selection**

These are January–May **selection** scores, not June–August confirmation or September results.
Nonselected variants were not all carried forward. AET on these same selection dates scores 3.77263.

| Trial | January–May MAE | Outcome |
| --- | ---: | --- |
| Original-sized graph with explicit AET cross-zone features | 4.19980 | Not selected |
| Original-sized graph with those features, coefficient L1 and 0.8 shrinkage | 4.09640 | Not selected |
| Compact graph with local prediction path | 3.96949 | Selected among the AET-informed standalone graph trials |
| Adaptive directed graph, point prediction, no regime mixture | 3.97714 | Not selected |
| Adaptive graph with daily-regime clustering, point prediction | 4.01576 | Not selected |
| Adaptive clustered Bayesian graph | 3.96106 | Selected among graph-enabled probabilistic-study candidates |
| Matched Bayesian model without graph messages | 3.95545 | Control; slightly better than its graph counterpart |
| AET + full-strength graph error correction | 3.98431 | Strength 0.25 selected instead |
| AET + full-strength no-message error correction | 3.86173 | Strength 0.25 selected instead |

Blend/correction grids included zero, 0.1, 0.25, 0.5 and one. Zero was eligible, representing
AET only. All trials and strengths remain in their individual study ledgers, including poorer results.

**Why earlier messages mentioned AET around 3.95**

The first comparison used eight January–August monthly folds, **98 dates**, rather than just
June–August. On that broader development window:

| Model or baseline | January–August MAE |
| --- | ---: |
| Flat hourly repetition | 8.12762 |
| PCHIP smooth reconstruction | 5.61848 |
| Fixed slope × 2 rule | 4.83222 |
| Original temporal graph | 4.26107 |
| AET | **3.94790** |

Thus AET 3.94790, 4.22464 and 5.04427 refer to different evaluation dates. They are not conflicting
results, training scores, or interchangeable benchmarks. The current dashboard's 1–30 September
selection can also cover different dates from the September 19–30 assessment above.

**Less training and longer validation: the overfitting check**

| Training dates | Validation dates | Original graph | Compact graph | AET | Fixed 25% graph blend |
| ---: | --- | ---: | ---: | ---: | ---: |
| 91 | All 242 valid January–August dates | 5.11694 | 4.32663 | 3.99074 | **3.97258** |
| 180 | All 153 valid April–August dates | 4.51466 | 4.28421 | 4.02628 | **4.01242** |
| 271 | All 62 July–August dates | 4.38733 | 4.22595 | 4.00355 | **3.99294** |

These fixed-weight diagnostics score every available day in their windows, not just days 19 onward.
The original graph fits training more closely but generalizes worse than the compact graph. This
is consistent with overfitting or unstable extrapolation, while changing seasons and model age also
matter. The blend's gain over AET repeats, but stays small: approximately **0.27%–0.46%**.
The overlapping windows are not independent replications. [Full findings](results/long_validation_check/findings.md).

**Probabilistic outputs and anomaly prediction**

| Bayesian graph diagnostic | June–August | September 19–30 |
| --- | ---: | ---: |
| Raw nominal 90% interval coverage | 79.69% | 86.63% |
| Calibrated nominal 90% interval coverage | 89.28% | 94.12% |
| Mean calibrated interval width, EUR/MWh | 18.48 | 30.28 |

Calibration improves measured coverage, but September bands are wider and conservative on average.
These are marginal quarter-price bands, not a simultaneous whole-day guarantee. Raw negative-price
probabilities are not separately calibrated. Bayesian inference covers output weights and graph offsets,
with a deterministic encoder; it is partial approximate Bayesian inference. It does not apply to the
new deterministic AET error-correction models.

The existing dashboard has a **heuristic risk indicator** based on ramps, daily volatility and recent
residual magnitude. A trained, calibrated predictor of high AET error or extreme quarter-hour movement
has **not** been implemented yet. Detecting unusual supplied hourly inputs, predicting hidden quarter
spikes, and predicting large AET errors are different tasks. Some intrahour anomalies cannot be
uniquely recovered from the hourly mean alone.

**Possible improvements: what is done and what could come next**

The first priority is a predeclared rolling evaluation protocol for future experiments. Within
that protocol, the most useful new direction to test is a **price-only predictor of high AET error**,
followed by targeted ramp/boundary features and a simpler correction baseline. These would test
where extra modeling helps before adding another larger graph network.

| Improvement type | What was already tested or implemented | Possible next work and evidence required |
| --- | --- | --- |
| Error-focused feature engineering | Curve geometry, calendar/history, AET cross-zone features and predicted shape; errors examined by hour/quarter/zone | Study ramp magnitude, turning points and Q1/Q4 boundary errors; test a small number of price-only interactions on inner rolling folds. Verify gains on later dates. |
| AET residual correction | Chronological OOF labels, graph and no-message networks, magnitude regularization and frozen strength | Compare a simpler linear correction; examine recency weighting, teacher-age shifts and a few fixed seeds. Retain the chronological teacher/outer-fold separation. |
| Anomaly/error-risk prediction | Existing heuristic warnings and separate Bayesian uncertainty; no learned anomaly classifier | Predict whether hourly AET error exceeds a zone-specific threshold fitted on earlier OOF errors. Validate calibration, precision/recall and error captured in flagged hours. Start with warnings; test any specialist correction separately. |
| Graph design | Temporal attention, adaptive directed edges, two-hop messages, regime/zone clustering and masked-edge diagnostics | Test sparse or more strongly regularized messages only if they improve a matched no-message control. Keep attention separate from error-based usefulness; no country/trade-flow claims. |
| Regime-specific modeling | Soft daily-regime clusters in the Bayesian graph | Test a small specialist for difficult ramps or volatile curves, with a gate based only on known inputs. Check sample sizes and stability; clustering did not establish a large MAE gain so far. |
| Ensembling and shrinkage | Fixed 25% compact-graph blend, 10% Bayesian blend and 0.25 error corrections | Investigate a small, regularized input-dependent correction strength using nested chronological training. Compare with the fixed blend and disclose every trial; do not choose weights on September. |
| Uncertainty calibration | Student-t scenarios and frozen training-window calibration | Benchmark against simple historical-residual intervals. Evaluate coverage together with width and proper scores; then examine zone/regime calibration and negative-price probability reliability. |
| Validation and stability | Rolling development folds, frozen September assessments, longer fixed-weight windows, paired-date diagnostics and software tests | Use predeclared outer rolling windows and several seeds. Report paired error differences and temporal uncertainty, count overlapping samples correctly, and keep September out of further selection. |
| Interpretation/dashboard | AET decomposition, heuristic risk explanations, graph explorer and edge-removal exports | Add clear model/period selectors, uncertainty coverage/width, reasoned anomaly warnings and actual-versus-predicted diagnostics. Dashboard improvements aid understanding but do not themselves lower MAE. |

A sensible sequence is to fix the evaluation windows, train a separate high-error-risk classifier
from the OOF labels, and assess calibration and errors captured in flagged hours. Then test a
simple specialist/correction against the fixed blend, across several seeds and dates. Existing
residual corrections should remain references rather than assumed improvements after their
September result. Extra graph complexity needs evidence beyond the current no-message control.
No particular future change is guaranteed to achieve MAE below 5.

Every future point model should keep the zero-sum correction or equivalent hourly projection,
validate the serialized output, preserve legitimate negative prices, fit all transforms on earlier
training data, and use only challenge-permitted price information. External weather, load, generation
or identified trade-flow data are outside the current price-only challenge scope.

**Saved work and current status**

All Paola experiments and separate research artifacts are in `Paola_models/` on branch **Paola**.
The earlier `.paola-optimization/` checkout is no longer the active location. These summary edits
do not change AET weights, the dashboard, model recipes or any saved assessment results.
The latest model-code verification recorded **82 passing tests**, including chronology, target/future
poisoning, hourly means, artifact round trips and integrated CLI inference; standalone correction
retraining reproduced the saved model. This document reports that existing verification.

Useful entry points: [README and results index](README.md), [compact graph/blend](OPTIMIZATION.md),
[Bayesian graph](PROBABILISTIC_GRAPH.md), [AET residual integration](AET_RESIDUAL_GRAPH.md),
and [interactive graph explorer](results/adaptive_probabilistic_study/graph_explorer.html).
