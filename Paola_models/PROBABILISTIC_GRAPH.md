# Adaptive probabilistic temporal graph

This experiment asks whether explicit learned zone-to-zone messages improve reconstruction of
quarter-hour prices from the supplied hourly curves, and which connections the model relies on.
The current AET model already includes other zones' hourly spreads and slopes as LightGBM features.
It does not use a graph neural network, known trading volumes, or a physical transmission graph.
Its existing methodology records a 1.43% deterioration when cross-zone features were removed from
the earlier unscaled model; this is historical evidence for peer features, not a new GNN result.
See [AET features](../AET-HACKATHON/aet/features.py) and
[the recorded ablations](../AET-HACKATHON/docs/METHODOLOGY.md).

This challenge supplies the hourly price of every target zone. Neighbor information can therefore
improve the **within-hour quarter-price shape**, while each zone's hourly level is fixed by the input.
The experiment does not forecast how a shock changes another country's hourly market-clearing price.

The official nodes are **ZONE_0–ZONE_3**. Their country identities and actual interconnections are
not supplied. We can investigate statistical dependencies between those zones; identifying countries,
border flows, or causal effects would require additional identified data and a different design.

## Architecture and research choices

| Component | Implementation and purpose |
| --- | --- |
| Temporal encoder | Two gated, residual temporal convolutions with dilations 1 and 2; processes the supplied daily hourly curve. |
| Directed adaptive graph | Query-dependent additive attention, training-derived similarity prior, and learned regime-specific directed offsets. Each receiving row normalizes over source zones. |
| Multi-hop messages | A mixture of one-hop and two-hop aggregation allows indirect dependencies. A learned gate limits the message correction. |
| Daily regimes | Training-only two-component diagonal Gaussian mixture, fitted with NumPy EM to daily hourly descriptors. Soft probabilities gate coefficient experts and graph priors. |
| Zone clusters | Up to two hierarchical clusters from training-only residual-tilt correlations. These inform priors and visualization colors. |
| Local prediction path | Direct local features alongside the graph embedding, so reconstruction need not depend entirely on neighbors. |
| Constraint | Three zero-sum shape coefficients added to cumulative PCHIP; final float64 projection preserves hourly means. Negative prices remain valid. |
| Point models | Price L1 objective, with and without regime clustering. |
| Probabilistic model | Gaussian mean-field variational output weights and graph offsets; Student-t latent-coefficient likelihood with df=4, KL penalty, and auxiliary price L1 loss. |
| Intervals | Last 14 calendar days within the training cutoff calibrate absolute point error divided by predictive standard deviation, separately by zone. |

The architecture draws on [Graph WaveNet](https://arxiv.org/abs/1906.00121),
[MTGNN](https://arxiv.org/abs/2005.11650), and [GATv2](https://arxiv.org/abs/2105.14491).
Weight uncertainty follows the reparameterized variational approach in
[Bayes by Backprop](https://arxiv.org/abs/1505.05424). These are adaptations to four small anonymous
nodes, not reproductions of those papers or a claim of state-of-the-art challenge performance.
Recent approaches such as [UQGNN](https://arxiv.org/abs/2508.08551) motivate evaluating uncertainty,
but their architectures and benchmark claims cannot be transferred to this dataset without testing.

This is **partial approximate Bayesian inference**: the encoder is deterministic, the posterior is
factorized, and the auxiliary MAE term modifies the pure likelihood objective. Clustering uses EM,
not Bayesian inference. We do not claim to have integrated all model uncertainty or identified a
true posterior over physical relationships. A point estimate uses posterior-mean weights; it is not
an exact posterior predictive median. Distributional modeling can improve uncertainty without improving MAE.
The Student-t likelihood factorizes over latent coefficients. Shared random parameters and the
mean-preserving basis induce some dependence, but this is not a full learned multivariate noise
covariance model across all zones and hours.

## Evaluation and information boundaries

`advanced_study.py` records a bounded plan before training: four fixed candidates on January–May
monthly folds, day 19 to month-end. The lowest-MAE graph candidate and one of five AET blend weights
are selected there. The no-message Bayesian model is a selection-period control. The selected point
model and the probabilistic model then run on June–August folds without further tuning. Earlier
research already inspected these dates, so this is development confirmation, not an independent test.
September and October are excluded from this new study.

A subsequent explicitly requested **frozen September 19–30 assessment** is reported separately:
[results and uncertainty diagnostics](results/adaptive_probabilistic_september/assessment_results.md).
It did not retrain, recalibrate or change the development-selected recipe or blend weight.

Every fold trains through the preceding month-end. For Bayesian models, the last 14 calendar days
of that window fit interval calibration only. Weights, feature scaling, daily clusters, zone clusters,
and graph priors use the earlier proper-training period. Both cutoffs are exported. This reduces
the Bayesian model's weight-training data compared with the point candidates. The no-message
control uses the same Bayesian calibration split; because its regime gate summarizes all zones,
it removes graph messages but does not eliminate every cross-zone pathway.

History features always read earlier dates. Earlier scored actuals may enter later-day history,
consistent with the existing daily challenge replay. Inference never fits weights, scalers, clusters,
graph priors or calibration. All scenarios preserve each supplied hourly mean, including negative hours.
Uncertainty bands and risk probabilities stay separate from the official point-prediction CSV.

## Reading uncertainty and graph diagnostics

- Compare **MAE** on identical dates and zones. All four zones carry equal weight.
- Compare raw predictive and calibrated **90% coverage** with width. Wider intervals alone are not
  an improvement. CRPS and interval score penalize spread and misses; lower is better.
- Calibration is empirical and marginal. Quarter errors are temporally dependent; the finite-sample
  quantile adjustment does not establish guaranteed coverage under drift. Whole-zone-day coverage
  is reported separately. See [time-series conformal research](https://proceedings.mlr.press/v139/xu21h.html)
  for why temporal structure requires additional care; this implementation does not reproduce EnbPI.
- Epistemic standard deviation covers conditional location variation from sampled output weights
  and graph offsets. Predictive standard deviation also includes Student-t coefficient noise.
- Raw probability of a negative price is a model frequency, **not separately probability-calibrated**.
- Attention indicates message allocation. It is not an effect size and need not track accuracy gains.
- **Edge removal delta** equals receiving-zone MAE after masking an edge minus original MAE.
  Positive means the connection helped this fitted model; negative means removal helped. Attention is
  renormalized and alternate two-hop paths remain. This is not retraining or a causal intervention.
- **Neighbor stress** shifts a source zone's full hourly curve by +10 EUR/MWh and measures changes
  in another zone's predicted quarter-hour shape. This includes changes in regime probabilities.
  It is a hypothetical input sensitivity, not a realistic estimate of price transmission.
- `final_edge_posterior.csv` contains uncertainty on additive **logit offsets**. Those standard
  deviations are not confidence intervals on final normalized attention or physical flows.

## Run and explore

From ApplePi, using the existing environment:

```bash
# Fixed study; choose a fresh output folder for any later run.
.venv-mae/bin/python -m Paola_models.advanced_study

# Build a fully offline HTML explorer and static graph diagnostics.
.venv-mae/bin/python -m Paola_models.graph_explorer \
  --results Paola_models/results/adaptive_probabilistic_study
```

Open `results/adaptive_probabilistic_study/graph_explorer.html` in a browser. Choose an attention,
edge-removal or sensitivity view, then a date/hour. The price chart compares predictions with actuals
and calibrated intervals. All diagnostics in that explorer belong to the **Bayesian graph** even if
another candidate has better MAE. `advanced_results.md` holds the candidate comparison.

The new study saves its own artifacts through August. The Bayesian artifact's weights end August 17,
with August 18–31 reserved for calibration. It is an August research snapshot, not a final full-year
submission artifact. Existing AET and earlier graph artifacts are hash-checked and remain unchanged.

```bash
# Required point output, using either new artifact kind.
.venv-mae/bin/python -m Paola_models.cli infer \
  --model Paola_models/results/adaptive_probabilistic_study/artifact_clustered_bayesian \
  --hourly path/to/new_hourly.csv --out path/to/new_point_predictions.csv

# Bayesian diagnostics and coherent scenarios; --out is a new directory.
.venv-mae/bin/python -m Paola_models.probabilistic_cli infer \
  --model Paola_models/results/adaptive_probabilistic_study/artifact_clustered_bayesian \
  --hourly path/to/new_hourly.csv --out path/to/new_diagnostics
```

Optional `--history path/to/history_15min.csv` supplies earlier quarter prices. Four complete
24-hour zone curves are required. Targets must follow the artifact's data/calibration cutoff.
The diagnostic directory contains `point_predictions.csv`, `uncertainty.csv`, `graph.csv`, and
`scenarios.npz`. Scenario dimensions are date × draw × zone × hour × quarter. Raw scenarios preserve
hourly means; calibrated marginal interval endpoints need not describe one jointly valid scenario.

Standalone training is available with `python -m Paola_models.probabilistic_cli train --out NEW_DIR`.
It uses the fixed Bayesian recipe, not a new optimization loop. In addition to `requirements.txt`,
building the explorer needs the local Plotly and Matplotlib packages listed in
`requirements-probabilistic.txt`. Inference needs no network or explorer packages.

## Practical next decision

Keep the strongest measured point model as the accuracy reference. Promote a new graph only if
repeatable accuracy gains justify its complexity; useful uncertainty and visualization are separate
benefits. Check which regime, zone and hour contribute errors before expanding the architecture.
A future locked period or predeclared outer rolling evaluation is needed for stronger generalization
evidence. Country-level trade claims require identified countries, flows and an appropriate causal design.

For a business-facing dashboard, distinguish reconstruction accuracy from market risk. Wider bands
can flag difficult intrahour timing; edge removal can identify which peer inputs are useful; regime
labels can separate recurring curve patterns. These are diagnostics to validate, not trading returns
or identified congestion states. Evaluate negative-price event reliability and whole-day scenario
dependence before treating the exported raw probabilities as operational risk estimates.
