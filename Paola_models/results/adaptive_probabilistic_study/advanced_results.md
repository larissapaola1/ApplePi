# Adaptive probabilistic graph: bounded research results

## Selection and confirmation MAE, EUR/MWh

| stage | method | MAE |
| --- | --- | --- |
| selection_Jan_May | AET | 3.7726 |
| selection_Jan_May | Previous optimized graph | 3.9695 |
| selection_Jan_May | Previous fixed blend | 3.7573 |
| selection_Jan_May | adaptive_point | 3.9771 |
| selection_Jan_May | clustered_point | 4.0158 |
| selection_Jan_May | clustered_bayesian | 3.9611 |
| selection_Jan_May | no_message_bayesian | 3.9555 |
| selection_Jan_May | New selected blend | 3.7681 |
| confirmation_Jun_Aug | AET | 4.2246 |
| confirmation_Jun_Aug | Previous optimized graph | 4.3440 |
| confirmation_Jun_Aug | Previous fixed blend | 4.2057 |
| confirmation_Jun_Aug | clustered_bayesian | 4.3359 |
| confirmation_Jun_Aug | New selected blend | 4.2195 |

Selected graph: **clustered_bayesian**. Selected graph blend weight: **0.1**. These choices were
written before June–August evaluation. All models use the same zone/date scoring keys.
The no-message Bayesian variant is a control, not eligible for graph-model selection.
The former graph and former AET blend are reported alongside all new trials.

## Probabilistic performance on June–August

| market_id | raw90_coverage | calibrated90_coverage | mean_interval_width | interval_score90 | CRPS | whole_zone_day_inside |
| --- | --- | --- | --- | --- | --- | --- |
| ZONE_0 | 0.8303 | 0.9090 | 14.3678 | 19.8169 | 2.4691 | 0.0000 |
| ZONE_1 | 0.7607 | 0.8769 | 20.2955 | 33.0260 | 3.8618 | 0.0263 |
| ZONE_2 | 0.7615 | 0.8824 | 18.4497 | 26.4324 | 3.2171 | 0.0000 |
| ZONE_3 | 0.8353 | 0.9030 | 20.8263 | 31.8829 | 3.5521 | 0.0000 |

Raw Bayesian predictive intervals and calibrated intervals are reported separately. Nominal
marginal coverage is 90%; compare measured coverage with interval width, interval score and CRPS.
Bayesian models reserve the last 14 calendar days within each training cutoff for interval calibration;
those dates do not fit network weights, scalers, clusters or graph priors. Thus their effective weight
training ends earlier than the point models. Exact dates are in `training_and_calibration_cutoffs.csv`.
Calibration is frozen before scoring and at inference. These intervals have no guaranteed coverage
under arbitrary temporal distribution shift. They are not simultaneous whole-curve confidence sets.

## Which neighbor connections matter to this model?

| source | target | MAE_increase_when_edge_removed | mean_prediction_change |
| --- | --- | --- | --- |
| ZONE_1 | ZONE_0 | 0.0109 | 0.1767 |
| ZONE_2 | ZONE_0 | 0.0054 | 0.0920 |
| ZONE_3 | ZONE_2 | 0.0051 | 0.1851 |
| ZONE_1 | ZONE_3 | 0.0041 | 0.1204 |
| ZONE_0 | ZONE_3 | 0.0041 | 0.1822 |
| ZONE_1 | ZONE_2 | 0.0036 | 0.2387 |
| ZONE_3 | ZONE_1 | -0.0029 | 0.0800 |
| ZONE_2 | ZONE_3 | -0.0037 | 0.0807 |
| ZONE_2 | ZONE_1 | -0.0047 | 0.0751 |
| ZONE_0 | ZONE_2 | -0.0063 | 0.1109 |
| ZONE_0 | ZONE_1 | -0.0065 | 0.3231 |
| ZONE_3 | ZONE_0 | -0.0086 | 0.1075 |

Positive removal delta means removing that source→target graph edge worsened held-out MAE.
Negative means removal helped. This is model-reliance evidence, not causation or measured trade flow.
Attention weights alone are not evidence of influence. The +10 EUR/MWh source-curve perturbations
are hypothetical input sensitivities, not interventions on a physical power system.
Nodes remain ZONE_0–ZONE_3: the official package does not identify countries or their real borders.

## What is implemented

GATv2-style query-dependent attention, learned directed regime-specific edge offsets, two-hop messages,
gated dilated temporal convolutions, direct local coefficient prediction, and a zero-sum output basis.
Training-only two-component diagonal Gaussian EM clusters daily price regimes. A separate training-only
hierarchical grouping of historical residual-tilt similarity supplies zone clusters and graph priors.
No external geographic adjacency or price data is used. All weights start from scratch.

The probabilistic model uses mean-field Gaussian variational distributions on output weights and
graph offsets, Student-t latent-coefficient likelihood (df=4), KL regularization, and an auxiliary MAE
term. This is partial, approximate Bayesian inference, not a fully Bayesian encoder or a posterior
guarantee. Every sampled four-quarter scenario preserves its supplied hourly mean; marginal interval
endpoints need not themselves form a valid scenario. Probabilistic outputs are separate from the
required point-prediction CSV.

The methods are research-inspired, not a claim of state-of-the-art performance on this challenge.
There are only four anonymous nodes and one training year. Complexity must earn its place in the
comparison above. No settings were retuned after confirmation; September and October were excluded.
All development dates were previously inspected, so this is not an independent final test.

## Outputs

`hourly_graph.csv`, `zone_clusters_by_fold.csv`, `edge_ablation.csv`, `neighbor_sensitivity.csv`,
`probabilistic_predictions.csv`, `distribution_scores.csv`, and `final_edge_posterior.csv` are
available for visualization and audits. Original artifacts remain unchanged. New isolated artifacts
are `artifact_clustered_bayesian/` and `artifact_clustered_bayesian/` (same folder if the Bayesian candidate wins).
The latter's manifest explicitly separates weight-training end from the data/calibration cutoff.
