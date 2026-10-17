# Reading log

Format and rules: `RESEARCH.md`. The seed entries come from the reference list of
`docs/source/weekend_roadmap.pdf`. They were not re-read for this log, so open each one and update
its `Read:` field before relying on it.

### P-001 Sax, C. and Steiner, P. (2013). Temporal Disaggregation of Time Series. The R Journal.
- URL: https://journal.r-project.org/articles/RJ-2013-028/        Read: not yet (roadmap summary)
- Claim (per roadmap): aggregate-preserving temporal disaggregation; roles of Denton and
  regression-based (Chow–Lin type) methods.
- Mechanism for us: mean-preserving reconstruction with smoothness or indicator series; basis of
  the cumulative-interpolation baseline (E1).
- Compliance: OK (method only)
- Ideas: E1 (done), R-001

### P-002 Troncoso, A. et al. (2007). Electricity Market Price Forecasting Based on Weighted Nearest Neighbors Techniques. IEEE Trans. Power Systems.
- URL: https://idus.us.es/items/ac140d42-8745-4345-8dfa-0cdf52ec0f0a        Read: not yet (roadmap summary)
- Claim (per roadmap): weighted nearest neighbours for electricity price forecasting.
- Mechanism for us: curve-conditioned analogue retrieval of residual shapes.
- Compliance: OK
- Ideas: E4 (done, +0.25 %, not promoted)

### P-003 Li, Y. et al. (2025). Multi-task Graph Adaptive Learning for Multivariate Electricity Price Short-Term Forecasting in Australia's NEM. IEEE Trans. Power Systems.
- URL: https://scholars.cityu.edu.hk/en/publications/multi-task-graph-adaptive-learning-for-multivariate-electricity-p/        Read: not yet (roadmap summary)
- Claim (per roadmap): graph models exploit inter-market dependence (different task).
- Mechanism for us: cross-zone smoothing of shape coefficients.
- Compliance: OK
- Ideas: E6 graph shrinkage (done, +0.15 %, not promoted)

### P-004 Ho-Nguyen, N., Alipour, H., Panagiotelis, A. and Athanasopoulos, G. (2026). Optimal Forecast Reconciliation for Quantiles. Monash EBS working paper 04/26.
- URL: https://www.monash.edu/business/ebs/research/publications/ebs/2026/wp04-2026.pdf        Read: not yet (roadmap summary)
- Claim (per roadmap): reconciliation should match the loss; coherent distributions can yield
  point quantiles that violate aggregation constraints.
- Mechanism for us: constrained L1 decision from conditional quantiles.
- Compliance: OK
- Ideas: E6 common-τ quantiles (done, −2 to −3 %, not promoted)

### P-005 SciPy. PchipInterpolator documentation.
- URL: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html        Read: not yet
- Mechanism for us: shape-preserving interpolation of the cumulative curve (current baseline).
- Compliance: OK

### P-006 LightGBM. Parameters documentation.
- URL: https://lightgbm.readthedocs.io/en/stable/Parameters.html        Read: not yet
- Mechanism for us: `regression_l1`, `quantile`/`alpha`, `monotone_constraints`, regularisation.
- Compliance: OK
- Ideas: R-002
