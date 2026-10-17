# Idea cards

Format and status values: `RESEARCH.md`. R-001 to R-004 are seeds written from the error analysis
in `docs/METHODOLOGY.md`. The research loop adds cards from the literature.

### R-001 Sawtooth-aware Denton baseline
- Status: proposed
- Source: P-001 (Denton-type disaggregation); EDA: the largest price step falls at the hour boundary,
  and q1/q4 carry about 60 % of the error.
- Hypothesis: a mean-preserving smoother that penalises 15-minute changes inside an hour more
  than changes across hour boundaries reproduces the ramp-and-jump shape better than PCHIP. That
  would give the residual learner a better starting point.
- Mechanism: per zone-day, minimise `Σ_t w_t (x_t − x_{t−1})²` subject to the 24 hourly means,
  with `w_t = 1` inside an hour and `ρ < 1` across a boundary. The KKT system (96 + 24 unknowns) has
  a closed-form solve. `ρ` is chosen once on the training slice (inner time split) and frozen.
- Plan: `baselines.py` gains a `denton` baseline with `ρ`; `pipeline.py` fits and stores `ρ`;
  a test for mean preservation. About 60 lines.
- Cost / risk: low cost. PCHIP already captures most of the ramp, and the spline baseline lost
  once the learner was added (E5).
- Priority: H

### R-002 Monotone constraint on the slope coefficient
- Status: proposed
- Source: P-006; shape atlas (the slope coefficient follows the sign of the local hourly slope).
- Hypothesis: forcing the slope-coefficient model to be non-decreasing in the zone's own −1/+1 h
  slopes regularises it on volatile days.
- Mechanism: LightGBM `monotone_constraints` on the slope booster only.
- Plan: per-coefficient parameters in `models.py` and a config key. About 25 lines.
- Cost / risk: very cheap; the constraint may be too strict where the cross-zone context matters.
- Priority: M

### R-003 Cross-hour smoothing of predicted coefficients
- Status: proposed
- Source: E6 graph shrinkage across zones (+0.15 %); adjacent hours share a regime.
- Hypothesis: per-hour tree predictions are noisy. A small fixed kernel across neighbouring hours
  (`[a, 1−2a, a]` on each coefficient) cuts variance without moving any hourly mean.
- Mechanism: post-processing in coefficient space (shapes stay zero-sum). `a` is chosen on the
  training slice (inner time split) and frozen.
- Plan: `pipeline.py` post-processor plus a config key. About 30 lines.
- Cost / risk: cheap; it may blur genuine hour-to-hour regime switches (sunrise, sunset).
- Priority: M

### R-004 Direct models for the edge quarters
- Status: proposed
- Source: error analysis (q1/q4 about 60 % of the error); coefficient L1 is only a surrogate for price L1.
- Hypothesis: two L1 models that predict the q1 and q4 deviations directly target where the error
  lives. q2 and q3 then follow from the curvature model and the zero-sum constraint.
- Mechanism: predict d1 and d4 (L1, EUR/MWh) and the curvature coefficient. Solve q2 and q3 from
  `Σ d = 0` and the curvature, so the shape stays zero-sum by construction.
- Plan: a new `ResidualModel` in `models.py` and a test. About 60 lines.
- Cost / risk: medium; the pooled quarter model (E2b) already lost, though it shared one model
  across all four quarters.
- Priority: M
