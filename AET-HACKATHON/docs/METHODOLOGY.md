# Methodology note

Every number below comes from the public replica described in section 2 and from the runs logged
in `results/ledger.jsonl` (not versioned; regenerate with the commands in section 10). Claims are
limited to what the backtests show.

**Summary.** Final method: PCHIP cumulative baseline + 0.8 × a zero-sum residual predicted by three
L1 LightGBM regressors on slope / curvature / zigzag coefficients, with same-day features from all
zones and lagged residual shapes from earlier dates. Development (8 monthly folds) 3.943 EUR/MWh,
−51.5 % vs flat; locked September 5.004, −54.1 %; real days 1–17 Oct 2026 (after the freeze)
4.939, −51.5 %, better than flat on every day.

## 1. Problem and constraint
Four quarter-hour prices per hour, average equal to the supplied hourly price (±0.0001 EUR/MWh).
Score: daily MAE, averaged per zone, then equally across zones.

## 2. Data boundaries
- Training, validation and selection: official file only, 1 Oct 2025 – 30 Sep 2026, DST days excluded.
- 1–18 Oct 2026 never consulted during development; the code refuses training rows after 30 Sep 2026.
- Inference: the day's hourly curves (all zones) and quarter-hour history strictly before that day.
- No external data, pretrained weights, embeddings or holiday calendars.

**Public replica used here.** The official package is not available, so every number in this note
comes from a replica built with `tools/fetch_energy_charts.py` (Energy-Charts `/price`, CC BY 4.0
from Bundesnetzagentur | SMARD.de). Zones: ZONE_0 = DE-LU, ZONE_1 = FR, ZONE_2 = AT,
ZONE_3 = IT-North. Results are not comparable with the hackathon leaderboard.
- 363 training dates per zone (1 Oct 2025 – 30 Sep 2026, DST days 26 Oct and 29 Mar excluded),
  139 392 rows, as in the brief. Prices from −499.99 to 747.10 EUR/MWh; 3.75 % of quarters negative.
- Timestamp convention (interval start) checked on the training window: mean |step| into
  q1/q2/q3/q4 = 10.76 / 8.31 / 7.55 / 8.48 EUR/MWh; the step across the hour boundary is the
  largest in every zone (DE-LU 9.73, FR 9.97, AT 14.48, IT-North 8.88).

## 3. Representation
- Baseline: cumulative-curve interpolation. With `C[0] = 0`, `C[h+1] = C[h] + H[h]` and any
  interpolator `F` through the knots, quarter `q` of hour `h` is `4 (F(h+(q+1)/4) − F(h+q/4))`; the
  four values telescope to `4 H[h]`, so the hourly mean is preserved algebraically.
  Parameter-free baselines on the full training year (`aet.eda`, official metric, EUR/MWh):

  | Baseline | ZONE_0 | ZONE_1 | ZONE_2 | ZONE_3 | Official | vs flat |
  |---|---|---|---|---|---|---|
  | flat | 7.915 | 7.379 | 9.426 | 5.590 | 7.578 | — |
  | linear ramp | 5.170 | 5.486 | 6.913 | 4.277 | 5.461 | −27.9 % |
  | PCHIP | 5.031 | 5.403 | 6.773 | 4.303 | 5.377 | −29.0 % |
  | Akima | 5.034 | 5.415 | 6.792 | 4.361 | 5.401 | −28.7 % |
  | natural spline | 4.941 | 5.363 | 6.752 | 4.306 | 5.340 | −29.5 % |

  Most of the recoverable shape is a ramp that follows the hourly curve: in falling hours the mean
  deviation from the hourly price is about +12 / +3 / −3 / −11 EUR/MWh over q1..q4 in ZONE_0 and the
  mirror image in rising hours (`results/eda/shape_atlas.csv`).
- Residual: zero-sum orthonormal basis (slope, curvature, zigzag). Share of residual energy:

  | Reference | ZONE_0 | ZONE_1 | ZONE_2 | ZONE_3 |
  |---|---|---|---|---|
  | over flat: slope / curv. / zigzag | 94 / 5 / 2 % | 86 / 7 / 7 % | 91 / 6 / 3 % | 86 / 9 / 5 % |
  | over PCHIP: slope / curv. / zigzag | 84 / 13 / 4 % | 72 / 16 / 12 % | 83 / 12 / 5 % | 75 / 16 / 9 % |

  After PCHIP the slope is still the dominant component, so the learner mainly corrects ramp
  amplitude; curvature and zigzag carry a fifth to a quarter of the energy.

**Drift.** PCHIP residual MAE per unit of mean |hourly slope| by month (`results/eda/drift.csv`):

| Month | ZONE_0 | ZONE_1 | ZONE_2 | ZONE_3 |
|---|---|---|---|---|
| 2025-10 | 0.558 | 0.599 | 0.727 | 0.550 |
| 2025-12 | 0.594 | 0.629 | 0.738 | 0.508 |
| 2026-02 | 0.448 | 0.609 | 0.592 | 0.464 |
| 2026-04 | 0.333 | 0.486 | 0.451 | 0.385 |
| 2026-06 | 0.322 | 0.506 | 0.409 | 0.449 |
| 2026-08 | 0.297 | 0.388 | 0.507 | 0.555 |
| 2026-09 | 0.262 | 0.296 | 0.419 | 0.505 |

In ZONE_0–2 the shape left after PCHIP per unit of slope halves over the year, and PCHIP's gain over
flat grows (ZONE_0: 29 % in Oct 2025, 45 % in Sep 2026). ZONE_3 shows no trend. One year cannot
separate market adaptation from season (summer solar ramps are steeper and smoother), but either
way October 2025 looks atypical, which motivates the recency ablation and the history features.

## 4. Features
One row per (date, zone, hour); `aet/features.py` builds them identically for training,
backtesting and inference.

| Group | Features | Why |
|---|---|---|
| Calendar | zone (categorical), hour, weekday, weekend, season sin/cos | shapes differ by zone and time of day; solar ramps change with season |
| Own curve | level, slopes at −2/−1/+1/+2 h, curvature | the intra-hour ramp follows the local hourly slope (section 3) |
| Day | mean, range, std, negative-hour count, level relative to the day mean | regime of the day (scarcity, solar surplus) |
| Cross-zone | for every zone: spread to this zone, its −1/+1 slopes | the coupled zones share ramps; removing them costs 1.7 % (7.2) |
| Baseline | the PCHIP shape's own three coefficients | lets the model correct the interpolator rather than relearn it |
| History | residual coefficients 1 and 7 days before, 7- and 28-day means with counts | recent shape amplitude; worth 2.8 % (E3 vs E2) |

Causality: same-day inputs are only the 24 hourly prices of each zone, which the jury supplies.
Everything else comes from `HistoryStore`, a calendar-indexed store that only reads slots strictly
before the target date (lags in calendar days, so the day after a DST-excluded date sees a missing
lag, not the wrong day). In training each row also sees only earlier dates. Hours 23–24 never see
the next day. In the final configuration hours 1–2 do not use the previous day either
(`prev_day_edge: false`, section 7.3). Sentinel tests poison every future row and check that
features and predictions do not change.

## 5. Model
- `P(h, q) = PCHIP(h, q) + 0.8 · Σ_j c_j(x) b_j(q)` with the orthonormal zero-sum basis
  `b = slope [−3,−1,1,3]/√20, curvature [1,−1,−1,1]/2, zigzag [−1,3,−3,1]/√20`.
- One LightGBM regressor per coefficient, objective `regression_l1`, 500 trees, learning rate
  0.05, 15 leaves, 100 rows per leaf, seed 42, `deterministic` + `force_row_wise`.
- Why coefficients and not quarters: one regressor per coefficient beat the pooled price-space
  model by 2.1 % without history (E2 vs E2b) and 1.3 % with history (E3 vs E3b), and it also
  beat the five-quantile common-τ variant (7.4).
- Why 0.8: shrinking the learned amplitude passed the gate (+0.87 %, 7.2), consistent with
  the drift; 0.7 and 0.9 do not improve on it (7.5).
- Training on all 363 dates: 4.8 s on a 20-thread desktop CPU; inference 0.13 s per day including
  file I/O and validation.

## 6. Validation
- Rolling origin with history replay: each fold model is trained up to a month end and frozen,
  then scores days 19..end of the next month, receiving for every scored date only that date's
  hourly curves and all quarter-hours strictly before it (the 18-day gap mimics 30 Sep → 19 Oct).
- Eight monthly development folds (Jan–Aug 2026) for every decision; the roadmap's three folds
  (Apr, Jun, Aug) are a subset and are reported for the final model only.
- Promotion gate (predeclared): ≥0.5 % lower score than the incumbent, better on ≥2/3 of folds
  (6/8), no zone more than 2 % worse; one change at a time; close results keep the simpler method.
- Locked block: train to 31 Aug, score 19–30 Sep, opened once after freezing (7.6).

Per-fold scores (development, EUR/MWh):

| Fold | flat | PCHIP | E3 (handoff default) | Final (I2) |
|---|---|---|---|---|
| Jan | 6.397 | 4.841 | 3.497 | 3.548 |
| Feb | 5.506 | 4.024 | 3.047 | 3.010 |
| Mar | 8.767 | 6.308 | 4.454 | 4.306 |
| Apr | 8.817 | 5.977 | 4.370 | 4.332 |
| May | 8.490 | 5.509 | 3.575 | 3.509 |
| Jun | 10.142 | 6.685 | 4.268 | 4.347 |
| Jul | 8.246 | 5.601 | 4.204 | 4.072 |
| Aug | 8.308 | 5.796 | 4.327 | 4.295 |
| **Pooled** | **8.128** | **5.619** | **3.984** | **3.943** |

Final vs E3, paired: better on 6/8 folds, 58 % of zone-days and 64 % of dates; the largest single-date
loss is +0.72 (26 Jun), the largest gain −0.60 (21 Mar), so no single day drives the result. On the
roadmap's three folds alone the two are tied (4.324 vs 4.322): the evidence for I2 over E3 comes
from the eight folds and is modest (1.0 %).

## 7. Results
### 7.1 Ladder E0–E3 (8 monthly development folds, Jan–Aug 2026, days 19..end scored)
`results/core_*` / `configs/core.yaml --folds monthly`. Official score in EUR/MWh; the gate column
compares each rung with the previous one (≥0.5 % better, ≥6/8 folds, no zone >2 % worse).

| Candidate | Score | vs flat | vs previous rung | Folds | Worst zone | Day win rate | Gate |
|---|---|---|---|---|---|---|---|
| E0 flat | 8.128 | — | — | — | — | — | ref |
| E1 linear ramp | 5.758 | −29.2 % | (vs PCHIP) −2.5 % | 0/8 | +4.3 % | 0.30 | – |
| E1 PCHIP | 5.619 | −30.9 % | vs flat −30.9 % | 8/8 | −23.1 % | 1.00 | PASS |
| E2 coef (no history) | 4.098 | −49.6 % | vs PCHIP −27.1 % | 8/8 | −13.1 % | 0.95 | PASS |
| E2b quarter (no history) | 4.184 | −48.5 % | vs E2 +2.1 % | 1/8 | +2.6 % | 0.38 | – |
| **E3 coef + history** | **3.984** | **−51.0 %** | vs E2 −2.8 % | 8/8 | −1.6 % | 0.66 | PASS |
| E3b quarter + history | 4.036 | −50.3 % | vs E3 +1.3 % | 2/8 | +1.9 % | 0.40 | – |

Per zone, E3: ZONE_0 3.19, ZONE_1 4.37, ZONE_2 4.61, ZONE_3 3.76 (flat: 8.68 / 7.83 / 10.15 / 5.85).
Per quarter, E3: 4.82 / 3.14 / 3.16 / 4.81 (flat: 11.66 / 4.54 / 4.43 / 11.88): the edge quarters
carry most of the error and most of the gain. The residual is clearly predictable from the
current-day curves (E2), and fresh history adds a further, consistent 2.8 % (E3).

### 7.2 E5 ablations, round 1 (reference E3, same folds)
`configs/ablations.yaml`. One change at a time against E3 (3.984).

| Candidate | Score | vs E3 | Folds | Worst zone | Day win rate | Gate |
|---|---|---|---|---|---|---|
| E5 Haar basis | 4.004 | −0.51 % | 3/8 | +1.4 % | 0.48 | – |
| E5 Akima baseline | 4.032 | −1.22 % | 3/8 | +2.2 % | 0.46 | – |
| E5 natural-spline baseline | 4.008 | −0.62 % | 3/8 | +1.3 % | 0.47 | – |
| **E5 residual scale 0.8** | **3.949** | **+0.87 %** | 6/8 | +0.4 % | 0.55 | **PASS** |
| E5 recency half-life 90 d | 3.979 | +0.12 % | 3/8 | +0.3 % | 0.53 | – |
| E5 no previous-day edge | 3.975 | +0.22 % | 7/8 | +0.1 % | 0.55 | – |
| E5 no cross-zone features | 4.050 | −1.67 % | 1/8 | +2.7 % | 0.35 | – |
| E5 200 trees | 3.995 | −0.29 % | 2/8 | +0.7 % | 0.44 | – |

Promoted: residual scale 0.8 (new incumbent "E3 + s0.8"). Shrinking the learned amplitude is
consistent with the drift in section 3: the residual model learns an average amplitude over the
year, larger than the recent one. The spline baseline is the best parameter-free method on its own
(section 3) but loses once the residual model sits on top. Dropping the previous-day edge is
simpler and not worse; it is re-checked against the new incumbent below.

### 7.3 E4 analogue memory (reference E3 + s0.8 = 3.949)
`configs/memory.yaml`. For each zone and hour, the K earlier dates (same zone and hour, inside the
window) with the closest standardised curve descriptors — slopes at −2..+2, day level and range,
current-hour spreads to every zone, weekend flag; scaler fitted on the training slice and frozen —
are averaged with one Gaussian weight per neighbour (bandwidth = distance of the K-th neighbour).
The retrieved residual coefficients, nearest distance, candidate count (0 = missing) and effective
neighbour count become features. Strictly causal in training as well (only earlier dates).

| Candidate | Score | vs ref | Folds | Worst zone | Day win rate | Gate |
|---|---|---|---|---|---|---|
| E4 K=8, 180 d | 3.940 | +0.25 % | 5/8 | +0.1 % | 0.53 | – |
| E4 K=16, 180 d | 3.941 | +0.22 % | 6/8 | +0.1 % | 0.51 | – |
| E4 K=32, 180 d | 3.955 | −0.14 % | 4/8 | +0.3 % | 0.46 | – |
| E4 K=16, full history | 3.946 | +0.10 % | 6/8 | +0.3 % | 0.50 | – |
| no previous-day edge | 3.943 | +0.16 % | 5/8 | +0.0 % | 0.53 | simplification |

No memory variant reaches the 0.5 % gate, so E4 stays out of the final pipeline (the code and tests
remain as a documented ablation). The fixed 1/7-day lags and 7/28-day means already carry most of
the recent-shape information. Dropping the previous-day edge was slightly better in both rounds
(+0.22 %, 7/8 folds; +0.16 %, 5/8 folds; no zone worse by more than 0.1 %): it is adopted as a
simplification. New incumbent "I2" = E3 + residual scale 0.8 + no previous-day edge.
Example (diagnostics): for ZONE_0, hour 19 on 2026-09-15, K=16 retrieved 11 Sep, 7 Sep, 24–27 Aug,
early April and late July dates, with weights 0.05–0.09.

### 7.4 E6 specialised options (reference I2 = 3.943)
`configs/e6.yaml`. Graph shrinkage: per hour, the 4×3 matrix of total deviation coefficients is
smoothed across zones with `C_s = solve(I + λL, C)`, `A[z,k] = exp(−|H_z − H_k| / s)`, `s` = median
training spread between zones (floor 1 EUR/MWh); every zone stays zero-sum. Constrained quantiles:
five pooled LightGBM quantile models (α = 0.1…0.9) over stacked quarters, crossing repair by
sorting, then the common τ with `Σ_q Q_q(τ) = 0`, solved exactly on the piecewise-linear curves
(fallback: centred medians).

| Candidate | Score | vs I2 | Folds | Worst zone | Day win rate | Gate |
|---|---|---|---|---|---|---|
| shrinkage λ = 0.1 | 3.937 | +0.15 % | 6/8 | +0.5 % | 0.51 | – |
| shrinkage λ = 0.3 | 3.975 | −0.80 % | 0/8 | +2.3 % | 0.41 | – |
| common-τ quantiles (scale 0.8) | 4.064 | −3.07 % | 0/8 | +4.7 % | 0.26 | – |
| common-τ quantiles (scale 1.0) | 4.024 | −2.05 % | 1/8 | +4.5 % | 0.35 | – |

Neither option is promoted. The quantile decision itself behaves as derived (the summed quantile
curve failed to bracket zero in 1 of ~9 000 scored hours), but pooled per-quarter learners are
weaker than one regressor per basis coefficient here, as E2b/E3b already showed, and the
decision cannot recover that. Shrinkage helps ZONE_3 slightly and hurts ZONE_0; with λ = 0.3 it
oversmooths, as the roadmap warned.

### 7.5 Round 4: narrow ablations (reference I2 = 3.943)
`configs/round4.yaml`.

| Candidate | Score | vs I2 | Folds | Worst zone | Day win rate | Gate |
|---|---|---|---|---|---|---|
| residual scale 0.7 | 4.001 | −1.47 % | 0/8 | +2.2 % | 0.32 | – |
| residual scale 0.9 | 3.935 | +0.21 % | 5/8 | +0.6 % | 0.54 | – |
| 1000 trees | 3.932 | +0.28 % | 6/8 | −0.0 % | 0.59 | – |
| slope + curvature only (no zigzag model) | 3.973 | −0.77 % | 0/8 | +2.1 % | 0.32 | – |
| recency half-life 180 d | 3.957 | −0.35 % | 2/8 | +0.8 % | 0.46 | – |

Nothing passes. The scale optimum sits between 0.8 and 0.9 and is flat there; 0.8 stays. The
zigzag component is small (section 3) but still predictable: dropping its model costs 0.8 %.

**Stopping rule.** Two consecutive rounds (E6, round 4) promoted nothing, so the search stops and
I2 is frozen as the final configuration before the locked September block is opened.

### 7.6 Locked September block (opened once, after freezing)
Train to 31 Aug 2026, score 19–30 Sep with history replay (`configs/final.yaml --include-locked`).

| Candidate | Score | vs flat | ZONE_0 | ZONE_1 | ZONE_2 | ZONE_3 |
|---|---|---|---|---|---|---|
| flat | 10.899 | — | 10.78 | 9.79 | 13.30 | 9.72 |
| PCHIP | 6.927 | −36.4 % | 5.73 | 5.91 | 8.58 | 7.49 |
| E3 (handoff default) | 5.433 | −50.1 % | 4.46 | 4.97 | 6.76 | 5.54 |
| **Final (I2)** | **5.004** | **−54.1 %** | 3.84 | 4.49 | 6.26 | 5.43 |

The final model generalises to the unseen month: −54.1 % vs flat (−51.5 % in development) and
7.9 % better than E3, beating it on every scored date and in every zone. September prices are more
volatile (flat 10.90 vs 8.13 in development), so absolute errors are higher.

### 7.7 Real days after the freeze: 1–17 October 2026
Same frozen artifacts (`aet.train` on all 363 dates), the jury entry point `aet.infer` run once per
day with that day's `test_hourly.csv` and all quarter-hours before it, scored with `tools.score`.
These dates were never consulted before the freeze.

| | Score | flat | vs flat | ZONE_0 | ZONE_1 | ZONE_2 | ZONE_3 |
|---|---|---|---|---|---|---|---|
| 1–17 Oct 2026 | **4.939** | 10.191 | **−51.5 %** | 3.21 | 4.67 | 6.47 | 5.41 |

Daily scores range from 3.05 (3 Oct, −61 %) to 6.50 (9 Oct, −35 %); every day beats flat. All 17
output files passed the eligibility check (largest hourly-mean error 2.5e-11), with no fallback.
19 October (the confirmed evaluation date) is auctioned on 18 October around 12:45 CET; the command
to score it is in section 10.

## 8. Rejected ideas
| Idea | Evidence (vs incumbent) | Decision |
|---|---|---|
| Pooled quarter model, price-space L1 (E2b/E3b) | −2.1 % / −1.3 % | coefficient models kept |
| Haar basis | −0.5 %, 3/8 folds | poly basis kept |
| Akima / natural-spline baseline under the learner | −1.2 % / −0.6 % | PCHIP kept (spline is best alone, not with the learner) |
| Recency weighting, half-life 90 d (vs E3) / 180 d (vs I2) | +0.1 % / −0.3 % | below the gate |
| 200 / 1000 trees | −0.3 % / +0.3 % | 500 kept |
| Residual scale 0.7 / 0.9 | −1.5 % / +0.2 % | 0.8 kept |
| Learn only slope + curvature | −0.8 % | zigzag model kept |
| No cross-zone features | −1.7 % | cross-zone kept |
| E4 analogue memory, K ∈ {8, 16, 32}, 180 days or full history | at best +0.25 % | below the gate; off |
| E6 graph shrinkage, λ ∈ {0.1, 0.3} | +0.15 % / −0.8 % | below the gate; off |
| E6 common-τ constrained quantiles | −2.1 … −3.1 % | off |

E4 and E6 stay in the code with tests, so the evidence can be reproduced; the final configuration
does not enable them.

## 9. Robustness and limitations
- **Output safety**: fallback chain learned → baseline → flat per hour, float64 projection onto
  the hourly mean, CSV validated before writing and after reload; any reconstruction error writes
  the flat baseline so the file is always compliant. Zones unseen in training get the PCHIP
  baseline. Covered by tests.
- **Short or missing history**: the artifacts carry the training prices as a history store and the
  jury file is merged over it, so lags keep their meaning; with no history at all the lag features
  are missing (LightGBM handles NaN) and the output stays valid.
- **Determinism**: repeated inference produces byte-identical files (test).
- **Drift**: the residual left after PCHIP per unit of slope halved over the year in three zones
  (section 3). The 0.8 scale and the 7/28-day history features absorb part of it, but a model
  trained once on the whole year will lag a market that keeps changing.
- **Worst cases**: errors concentrate in volatile zone-days, e.g. ZONE_2 on 22–23 Jan 2026 (13.2 and
  12.5 vs flat 22.7 and 20.8) and 24 Jun 2026 in ZONE_0/ZONE_2. Only 1 of 392 development zone-days
  is worse than flat. Edge quarters (q1, q4) carry about 60 % of the error.
- **Replica caveat**: data come from Energy-Charts rather than the official file. The timestamp
  convention was checked (section 2), but the official numbers will differ.

## 10. Reproducing
Python 3.12, pinned `requirements.txt`. From a clean checkout:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
.venv/bin/python -m pytest -q                                          # 45 tests, ~12 s
.venv/bin/python -m tools.fetch_energy_charts --out data --eval-date 2026-10-17
.venv/bin/python -m aet.eda --train data/train_prices_15min.csv --out results/eda
.venv/bin/python -m aet.backtest --train data/train_prices_15min.csv --config configs/core.yaml --folds monthly
.venv/bin/python -m aet.backtest --train data/train_prices_15min.csv --config configs/ablations.yaml --folds monthly --reference E3_coef_hist
.venv/bin/python -m aet.backtest --train data/train_prices_15min.csv --config configs/memory.yaml --folds monthly
.venv/bin/python -m aet.backtest --train data/train_prices_15min.csv --config configs/e6.yaml --folds monthly
.venv/bin/python -m aet.backtest --train data/train_prices_15min.csv --config configs/round4.yaml --folds monthly
.venv/bin/python -m aet.backtest --train data/train_prices_15min.csv --config configs/final.yaml --folds monthly --include-locked
.venv/bin/python -m aet.train --train data/train_prices_15min.csv --config configs/final.yaml --out artifacts
# one evaluated day (19 Oct works from 18 Oct ~13:00 CET)
.venv/bin/python -m tools.fetch_energy_charts --out data --eval-date 2026-10-19
.venv/bin/python -m aet.infer --hourly data/eval/2026-10-19/test_hourly.csv \
    --history data/eval/2026-10-19/history_15min.csv --model artifacts --out predictions.csv
.venv/bin/python -m tools.score --pred predictions.csv --truth data/eval/2026-10-19/truth_15min.csv
```

Runtimes on the development machine (20 threads): each monthly backtest of one LightGBM candidate
≈ 20 s, the five-quantile candidate ≈ 70 s, final training 4.8 s, inference 0.13 s per day.
