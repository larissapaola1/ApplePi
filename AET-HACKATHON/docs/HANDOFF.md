# Handoff from the claude.ai planning chat (17 Oct 2026)

## 1. Problem understanding
- Since 1 Oct 2025 the European day-ahead coupling (SDAC) clears 15-minute prices, but forecasts,
  history and forward curves are often hourly. AET wants a realistic 15-minute shape inside each
  hour that keeps the hourly mean exactly.
- The flat replication is wrong for valuation because quarter-hours show recurring intra-hour
  patterns, notably the sawtooth: ramps inside the hour with jumps at hour boundaries, plausibly
  because many bids and schedules still move in hourly blocks while demand and renewables vary
  within the hour.
- Representation: `P(h, q) = H(h) + d(h, q)` with `sum_q d(h, q) = 0`. The model learns the shape,
  never the hourly level.

## 2. Review of the weekend roadmap (`docs/source/weekend_roadmap.pdf`)
Verdict: strong plan; its main value is protecting the submission (causality, freeze, locked block,
promotion gate, eight-hour fallback plan). Maths checked:
- Cumulative-curve interpolation telescopes, so the four quarters average to the hourly price for
  any interpolator through the knots.
- Constrained L1 point decision: minimising `sum_q E|R_q - a_q|` subject to `sum_q a_q = 0` gives
  `2 F_q(a_q) - 1 + mu = 0`, hence `a_q = Q_q(tau)` with a common `tau` such that
  `sum_q Q_q(tau) = 0`. Correct.

Changes adopted (all implemented):
1. More folds: training costs seconds, and 3 folds x ~12 days is thin evidence for a 0.5 % gate.
   `--folds monthly` gives 8 development folds (Jan-Aug 2026); September stays locked. The
   roadmap's 3 folds remain the default in `configs/core.yaml`.
2. Check drift before modelling: the 15-minute market started on 1 Oct 2025 and the sawtooth
   amplitude may have changed while bidders adapted. If so, October 2025 may be atypical and recent
   data deserves more weight (`E5_recency_90d`, history features). `aet.eda` writes `drift.csv`.
3. Hour 1 is not an edge: the previous day is in the history (18 Oct when reconstructing 19 Oct),
   so its last hours feed the backward slopes (`features.prev_day_edge`).
4. Merge the supplied history over the training prices stored in the artifacts (supplied values
   win), so a short history never changes what the lag features mean.
5. Extra contender E2b: one pooled L1 model over the four quarters plus centring optimises
   price-space MAE directly (instead of coefficient L1) and is the stepping stone to the E6
   quantile variant (same rows, other alphas).
6. Schedule: with the core already written, the analogue memory (E4) can move earlier.

## 3. Design decisions in the code
- Baselines (parameter-free, also fallbacks): flat, centred linear ramp, cumulative PCHIP, Akima,
  natural spline.
- Bases: `poly` = normalised slope `[-3,-1,1,3]`, curvature `[1,-1,-1,1]`, zigzag `[-1,3,-3,1]`;
  `haar` = halves plus details. Both orthonormal and zero-sum.
- Residual models: `none`; `lgbm_coef` (three L1 LightGBM regressors on coefficients);
  `lgbm_quarter` (pooled L1 over stacked quarters, then centring). LightGBM runs with
  `deterministic` and `force_row_wise`, fixed seed.
- Features: zone, hour, weekday, weekend, season sin/cos; level, slopes at -2/-1/+1/+2 hours and
  curvature (previous day used for hour 1, NaN beyond hour 24); day mean, range, std, negative
  hours, relative level; for every zone the spread to this zone and its +/-1 slopes; the
  baseline's own coefficients; lag-1 and lag-7 residual coefficients and 7/28-day means with
  counts.
- `HistoryStore`: calendar-indexed slots, so lags are counted in days (DST-excluded dates give NaN,
  not the wrong day); arithmetic means keep shapes zero-sum.
- Inference: per-date replay with history strictly before the date; fallback chain; float64
  projection; CSV written with `%.10f`, validated before and after reload; any reconstruction
  error writes the flat baseline so the output is always compliant.
- Backtest: official metric pooled over folds; summary with gain over flat and over the reference,
  folds improved, worst zone, paired daily win rate, per-quarter MAE, fallbacks, runtimes, and the
  roadmap's promotion gate; every run appended to `results/ledger.jsonl`.
- Configs: YAML candidates deep-merged over `defaults`; `extends:` inherits another file.

## 4. Data situation
- David is not in the hackathon, so there is no official `train_prices_15min.csv`.
- Eurostat does not help (semi-annual retail prices only). The ENTSO-E API needs registration and a
  token. Chosen source: Energy-Charts `/price` (Fraunhofer ISE) — no key, EUR/MWh, CC BY 4.0 with
  attribution, 15-minute values.
- `tools/fetch_energy_charts.py`: monthly API calls with retries, cache for settled months only,
  CET/CEST delivery dates, DST days and any non-96-quarter day excluded, only days complete in all
  zones kept, anonymised `ZONE_i` (mapping in `data/zones.json`), eval folders with
  `test_hourly.csv`, `history_15min.csv` and `truth_15min.csv`. Default zones: DE-LU, FR, AT,
  IT-North.
- Assumption to verify on real data: timestamps mark interval starts. The tool prints the mean
  absolute step into each quarter position; the step into q1 (across the hour boundary) should
  usually be the largest.
- The claude.ai sandbox could not resolve `api.energy-charts.info` even after it was added to the
  allowlist (the change probably needs a new session). Locally there is no such restriction.

## 5. Verified so far (synthetic data only)
- 32 tests pass: basis and baseline invariants (including negative prices), future-sentinel
  causality for features and predictions, calendar lags across excluded dates, schema validator
  rejections, fallback chain, deterministic repeated inference, offline train -> infer, backtest,
  and the replica tools against simulated API responses (363 training dates, DST excluded).
- Full-size synthetic timings: ladder E0-E3 on 3 folds about 95 s, final training about 12 s,
  inference under 1 s.
- These are wiring checks. No accuracy on real prices has been measured yet.

## 6. Open items
- E4 analogue memory (roadmap section B): for each zone and hour, retrieve K earlier days with
  similar local hourly curves (slopes at -2..+2, level, range, cross-zone spreads, weekday/weekend),
  standardised with a scaler fitted on the training slice only and frozen; average their zero-sum
  residuals with one weight per neighbour; expose the result, nearest distance and effective count
  as features. Strictly causal in training too. Start with K in {8, 16, 32} and a 180-day window;
  also try the full history, since a 180-day window before October is summer-heavy.
- E6: graph shrinkage across zones or the constrained common-quantile decision; at most one.
- `docs/METHODOLOGY.md`: skeleton waiting for real numbers.
