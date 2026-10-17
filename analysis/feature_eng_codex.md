# Feature engineering recommendations from the final EDA

Source: [analysis/eda_final.ipynb](analysis/eda_final.ipynb). Input availability follows the official [Data/README.txt](Data/README.txt). These are recommendations for experiments, not validated improvements.

**Start with the shape of the supplied hourly curve:** neighboring-hour slopes, quarter position, zone, curvature, and daily range. Add calendar context next, then test historical residual patterns and cross-zone information separately.

## 1. Frame the target correctly

This is reconstruction of four quarter-hour prices from a known hourly average. For date `d`, zone `z`, hour `h`, and quarter `q`:

```text
H[d,z,h] = mean of the four observed quarter prices during training
r[d,z,h,q] = price[d,z,h,q] - H[d,z,h]
```

Train on `r`, or on a correction to a slope-based residual baseline. At inference, `H` comes from the supplied hourly file. The four predicted residuals must sum to zero:

```text
r_final[q] = r_pred[q] - mean(r_pred[1:4])
price_pred[q] = H + r_final[q]
```

Here `mean(r_pred[1:4])` means all four quarters, not Python slice syntax. Apply this projection within each `(delivery_date, market_id, hour_index)` group and validate the hourly mean after output serialization. Projection guarantees consistency, but does not guarantee a lower MAE.

The notebook's variance decomposition shows that within-hour residuals account for only **4.0–7.0%** of price variance. Learning the already-known hourly level again would distract from the actual task.

## 2. Evidence that sets the priorities

The following values come from saved numerical outputs; some narrative cells contain older figures.

| EDA finding | Evidence | Feature implication |
|---|---|---|
| Hourly slope predicts within-hour tilt | Correlation of `H[h+1] - H[h-1]` with `Q4 - Q1`: **0.91, 0.86, 0.76, 0.83** for ZONE_0–ZONE_3 | Make slopes the first feature family |
| A simple slope rule already helps substantially | Descriptive full-sample MAE: flat **7.58**, slope × 1 **5.46**, slope × 2 **4.57 EUR/MWh** | Require learned features to beat a slope baseline out of time |
| Quarter position matters | Q1/Q4 mean absolute residuals are roughly **2–3×** Q2/Q3 | Encode quarter position and its interaction with slope |
| Zone behavior differs | Flat MAE: **7.91, 9.43, 5.59, 7.38** for ZONE_0–ZONE_3 | Include categorical zone and zone-dependent slope effects |
| Winter is quieter | Mean absolute residual **5.0** in winter versus **8.3–8.6** in other seasons | Test seasonal amplitude and timing interactions |
| Residuals have heavy tails | Excess kurtosis **7.57–24.21** across zones | Optimize the MAE objective; retain extreme observations |

The slope multiplier was explored on the same observations used to report its error. These numbers describe the dataset; they are not holdout performance. Price-pattern explanations involving solar are hypotheses from timing, not measurements of solar generation.

## 3. First feature set: local hourly geometry

Compute these features on **one row per date, zone, and hour**, then repeat them for the four quarters if using a quarter-level model. Sort and group by date and zone before taking neighboring hours.

Let `H_prev = H[h-1]`, `H_next = H[h+1]`, and `o_q = (q - 2.5) / 4`, giving quarter-center offsets `[-0.375, -0.125, 0.125, 0.375]` hours.

| Feature | Definition | Purpose / EDA connection |
|---|---|---|
| `market_id` | Categorical zone, not an ordinal price scale | Different risk and tilt strength per zone |
| `hour_index` | 1–24; hour 1 means 00:00–01:00 | Daily ramp and turning-point context |
| `quarter_index`, `quarter_offset` | Categorical 1–4; `o_q` | Sawtooth position and signed tilt |
| `hourly_level` | `H[h]` | Price-regime context |
| `slope_back` | `H[h] - H_prev` | Direction into this hour |
| `slope_forward` | `H_next - H[h]` | Direction out of this hour |
| `slope_center` | `(H_next - H_prev) / 2` | Main predictor of within-hour tilt |
| `abs_slope_center` | `abs(slope_center)` | Ramp magnitude |
| `curvature` | `slope_forward - slope_back` | Distinguish straight ramps from bends |
| `is_turning_point` | `slope_back * slope_forward < 0` | Detect peaks and valleys from that day's curve |
| `slope_x_offset` | `slope_center * o_q` | Explicit quarter-specific slope prediction |
| `is_first_hour`, `is_last_hour` | `h == 1`, `h == 24` | Identify one-sided slope estimates |
| `prev_missing`, `next_missing` | Neighbor availability indicators | Distinguish missing context from a genuine zero slope |

At hour 1, use `H[2] - H[1]` as the center-slope fallback; at hour 24 use `H[24] - H[23]`. Do not wrap hour 24 to hour 1 of the same day. Optionally add neighbors two hours away after the first feature set has been assessed.

All 24 target-day hourly values are supplied, so **using the next hour's hourly average is allowed**. Using the next hour's observed quarter prices is not.

### Slope baseline

```text
r_base[q] = k_z * slope_center * o_q
```

Its hourly mean is automatically zero. Compare fixed `k = 1` and `k = 2`; estimate any zone-specific `k_z` using training folds only. The notebook's full-sample best grid values are hypotheses, not deployable calibration estimates. A learned model can predict `r - r_base` and then apply the same final consistency projection.

For linear models, explicitly test `zone × slope_x_offset` and `hour × slope_x_offset`. Tree models can learn these interactions from their component features, so avoid generating every possible interaction automatically.

## 4. Second feature set: daily regime and calendar

These features use only the supplied target-day hourly curve or its date.

| Feature family | Suggested definitions | Why test it |
|---|---|---|
| Daily amplitude | `day_range = max(H) - min(H)`; daily standard deviation | The EDA links wide hourly daily ranges with larger within-hour movement |
| Relative level | `H[h] - mean(H)`; optionally divide by `max(day_range, 1 EUR/MWh)` | Position within the day's own price curve |
| Low-price context | `H[h] < 0`, daily count of negative hourly averages | Cheap/negative-price context visible in the permitted inputs |
| Clock cycle | `sin/cos(2*pi*(h-1)/24)` | Smooth transition across midnight; retain raw hour for trees |
| Annual cycle | `sin/cos(2*pi*(day_of_year-1)/365.25)` | Smooth seasonal timing without a separate model for every month |
| Winter flag | Month in December–February | Test the observed winter amplitude difference directly |
| Day/night | Day = `hour_index` 7–22; night otherwise | Matches the notebook's 06:00–22:00 definition |
| Weekday/weekend | Weekday category and weekend flag | Plausible calendar context; incremental value is not established by this notebook |

Useful interaction candidates are `winter × slope_x_offset`, `day_range × quarter_offset`, and `annual_cycle × hour`. Start with few interactions and assess their contribution.

Negative **hourly** averages do not reveal all negative quarters: a positive hourly average can contain negative quarter prices. Never substitute the observed target-day count of negative quarters for these permitted hourly features.

Do not hard-code a nonnegative prediction constraint for ZONE_2 just because its training minimum was zero. Do not log-transform raw prices, since negative and zero prices are valid. Retain raw EUR/MWh features; fit any scaling or learned thresholds on training data only.

## 5. Additional experiments: history and other zones

The EDA gives a stronger case for local slopes than for these extensions. Treat their benefit as unproven until tested.

**Historical shape features:** join residuals for the same zone, hour, and quarter on exact dates `d - 1 day` and `d - 7 days`. Test trailing 7- and 28-calendar-day means or medians of residuals and absolute residuals, together with observation counts and missing indicators. Signed residuals describe typical direction; absolute residuals describe typical magnitude.

Use only history with `delivery_date < d`. Join by calendar date rather than shifting rows: the official dataset omits DST-change dates. Define a fixed fallback, such as a training-only zone/hour/quarter template and then the slope baseline. Recenter fallback residual vectors to preserve the hourly mean. A past-day final-hour average can optionally supply extra context at the start of the target day; mark its availability separately. Tomorrow's values are unavailable at the target-day end.

**Cross-zone hourly features:** for each other zone `j`, test `H_j[h] - H_z[h]` and zone `j`'s hourly slopes. The notebook observes shared daily shapes and some coincident extremes, which motivates testing shared information. It does not establish predictive lift. Use current-day hourly inputs for these features, never another zone's current-day quarter prices.

After solution freeze, recent organizer-supplied history may feed only inference rules fixed beforehand. Do not refit templates, slope multipliers, model weights, or hyperparameters on that history; predetermined rolling-feature calculations are distinct from recalibration.

## 6. Keep diagnostic quantities out of the predictors

The following notebook quantities are useful as training labels or retrospective diagnostics, but are unavailable predictors for a target date:

- Actual `Q4 - Q1`, within-hour standard deviation, quarter-price range, and boundary jumps.
- Actual residual magnitude, target-day quarter-price extremes, or counts of negative quarters.
- Centered rolling price/residual statistics that include future delivery dates.
- Residual templates, encodings, scalers, or thresholds fitted using validation targets.

Construct training hourly averages from exactly four distinct quarters per hour, with unique keys and complete 24-hour zone-days. Compute inference features from the corresponding supplied hourly column. Keep arithmetic precision consistent and check sensitivity to supplied hourly-input rounding.

Only official package data and deterministic calendar fields are needed. Do not add external weather, renewable production, load, holiday datasets, or inferred zone geography under the package's external-data restriction.

## 7. Validate feature families in order

1. **Baselines:** flat hourly repetition, slope × 1, and slope × 2.
2. **Local geometry:** zone, hour, quarter position, hourly level, slopes, curvature, turning-point and edge flags.
3. **Regime/calendar:** daily amplitude, relative level, low-price context, clock and annual cycles, winter/day-night flags.
4. **History:** exact-date lags and strictly trailing summaries, with fixed missing-history behavior.
5. **Cross-zone:** supplied hourly spreads and slopes.

Use chronological expanding-window folds, keeping each delivery date and all its zones/hours/quarters together. Fit every learned transformation using only the fold's training dates. If earlier validation dates become historical inputs for later dates, reveal them only as the organizer's history-release schedule would permit.

Score quarter-price MAE per date and zone, average dates within each zone, then average the four zones equally, matching the notebook's scoring function. Also inspect error by zone, Q1–Q4, day edges, turning points, and high-amplitude days. Evaluate the final projected predictions; an improvement in a residual coefficient alone is insufficient.

The notebook explored the full October 2025–September 2026 dataset, including September extremes. **September is therefore already EDA-exposed**, despite the root README's earlier holdout plan. Keep it out of further feature selection; any later September assessment must disclose that exposure. Compare feature families on earlier chronological folds, and reserve truly unseen organizer evaluation dates for final evidence without accessing restricted dates before freeze.

**Recommended first implementation:** a mean-preserving slope baseline plus a residual-correction model using local geometry, daily range, and calendar context. Promote historical or cross-zone additions only when they improve chronological validation consistently.
