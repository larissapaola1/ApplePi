# Does the AET model make sense given Paola's EDA?

**Yes: the architecture is well aligned with the EDA, and the recorded backtests support adding a learned correction.** The strongest connection is that neighboring hourly slopes explain much of the quarter-hour tilt. The main open question is where the remaining errors occur, rather than whether the model targets the right problem.

Scope: review of [EDA_final_Paola.ipynb](EDA_final_Paola.ipynb), [EDA_analysis.md](EDA_analysis.md), the [aet implementation](../AET-HACKATHON/aet/), experiment configurations, diagnostics, and the saved [official model manifest](../AET-HACKATHON/artifacts_official/manifest.json). Scores below are existing results from [METHODOLOGY.md](../AET-HACKATHON/docs/METHODOLOGY.md) and [VOLATILITY.md](../AET-HACKATHON/docs/VOLATILITY.md); this review does not rerun training or backtests. Notebook figures quoted here use saved calculation outputs, since some narrative cells contain older numbers.

## EDA → model alignment

| What the EDA shows | What the final model does | Assessment |
|---|---|---|
| Hourly averages already account for 93–96% of price variance; the task is the remaining within-hour movement | Starts with a mean-preserving cumulative PCHIP baseline, then learns observed-minus-baseline corrections | Correct target: reconstruct the shape around known hourly prices |
| Neighboring-hour slope correlates **0.76–0.91** with `Q4 − Q1`; quarter profiles look approximately linear | Uses local slopes and curvature, baseline-shape coefficients, and a slope/curvature/zigzag output basis | Strongest EDA justification for the design |
| Q1/Q4 errors are roughly 2–3× Q2/Q3 errors | The slope basis `[-3, -1, 1, 3]` naturally puts larger deviations at the outer quarters | Captures the observed geometry without four unrelated price predictions |
| Zones differ in level, volatility, and tilt strength | Three LightGBM regressors are pooled across zones with categorical zone and cross-zone features | Allows shared patterns and zone-dependent responses; separate models per zone are not automatically necessary |
| Winter is quieter; wide-range days move more within hours | Includes annual sine/cosine, hour, daily range/std, relative level, and negative-hour count | Relevant context is already present; explicit winter/day-night flags are optional experiments |
| Residuals have heavy tails | Uses L1 coefficient regression and selects models using reconstructed-price MAE | Reasonable, but coefficient MAE is only a surrogate for the scored price MAE |

The three zero-sum basis vectors span **every** possible four-quarter residual whose sum is zero. Their names describe components, not a restriction to smooth or linear shapes. All coefficients can vary with the inputs. A separate quarter-position feature is therefore unnecessary for this coefficient model; quarter position enters through decoding. See [basis.py](../AET-HACKATHON/aet/basis.py) and [models.py](../AET-HACKATHON/aet/models.py).

## What the recorded evidence supports

The deployed configuration is **cumulative PCHIP + 0.8 × learned correction**, with history, cross-zone inputs, and previous-day boundary features enabled. Shrinkage applies to the learned correction, not to the entire deviation from the hourly average. A final numerical projection preserves hourly means.

| Model | Jan–Aug development MAE | September 19–30 MAE |
|---|---:|---:|
| Flat hourly repetition | 8.1276 | 10.8988 |
| PCHIP baseline | 5.6185 | 6.9270 |
| Unscaled coefficient model + history | 3.9838 | 5.4270 |
| Final model | **3.9479** | **5.0444** |

EUR/MWh; lower is better. Development uses eight rolling-origin folds. The final September result improves **53.7% over flat** and **27.2% over PCHIP**. This supports the learned pipeline beyond interpolation, although it does not identify the contribution of every feature family.

Paola's simple slope × 2 rule scores **4.57** versus flat **7.58** on the full EDA sample. **Do not compare 4.57 directly with 3.9479:** the dates, training, and selection differ. A slope baseline evaluated on the same chronological folds is the most informative missing comparison. The EDA's full-sample multiplier search is descriptive, not validation.

## Most useful insights and next checks

- **Investigate ZONE_2 first.** It has the lowest full-year flat error and weakest slope–tilt correlation, yet its September final MAE (**5.51**) is second highest, behind ZONE_1 (**6.26**). This suggests that a calm market is not necessarily the easiest to reconstruct on new dates. Compare per-zone flat/PCHIP gains and daily errors before concluding that the model fails there.
- **Inspect the shape components and difficult hours.** Compare slope-only versus all three coefficients on development folds, and break down final errors by Q1–Q4, turning points, morning/evening ramps, and first/last hours. The EDA motivates these checks but does not prove which component fixes each error.
- **Keep complexity tied to demonstrated gains.** Recorded ablations support cross-zone features; removing them worsened unscaled development MAE by 1.43%. Previous-day edges have a smaller recorded benefit. Analogue memory, graph smoothing, quantile alternatives, and magnitude-history features were not promoted. Shared market rhythms alone do not justify forcing zones toward identical shapes.
- **Distinguish information availability from leakage.** All target-day hourly values are supplied, so forward hourly slopes and daily range are legitimate. Historical quarter-price features use strictly earlier calendar dates, including correct handling of DST gaps. Tomorrow's curve and current-day observed quarter shapes are unavailable. See [features.py](../AET-HACKATHON/aet/features.py) and [history.py](../AET-HACKATHON/aet/history.py).

## Feature engineering: what to add, and why

The existing [feature builder](../AET-HACKATHON/aet/features.py) already covers most obvious recommendations: ±2-hour context, backward/forward slopes, curvature, zone, calendar seasonality, daily amplitude, cross-zone spreads, baseline coefficients, exact 1-/7-day residual-coefficient lags, and trailing 7-/28-day means. Adding those again under different names is unlikely to help. The next experiments should expose specific situations that are harder for the current representation.

In the definitions below, `H` is the supplied hourly price, `b = H[h] - H[h-1]`, and `f = H[h+1] - H[h]`. All candidates are hypotheses until evaluated on development folds.

| Priority | Proposed features | EDA rationale and implementation |
|---|---|---|
| High | Turning-point flag `b*f < 0`; peak/valley flags; `abs(b) + abs(f)` | A centered slope can cancel at a sharp peak or valley. Separate a genuinely flat hour from a reversal. Curvature is already present; these features make its context explicit. Keep unavailable neighbors missing. |
| High | Ramp imbalance `(f-b)/(abs(b)+abs(f)+1)`; normalized local movement `(abs(b)+abs(f))/max(day_range, 1)` | Distinguish steady ramps, asymmetric ramps, and a large local move relative to an otherwise calm day. Constants here are in EUR/MWh; retain the original unnormalized features. |
| High | Previous-day availability, days since last available history, available-lag flags | Existing rolling counts help, but a calendar-gap flag can distinguish missing edge context from a zero slope. Evaluate the first two and last two hours separately. Never fill with tomorrow's data. |
| Medium | Hour relative to the day's hourly minimum/maximum; distance to morning/evening peak | EDA timing shifts with season. The actual target-day curve may locate a ramp better than a fixed clock window. Define search windows and tie-breaking in advance, using hourly inputs only. |
| Medium | Midday-valley depth relative to a fixed morning/evening hourly reference; share of near-zero hourly values | Distinguish a deep valley from a uniformly cheap day, particularly in ZONE_3. Choose any near-zero threshold on training folds; these are price-curve descriptors, not solar-generation measurements. |
| Medium | Winter flag, day/night flag, seasonal interactions with local slope | Winter movement is lower, and the day/night price ordering changes seasonally. Raw hour and annual sine/cosine already encode much of this, so require an incremental gain. Daytime in the EDA is 06:00–22:00, or `hour_index` 7–22. |
| Medium | Cross-zone dispersion of hourly levels/slopes; disagreement with the other-zone mean slope | Existing pairwise inputs describe individual zones; summaries may identify coupled versus divergent hours more simply. Compare these with the existing features rather than imposing a common output shape. |
| Lower | Recent-minus-long-window coefficient means; lag-to-current hourly-curve mismatch | Signed 7-/28-day means already exist. Their differences may expose changing shape, while curve mismatch can indicate when yesterday's shape is irrelevant. Historical coefficient targets are corrections to PCHIP, not residuals around flat. |

Use explicit interactions mainly where diagnostics show a problem: turning point × zone, valley depth × hour, or winter × ramp size. LightGBM already learns interactions, so a large manual interaction expansion risks fitting noise.

**Do not automatically promote magnitude-history features.** They already exist behind `volatility_history`; the documented trial scored **3.9590 versus 3.9479**, improving only **2/8 folds**. Likewise, previous-day edge information is already enabled. An EDA-motivated idea can be useful for explanation or risk annotation without improving point predictions.

## Model improvements worth testing

### 1. Establish how much complexity is buying

Add fixed slope × 2 and a training-fitted zone-specific slope rule to the same development comparison as flat, PCHIP, and the incumbent. Use identical dates, history availability, and scoring. The current `linear` baseline in [baselines.py](../AET-HACKATHON/aet/baselines.py) implements slope × 1; it does not reproduce the EDA's stronger slope × 2 benchmark.

Then compare the complete incumbent against an otherwise identical no-history model. The core configurations include history/no-history candidates, but both inherit cross-zone inputs: that comparison is not an ablation of all shared-market information. Change one family at a time, then confirm a promising change on the actual final configuration, including its 0.8 correction scale.

### 2. Regularize amplitude before adding a new architecture

The retained 0.8 correction scale suggests that unscaled corrections are too aggressive on average. It does **not** prove that every volatile hour needs damping. Inspect Q1/Q4 signed error and predicted-minus-observed within-hour range first: a model can overreact on ordinary hours while underestimating rare ramps.

If biases are stable, test a small number of zone-specific or basis-component scales, regularized toward 0.8. Multiplying each coefficient by its own fixed scale preserves zero-sum reconstruction. Estimate scales inside chronological development, with a separate confirmation slice; avoid a separate scale for every zone/hour/month combination. This is a new experiment, not a recommendation to recalibrate the frozen model on evaluation history.

### 3. Match output complexity to evidence

Test slope-only, slope+curvature, and all three components under the same settings. `n_components` is already supported in [models.py](../AET-HACKATHON/aet/models.py), and [round4.yaml](../AET-HACKATHON/configs/round4.yaml) already specifies the two-component candidate; inspect its existing record before repeating it. If zigzag adds little stable price-MAE improvement, stronger regularization of that component may be preferable to removing useful slope flexibility.

Keep price MAE as the selection criterion. Independent coefficient medians do not generally minimize quarter-price MAE jointly. The repository already offers a quarter-level L1 model and constrained quantile alternative, but the recorded quantile trials worsened development MAE by **1.67–2.70%**. A theoretically closer objective is not evidence of a practical improvement.

### 4. Specialize only after locating repeatable errors

ZONE_2's weaker slope relation makes it a candidate for targeted diagnosis, not an automatic separate model. First compare local-geometry and historical-feature ablations by zone. If one zone has a consistent residual bias across months, test a small zone-specific correction or scale before fitting four independent models with much less data each.

Similarly, test a shallow regime model only if errors differ consistently between calm hours and reversals/steep ramps. Define regimes from available hourly inputs. Do not train an inference gate that requires the actual target-day quarter spread. Any specialist's improvement must survive evaluation of the complete system, including routing errors and ordinary hours.

### 5. Avoid repeating experiments that already failed

The documented pipeline comparisons did not support replacing PCHIP with Akima/spline, changing to a Haar basis, or promoting analogue memory and graph smoothing. The latter gains were below the existing promotion gate. More trees and recency weighting also appear in experiment configurations; their presence is not a positive result. Recover the matching experiment ledger before presenting them as new ideas or successful improvements.

A lower-complexity baseline plus targeted feature additions is the strongest next direction. One year of data contains few independent seasonal cycles; a large seasonal ensemble or sequence model would need stronger evidence than the descriptive EDA provides.

## A concrete development plan

1. **Build the error map before tuning.** Reuse [backtest.py](../AET-HACKATHON/aet/backtest.py) for daily/quarter MAE and [volatility.py](../AET-HACKATHON/aet/volatility.py) for hourly errors, Q1/Q4 bias, range bias, history counts, and risk groups. Add turning-point and ramp buckets computed from hourly inputs. Show sample sizes and each bucket's share of total absolute error, not just its conditional MAE.
2. **Run a short, predeclared comparison.** Start with the slope benchmark; then test turning-point/local-amplitude features; then history/edge availability features. Keep each family separate so a result has a clear interpretation. Retain the current official artifact while researching candidates.
3. **Use paired evidence.** Compare errors on the same date/zone observations, retaining all zones of a date together in any uncertainty resampling. A block bootstrap by consecutive dates can illustrate uncertainty under temporal dependence; it does not undo repeated selection on the same folds. Inspect worst days and zone regressions alongside aggregate gain.
4. **Apply the existing promotion gate.** Require at least 0.5% lower MAE, improvement in at least two-thirds of eight folds (**6/8**), and no zone more than 2% worse. Add a check that the gain is not concentrated in one extreme day. This is an engineering threshold, not statistical proof. Keep a record of unsuccessful trials too.
5. **Verify the exported path for a selected candidate.** Check exact calendar lags, unavailable-history behavior, no current/future-quarter access, complete keys, finite predictions, and hourly-mean error ≤0.0001 after writing and reloading the CSV. Existing tests cover many of these properties. Test added behavior rather than duplicating existing checks.

The development folds have already been reused, and September has been explored. Further experiments are exploratory; a new split within already inspected data is a confirmation aid, not an untouched final test. Never reopen September to choose the next feature set.

For existing diagnostics, the following command uses only development folds and writes a separate report. Run from `AET-HACKATHON/`; it trains fold models and does not overwrite the frozen artifacts:

```bash
python -m aet.volatility diagnose \
  --train ../Data/train_prices_15min.csv \
  --config configs/final.yaml \
  --out results/eda_model_review
```

This command is provided as the next executable step; it was not run for this document update.

## Business implications: where quarter-hour detail matters

**The model refines timing within a known hourly price.** It can support finer valuation or scheduling analysis when that hourly curve is available. It does not establish an ability to forecast unknown hourly prices or an executable trading advantage; the timestamp at which inputs are available must match the proposed use.

| EDA observation | Potential operational implication | Useful model check |
|---|---|---|
| Q1/Q4 carry much larger deviations than Q2/Q3 | Assets or contracts with uneven quarter-hour exposure are more sensitive to reconstruction quality | Q1/Q4 MAE, signed bias, and exposure-weighted valuation error |
| ZONE_0/ZONE_1 have heavier tails and evening spikes | Large quarter-specific errors can be concentrated in a few evening periods | Worst-day errors, evening-ramp errors, and range underestimation |
| ZONE_3 is cheaper and has more negative-price observations | Flexible consumption/storage scenarios merit evaluation around the midday valley | Negative-quarter detection, valley timing, and fees/physical constraints in a separate operational simulation |
| ZONE_2 has a high average level but less full-year variability | Average price and opportunity from timing are distinct; a high level alone does not imply high flexibility value | Improvement over that zone's own flat/PCHIP baseline, not absolute MAE alone |
| Day/night price ordering changes across seasons | A fixed year-round rule such as “night is cheaper” can misrepresent the observed profile | Seasonal valuation errors and timing of daily extrema |

Negative prices alone do not establish profitable consumption, and a reconstructed spread is not realized profit. Any operational study needs actual exposures, decision timing, efficiency, capacity, transaction costs, and the relevant settlement arrangement. Solar explanations in the EDA remain interpretations of the price profile because generation was not observed.

### Translate price accuracy into the relevant exposure

For a fixed quarter-hour energy exposure `E_q` in MWh, valuation error is `sum(E_q * (predicted_price_q - actual_price_q))`. A 10 EUR/MWh price error on **1 MW for 15 minutes** corresponds to **0.25 MWh × 10 = EUR 2.50**; 1 MWh delivered during that quarter would correspond to EUR 10.

An important consequence of the constraint: with identical MWh exposure in all four quarters of an hour, the net hourly valuation error cancels because both predicted and actual prices average to the same supplied hourly value. Business value therefore depends on **uneven intra-hour exposure or a quarter-specific decision**, rather than uniform hourly volume. Absolute quarter errors still matter when assessing gross errors or separate positions.

Keep business metrics alongside the official equal-zone MAE. An exposure-weighted score answers a different question and should not silently replace the competition objective. For a scheduling experiment, evaluate realized outcomes on held-out dates against an hourly-flat policy and an explicitly labeled perfect-information benchmark.

### An immediate use for diagnostics: review priority

The separate risk index already combines hourly slope, daily standard deviation, and past residual magnitude. [VOLATILITY.md](../AET-HACKATHON/docs/VOLATILITY.md) reports development low-risk versus high-risk model MAEs of **1.52 vs 5.93** in ZONE_0, **2.83 vs 8.50** in ZONE_1, **2.04 vs 5.75** in ZONE_2, and **2.30 vs 7.32** in ZONE_3.

These results support testing the index as a way to prioritize manual review or flag less reliable shape estimates, even though magnitude features did not improve the price model. The categories are relative within each zone, not a calibrated probability or common error threshold. Check high-risk sample counts and missed extremes before operational use; a hidden quarter spike can occur without an obvious hourly signal.

## Interpretation and implementation details to keep straight

- **“Spread” needs a definition.** The notebook's median 5.9 EUR/MWh figure describes within-hour standard deviation, not `max(Q1..Q4) - min(Q1..Q4)`. Historical `within_range` in the code and predicted-range bias use the latter. Label both explicitly.
- **Distribution fit is descriptive.** A two-Gaussian fit to raw prices does not establish two causal market regimes or prove that a two-model architecture will improve reconstruction. L1 is justified primarily by the official MAE objective; heavy tails help explain its practical appeal.
- **Shared average shape is not identical daily behavior.** Cross-zone features can help, but synchronized historical extremes do not justify forcing all zone predictions toward a common curve. The existing weak smoothing results reinforce that distinction.
- **Model features and diagnostic labels differ.** Actual quarter ranges, negative-quarter counts, `Q4 − Q1`, and realized boundary jumps may explain errors retrospectively. They cannot be target-day predictors. Past versions are allowed only through strictly earlier history.
- **Operational robustness deserves its own checks.** Missing model/history can trigger flat fallback with valid inputs; malformed hourly inputs are rejected. Record fallback counts, history coverage, and output validation separately from MAE. Existing numerical discrepancies between backtest and CSV paths also justify checking export behavior rather than claiming bit-identical outputs.

## Limits on the conclusion

September was already EDA-exposed and its assessment covers only 12 dates, so it is not a wholly untouched test. The full-year official artifact also includes September training data; the reported September score belongs to a separate model trained through August. Keep those artifacts distinct when plotting predictions or reporting performance.

The repository separately documents unresolved prior external-replica/October-data exposure in [PROVENANCE.md](../AET-HACKATHON/docs/PROVENANCE.md). Technical alignment and official-data backtests do not settle competition eligibility. These limits qualify the strength of the evidence, while the architecture itself remains a coherent response to the EDA.
