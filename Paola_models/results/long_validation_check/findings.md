# Findings: does the model generalize with less training?

**The optimized model generalizes better than the original graph in all three scenarios.
The 25% graph / 75% AET blend improves slightly over AET in 3/3 scenarios.**
This is encouraging evidence across a much longer period, not proof that overfitting is absent.
All four zones retain equal weight, and settings were fixed before this diagnostic.

## All validation results, EUR/MWh

| training_dates | validation_dates | Original TGNN | Optimized TGNN | AET | Fixed blend | Flat |
| --- | --- | --- | --- | --- | --- | --- |
| 91.0000 | 242.0000 | 5.1169 | 4.3266 | 3.9907 | 3.9726 | 7.6085 |
| 180.0000 | 153.0000 | 4.5147 | 4.2842 | 4.0263 | 4.0124 | 8.2218 |
| 271.0000 | 62.0000 | 4.3873 | 4.2260 | 4.0036 | 3.9929 | 8.1962 |

The longest run trained on October–December (91 valid dates) and validated every available
January–August date (242 dates, 92,928 quarter-price observations). The older monthly evaluation
covered 98 dates. The shorter validation windows overlap the longer one, so they are not
independent replications and their row counts should not be added to claim extra independent data.

## Overfitting warning in the original graph

For the 91-day training run:

- Original graph: training **2.905**, validation **5.117**.
- Optimized graph: training **3.091**, validation **4.327**.
- AET: training **2.993**, validation **3.991**.

The original graph fits training more closely yet predicts later dates less accurately than the
smaller optimized graph. That pattern is consistent with overfitting or less stable extrapolation.
It supports keeping the simpler graph rather than pursuing a lower training loss. It does not
identify overfitting as the sole cause: calendar/price regimes and model age change too.
Even flat-baseline error changes between training and validation, showing different target difficulty.

## More training helps on the same July–August dates

| scenario | Original TGNN | Optimized TGNN | AET | Fixed blend |
| --- | --- | --- | --- | --- |
| train_3_months | 6.1340 | 4.6584 | 4.3253 | 4.2950 |
| train_6_months | 4.8273 | 4.4173 | 4.1090 | 4.1013 |
| train_9_months | 4.3873 | 4.2260 | 4.0036 | 3.9929 |

All rows here predict identical target dates. The three models use progressively later training
cutoffs, so more data also means fresher data. These effects cannot be separated by this test.
The separate training-history-only replay in the full report shows that fresh earlier-date history
usually helps the optimized model and blend; it is a legitimate input, not a weight update.
The original graph is less stable: fresh lag inputs do not improve it under every cutoff.

## How strong is the blend advantage?

| scenario | validation_dates | blend_minus_AET_MAE | gain_vs_AET_pct | fraction_dates_blend_better | paired_daily_block95_low | paired_daily_block95_high |
| --- | --- | --- | --- | --- | --- | --- |
| train_3_months | 242.0000 | -0.0182 | 0.4552 | 0.5537 | -0.0277 | -0.0097 |
| train_6_months | 153.0000 | -0.0139 | 0.3444 | 0.5817 | -0.0236 | -0.0046 |
| train_9_months | 62.0000 | -0.0106 | 0.2652 | 0.5806 | -0.0199 | -0.0006 |

The observed gains are about **0.27%–0.46%**,
still below the existing 0.5% promotion requirement. Paired daily seven-observation block-bootstrap
intervals are reported, but depend on resampling assumptions and repeated use of development data.
They cannot certify an independent final-test advantage.

## Recommendation

Keep AET as the main baseline and retain the smaller graph/blend as research candidates.
The blend's small edge is more credible than a single 12-day September number because it repeats
over longer fixed-weight windows, but it does not justify promising a sub-5 score on every period.
Use future organizer-held dates for final evidence; do not reduce training merely because one
historical window happens to score lower. Model age, seasonal coverage, and baseline difficulty matter.

No September/October assessment was run here, no settings were retuned, and no saved model
artifacts changed. All dates had been inspected in prior project work, so these redesigned splits
are diagnostic validation, not a newly untouched holdout.

[Full protocol, history stress tests, charts, and CSVs](generalization_report.md).
