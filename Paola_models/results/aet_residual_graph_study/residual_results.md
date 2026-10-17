# AET plus chronological graph residual correction

The formula is **AET + strength × graph-predicted AET error**. Correction labels are actual
quarter prices minus genuinely earlier-fit AET predictions; no AET training-set residuals
are used. Both AET and correction paths preserve the supplied hourly means.

## Scores, EUR/MWh

| stage | method | MAE |
| --- | --- | --- |
| selection_Jan_May | AET | 3.7726 |
| selection_Jan_May | Previous fixed blend | 3.7573 |
| selection_Jan_May | Previous optimized graph | 3.9695 |
| selection_Jan_May | graph raw correction | 3.9843 |
| selection_Jan_May | graph selected correction | 3.7604 |
| selection_Jan_May | no_message raw correction | 3.8617 |
| selection_Jan_May | no_message selected correction | 3.7499 |
| confirmation_Jun_Aug | AET | 4.2246 |
| confirmation_Jun_Aug | Previous fixed blend | 4.2057 |
| confirmation_Jun_Aug | Previous optimized graph | 4.3440 |
| confirmation_Jun_Aug | graph raw correction | 4.3385 |
| confirmation_Jun_Aug | graph selected correction | 4.2162 |
| confirmation_Jun_Aug | no_message raw correction | 4.2755 |
| confirmation_Jun_Aug | no_message selected correction | 4.2068 |

The January–May selection chooses **no_message**, with correction strength
**0.25**. Each candidate's strength was frozen before June–August.
Zero is an eligible strength and means AET only. Raw correction rows show strength 1, rather
than an additional selected model. The former fixed blend is an accuracy reference.

## Every selection trial

| candidate | strength | selection_MAE |
| --- | --- | --- |
| graph | 0.0000 | 3.7726 |
| graph | 0.1000 | 3.7616 |
| graph | 0.2500 | 3.7604 |
| graph | 0.5000 | 3.7972 |
| graph | 1.0000 | 3.9843 |
| no_message | 0.0000 | 3.7726 |
| no_message | 0.1000 | 3.7596 |
| no_message | 0.2500 | 3.7499 |
| no_message | 0.5000 | 3.7595 |
| no_message | 1.0000 | 3.8617 |

## Monthly scores

| stage | fold | candidate | AET_MAE | raw_MAE | selected_MAE |
| --- | --- | --- | --- | --- | --- |
| selection_Jan_May | m2026_01 | graph | 3.5632 | 3.8904 | 3.5411 |
| selection_Jan_May | m2026_02 | graph | 3.0235 | 3.1007 | 3.0221 |
| selection_Jan_May | m2026_03 | graph | 4.3423 | 4.6324 | 4.3483 |
| selection_Jan_May | m2026_04 | graph | 4.3275 | 4.4756 | 4.3087 |
| selection_Jan_May | m2026_05 | graph | 3.5203 | 3.7062 | 3.4986 |
| selection_Jan_May | m2026_01 | no_message | 3.5632 | 3.7383 | 3.5330 |
| selection_Jan_May | m2026_02 | no_message | 3.0235 | 3.0709 | 3.0203 |
| selection_Jan_May | m2026_03 | no_message | 4.3423 | 4.4641 | 4.3361 |
| selection_Jan_May | m2026_04 | no_message | 4.3275 | 4.3700 | 4.2949 |
| selection_Jan_May | m2026_05 | no_message | 3.5203 | 3.5682 | 3.4841 |
| confirmation_Jun_Aug | m2026_06 | graph | 4.3135 | 4.4208 | 4.2995 |
| confirmation_Jun_Aug | m2026_07 | graph | 4.0725 | 4.2366 | 4.0833 |
| confirmation_Jun_Aug | m2026_08 | graph | 4.2948 | 4.3643 | 4.2723 |
| confirmation_Jun_Aug | m2026_06 | no_message | 4.3135 | 4.3140 | 4.2776 |
| confirmation_Jun_Aug | m2026_07 | no_message | 4.0725 | 4.2219 | 4.0738 |
| confirmation_Jun_Aug | m2026_08 | no_message | 4.2948 | 4.2935 | 4.2745 |

## Confirmation zones

| market_id | aet_error | corrected_error | candidate |
| --- | --- | --- | --- |
| ZONE_0 | 3.1845 | 3.1690 | graph |
| ZONE_1 | 4.9230 | 4.9110 | graph |
| ZONE_2 | 4.1987 | 4.2089 | graph |
| ZONE_3 | 4.5923 | 4.5761 | graph |
| ZONE_0 | 3.1845 | 3.1721 | no_message |
| ZONE_1 | 4.9230 | 4.9023 | no_message |
| ZONE_2 | 4.1987 | 4.1831 | no_message |
| ZONE_3 | 4.5923 | 4.5698 | no_message |

The selected recipe improves confirmation MAE versus AET by **0.422%**, wins **2/3**
months, and has worst-zone degradation **-0.373%**. Promotion gate passed: **False**
(>=0.5% gain, >=2/3 months won, no zone >2% worse). The incumbent is not automatically replaced.

## Chronology and interpretation

Teacher models fit the fixed official AET recipe through the preceding month-end and predict
all valid dates in the next month, starting November 2025. For each outer evaluation month,
the correction uses only OOF labels ending at its own preceding month-end. Features include
hourly geometry, earlier quarter-price history, AET shape coefficients and adjustments versus
PCHIP, teacher age and teacher training-day count. Scaling fits these correction-training rows only.
Outer scoring still uses days 19 to month-end, as in prior comparisons. Teacher age is explicit
because OOF training includes earlier days of each month as well.

The graph and matched no-message control have the same inputs, training recipe, local prediction
path and shrinkage grid. AET itself has cross-zone features, so disabling graph messages does not
remove all cross-zone information from the combined system. Attention is a model dependency,
not a physical trading flow or causal effect. Zero correction initialization, magnitude regularization
and small selected strength limit unnecessary changes to an already strong AET prediction.

The teacher recipe and earlier development folds were inspected previously; chronological OOF
prevents target-date fitting leakage but does not make this an untouched final test. One seed was
used, with no retuning after confirmation. September and October were excluded from this study.
Earlier scoring actuals may be used as strictly earlier history on later dates, as in the challenge.

## Artifacts and commands

`artifact_graph_through_august/` and `artifact_no_message_through_august/` are isolated integrated
artifacts. Each contains its frozen AET teacher, correction network, scaler, saved history and the
January–May-selected strength. Both components end August 31. Inference never fits or recalibrates.

```bash
.venv-mae/bin/python -m Paola_models.cli infer \
  --model Paola_models/results/aet_residual_graph_study/artifact_graph_through_august \
  --hourly test_hourly.csv --history history_15min.csv --out NEW_predictions.csv
```

`plan.json`, `selection_before_confirmation.json`, `teacher_oof_ledger.csv`, and
`outer_training_ledger.csv` document all cutoffs and choices. Complete OOF predictions and row-level
outer-fold predictions are saved, plus graph attention and masked-edge diagnostics. Protected
artifacts remain unchanged; all results are local in Paola_models on branch Paola.
