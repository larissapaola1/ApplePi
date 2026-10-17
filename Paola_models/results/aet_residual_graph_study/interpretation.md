# Integration outcome

**AET + chronological OOF error correction is implemented and trained.** The new formula is
AET + 0.25 × learned correction. The graph learns the difference between actual quarter prices
and an AET prediction generated without fitting that date. This is different from blending AET
with a second independent PCHIP reconstruction.

## Same-date June–August comparison

| Method | MAE, EUR/MWh |
| --- | ---: |
| Previous fixed graph/AET blend | **4.20575** |
| AET + no-message correction | 4.20680 |
| AET + graph correction | 4.21623 |
| AET | 4.22464 |

These scores cover June 19–30, July 19–31 and August 19–31: 38 dates, four equally weighted
zones, 14,592 quarter prices. They are **not September scores**. Both correction variants selected
strength 0.25 on January–May; the no-message variant was selected before confirmation.
Nothing was retuned after seeing June–August.

Subsequent user-requested September assessment: graph correction **5.06125**, no-message
correction **5.05213**, AET **5.04427**. Neither correction improves on AET in that block;
both still use the frozen 0.25 strength. This separate replay does not change the development
selection or results below. [September details](../aet_residual_graph_september/assessment_results.md).

The graph correction improves AET by **0.199%**. The no-message correction improves AET by
**0.422%**, improves all four zone averages, and wins two of three months. It still misses the
existing 0.5% promotion threshold and is slightly worse than the previous blend. Its July MAE
is 4.07377 versus AET 4.07254, so the improvement is not uniform across months.

At full strength, both correction networks worsen MAE. Shrinkage is essential in this run:
full graph correction 4.33846; full no-message correction 4.27548. A learned error estimate
is noisy, and applying all of it is not automatically beneficial.

## What this means for neighboring-zone information

The residual-learning integration is mildly helpful, but the matched no-message control performs
better than the graph on both selection and confirmation. AET already uses cross-zone features.
The current experiment gives no measured reason to add its extra graph message path to the
best correction recipe. It does not prove that zone interactions never matter.

The no-message variant remains a neural network with local temporal convolution and a direct
feature path. It is not a GNN with meaningful message passing. Both variants use the same
training recipe and feature set, and both include AET's existing cross-zone information.

## Information boundaries and saved models

- Teachers fit the official AET recipe through the preceding month-end and predict the next month.
- OOF labels start November 2025. January correction training uses only November–December errors.
  Later outer folds add only dates before their own scoring month.
- The final artifacts use 303 OOF dates through August. Each network has 3,710 parameters.
- Teacher age and training-day count are explicit features because early OOF teachers have less history.
- Features, labels and scalers never use outer scoring targets. Earlier-day actuals may enter later-day
  history, as in the existing daily assessment protocol.
- Both saved components end August 31. Inference freezes the teacher, correction, scaler and strength.
- Every correction has zero hourly mean; final prices preserve the supplied hourly averages.
- The study did not evaluate September or October. Previously inspected development dates and
  the existing AET recipe mean these results are not an independent test of the complete research process.

Both integrated artifacts are usable through `Paola_models.cli infer`:
`artifact_graph_through_august/` and `artifact_no_message_through_august/`.
All original AET and prior graph artifacts passed unchanged-hash checks. Work is local on Paola;
no commit or push was made.

## Validation and next decision

The test suite checks chronological teacher fitting, rejection of overlapping OOF/outer cutoffs,
zero correction for a perfect teacher, future-label poisoning, directed edge masking, negative-price
mean preservation, frozen inference and integrated artifact save/load/CLI output.
Both saved variants passed synthetic inference smoke checks. Twelve prediction outputs were
reloaded from CSV and validated; maximum hourly mean error was 1.14e-13 EUR/MWh.

Keep the previous best blend as the accuracy reference. Before adding complexity, repeat the fixed
correction recipe across seeds and a predeclared outer evaluation, and examine difficult ramp periods.
Do not tune correction strengths to September's already known result. These deterministic correction
models do not inherit the separate Bayesian graph's uncertainty intervals.

See [every trial and monthly/zone comparison](residual_results.md) and
[training/inference documentation](../../AET_RESIDUAL_GRAPH.md).
