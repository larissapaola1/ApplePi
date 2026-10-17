# Repository guidance: current official-data track

Use README.md, docs/METHODOLOGY.md and artifacts_official/manifest.json for the current model.
Read docs/PROVENANCE.md before discussing competition eligibility. Preserve original research
records in docs/archive/ and Git history. Do not delete October evidence to make the record
appear compliant, and do not automatically consult any October/evaluation-period data.

The current model is cumulative PCHIP + 0.8 times three zero-sum LightGBM coefficients,
with history, cross-zone and **prev_day_edge: true**. No memory, graph or magnitude extension.
Official development MAE 3.9479; September backtest 5.0444; exported inference 5.0442.
Replica values 3.943/5.004/4.939 belong to an earlier model and dataset, not current artifacts.

Train using an explicitly supplied official TRAIN_CSV path; never fetch external training data.
Inference: python -m aet.infer --hourly test_hourly.csv --history history_15min.csv --model artifacts_official --out predictions.csv
Keep model artifacts unchanged unless explicitly asked to retrain. Prefer artifacts_retrained
for reproduction. No refitting at jury inference. Calendar lookups must read only earlier dates.
Valid hourly references are required for fallbacks; invalid hourly input must fail clearly.

Run python -m pytest -q after code changes. The test suite has grown beyond the old 45-test
snapshot; report the count actually returned rather than copying a stale count.
No automatic research loop, external lookup or October scoring is authorised by this file.
Optional explorer dependencies are in requirements-explorer.txt.
