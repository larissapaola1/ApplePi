# autoresearch — autonomous improvement loop

Adapted from Karpathy's `autoresearch` (`program.md`): the human writes this file, a fixed harness
measures, and the agent edits the model, runs one experiment at a time, keeps what helps and
reverts what does not, without stopping. Start a session with:
"Read AUTORESEARCH.md and start an autoresearch session with tag `<tag>`".

Mapping to the original: `prepare.py` (fixed data and evaluation) → `tools/autoresearch.py` plus
`aet/backtest.py`; `train.py` (what the agent edits) → `configs/autoresearch.yaml` and the
modelling modules listed below; `val_bpb` → `search_score` (official metric, EUR/MWh, lower is
better).

## Goal
Lower `search_score` without breaking the challenge rules. Starting point (the frozen I2 model,
`configs/final.yaml`): **search 3.943182, confirm 3.680088**. What is already known is in
`docs/METHODOLOGY.md` sections 3–8; read them before choosing ideas.

## How a run is measured
`python -m tools.autoresearch` trains the `exp` candidate of `configs/autoresearch.yaml` on eight
monthly folds (training ends 31 Dec 2025 … 31 Jul 2026) and scores every day of the following month
with history replay, exactly as the jury runs inference.
- **Search set**: days 19..end of Jan–Aug 2026, the same days as the development folds. The
  promotion gate is applied here.
- **Confirm set**: days 1–18 of the same months. These are different dates, scored by the same fold models,
  and the gate never sees them. A kept change must not lose there. This is the guard against fitting the
  search days' noise, which matters when you run hundreds of experiments on the same folds.
- The comparison is paired, by day, against the stored per-day results of the incumbent (the
  last kept commit). Re-running the old configuration is not a valid reference, because your code
  change may alter it too.
- One run takes about 20 s, plus about 10 s for `pytest -q`.

## Hard rules (breaking any of them invalidates the whole session)
1. Never open, read or score `data/eval/**`, `results/october/**`, any `truth_15min.csv`, or any
   price dated after 2026-09-30. Never run `aet.backtest --include-locked`, `tools.score` or
   `tools.fetch_energy_charts --eval-date`. Those are post-freeze checks for the human only.
2. Causality: features for date D use only D's 24 hourly prices of every zone and quarter-hours
   strictly before D (`QuarterCube.before`, `HistoryStore`). In training, each row likewise only
   sees earlier dates. The sentinel tests must keep passing. Never weaken, skip or delete a test.
3. Anything fitted (scalers, scales, weights, thresholds) is fitted in `fit()` on the training slice
   and saved with the artifacts. Inference may only recompute fixed summaries over the supplied
   history: no refitting, recalibration or tuning at inference.
4. No external data, holiday calendars, pretrained weights or new dependencies
   (`requirements.txt` is frozen). Use numpy, pandas, scipy and LightGBM only.
5. Every kept run must be compliant: `max_mean_err` ≤ 1e-6 and `fallback_hours: 0`.
6. Budgets: a harness run over 10 minutes is a failure (`timeout 600`). For a kept run,
   `train_s` (eight fold trainings) stays under 300 s and `infer_s` (about 240 scored days) under
   60 s.
7. The harness, the backtest, the gate and the folds are fixed. Never edit `tools/`,
   `aet/backtest.py`, `aet/config.py`, `aet/schema.py`, `aet/cube.py`, `aet/train.py`,
   `aet/infer.py`, `configs/core.yaml` or any existing config other than `configs/autoresearch.yaml`.

## What you may edit
- `configs/autoresearch.yaml`: the `exp` candidate only (new keys are fine).
- `aet/features.py`, `aet/models.py`, `aet/baselines.py`, `aet/basis.py`, `aet/history.py`,
  `aet/pipeline.py`. In `pipeline.py`, leave `finalize` and `flat_predictions` alone, and make sure
  anything new that is fitted survives `save`/`load`.
- New tests in `tests/` for new code (encouraged when an idea adds logic).
- For analysis you may freely read `data/train_prices_15min.csv`, `results/eda/` and the per-day
  files in your state folder (`last_per_day.csv`, `incumbent_per_day.csv`). These are development
  data: use them to see where the error lives (zones, quarters, hours, days).

## Keep rule
Read the verdict lines that the harness prints:
- **keep**: `improves: yes`. The gate passed on the search set (at least 0.5 % better than the
  incumbent, better on at least 6/8 folds, no zone more than 2 % worse) and
  `confirm_vs_incumbent_pct` ≥ 0, with a compliant run.
- **keep-simpler**: the change removes code, features, models or parameters and `not_worse: yes`
  (both sets within −0.1 %). An equal result with less machinery is a win.
- **discard**: anything else.

Never relax these thresholds. Do not hunt for a pass by re-running one idea with tiny variations:
try at most three settings of a single idea (for example a 3-value grid), then move on. Small
gains that add a lot of complexity are not worth it: weigh the size of the diff against the gain.

## Setup (once per session)
1. Tag: use the one the human gives, or today's date (e.g. `oct18`). Create
   `git checkout -b autoresearch/<tag>` from `main`; the branch must be new.
2. Read: this file, `CLAUDE.md`, `README.md`, `docs/METHODOLOGY.md`, `aet/*.py`,
   `configs/core.yaml`, `configs/autoresearch.yaml`, `tools/autoresearch.py` (read-only).
3. Check `data/train_prices_15min.csv` exists (363 dates). If not:
   `.venv/bin/python -m tools.fetch_energy_charts --out data` (no `--eval-date`).
4. `.venv/bin/python -m pytest -q` must pass.
5. `STATE=results/autoresearch/<tag>; mkdir -p $STATE` then
   `.venv/bin/python -m tools.autoresearch --state $STATE --init > $STATE/run.log 2>&1`.
   Check that it prints `search_score: 3.943182` (unless the human changed the starting point).
6. Create `$STATE/results.tsv` with the header row below plus the baseline row, and
   `$STATE/notes.md` for the research log. Both live under `results/`, which git ignores.
7. If the human is present, confirm the setup in one line. Then start the loop.

## results.tsv
Tab-separated, one row per experiment, never committed:

```
commit	search_score	search_vs_pct	confirm_vs_pct	folds_better	gate	status	description
<commit>	3.943182	+0.000	+0.000	0/8	ref	keep	baseline: frozen I2
```

`status` is one of `keep`, `keep-simpler`, `discard` or `crash`. For a crash, write `0.000000` and
`nan` in the score columns. Keep `description` short and specific, e.g.
`slopes at ±3h for own zone`.

## The experiment loop
LOOP FOREVER:
1. `git log --oneline -1` is the incumbent. Choose one idea (backlog below, or your own, informed
   by notes.md and the per-day errors). Write its hypothesis as one line in notes.md.
2. Implement the smallest version that tests it. Commit it:
   `git add -A aet configs tests && git commit -m "exp: <description>"`.
3. Run `timeout 600 .venv/bin/python -m tools.autoresearch --state $STATE > $STATE/run.log 2>&1`.
4. Read the result with
   `grep -E "^(search_score|search_vs_incumbent_pct|folds_better|worst_zone_pct|gate|confirm_vs_incumbent_pct|max_mean_err|fallback_hours|train_s|infer_s|improves|not_worse):" $STATE/run.log`.
   If the output is empty, the run crashed: `tail -n 50 $STATE/run.log`. Fix an obvious bug, amend
   and re-run, up to three tries. Otherwise log `crash` and revert.
5. If the verdict is keep or keep-simpler: run `.venv/bin/python -m pytest -q`. If a test fails,
   fix the code (never the test) or discard. If the fix changed code, amend the commit and go back
   to step 3, because the stored result must come from the exact code you keep. When tests pass
   on an unchanged commit: `.venv/bin/python -m tools.autoresearch --state $STATE --promote`.
   The branch now advances.
6. Otherwise revert your experiment commit only: check `git log -1` shows it, then
   `git reset --hard HEAD~1`.
7. Append the row to results.tsv. Add one line to notes.md on what was learned, which is worth as
   much as the score.
8. Every 10 experiments: re-read notes.md and the tail of results.tsv, prune the backlog, and look
   at where the incumbent's error lives (`incumbent_per_day.csv` by zone, quarter, fold) before
   choosing the next direction.

NEVER STOP. Once the loop has begun, do not pause to ask whether to continue. The human may be
asleep and expects you to keep working until they interrupt you. If you run out of ideas, think
harder: re-read the methodology, inspect the worst days, combine near-misses (two changes that were
each +0.2–0.4 % on both sets), or try the opposite of something that failed. If your context is
compacted, re-read this file, notes.md and the tail of results.tsv, then continue.

## Idea backlog (not yet tried)
Model and target
- A separate scale per coefficient (slope / curvature / zigzag) instead of one 0.8, or per zone.
  Fit it on the fold's training slice (for example an inner time split) or declare it fixed:
  never choose it on the scored days.
- LightGBM capacity: `num_leaves` 7 or 31, `min_data_in_leaf` 50 or 200, `feature_fraction` 0.8,
  `lambda_l2`, learning rate 0.03 with 1000 trees.
- Objective for the coefficient models: `huber` or `fair` instead of `regression_l1`.
- Chained coefficients: curvature and zigzag models also receive the predicted slope coefficient.
- Average of 3–5 LightGBM seeds (deterministic; reduces variance).
- A convex blend of `lgbm_coef` and `lgbm_quarter` with one global weight shared by the four
  quarters (a blend of valid shapes stays valid).
- Predict the slope coefficient as a multiple of a floored |hourly slope| (amplitude ratio) rather
  than in EUR/MWh. Do not divide by H.
- Train only on recent dates (drop Oct–Nov 2025, see the drift in METHODOLOGY section 3) as a
  fixed rule of the fold's training slice.

Features (all from the same day's hourly prices or strictly earlier history)
- Slopes at ±3 h, local-extremum flags (the slope changes sign), |slope| rank within the day.
- Count of zones whose slope has the same sign as this zone; the max and min cross-zone spread.
- Residual coefficients of yesterday at hours h−1 and h+1; exponentially weighted residual means
  with a fixed half-life (3 and 14 days).
- Interactions: hour × weekend, hour × season.

## Already tried (see METHODOLOGY section 8; do not repeat without a new reason)
Pooled quarter L1 model (E2b/E3b), Haar basis, Akima or spline baseline under the learner,
recency weighting with a half-life of 90 or 180 days, 200 or 1000 trees, residual scale 0.7 or 0.9,
learning slope and curvature only, no cross-zone features, the previous-day edge (now off),
E4 analogue memory (K 8/16/32, 180 days or full history), E6 graph shrinkage (λ 0.1/0.3), E6
common-τ quantiles. The near-misses worth combining or revisiting: scale 0.9 (+0.21 % search,
+0.60 % confirm), 1000 trees (+0.28 %), E4 K=8 (+0.25 %), shrinkage λ 0.1 (+0.15 %).

## End of session (for the human, not the agent)
1. Review `results.tsv` and `notes.md`. The head of `autoresearch/<tag>` is the chain of kept
   changes. Report how many experiments were run, not only the winners.
2. To adopt the result: merge it, copy the `exp` settings into `configs/final.yaml`, retrain
   (`aet.train`) and rerun the development backtest. September was already opened once (5.004).
   A second look is reported as such and is never used to choose.
3. The clean out-of-sample test is the days after 17 Oct 2026 (19 Oct onwards) as they are
   published. Fetch, run `aet.infer` with the frozen artifacts and score both the new model and
   the current final on the same days.
