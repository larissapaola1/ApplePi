# Data provenance and eligibility: unresolved question

The repository contains two distinct development histories. This note records them; it does
not certify competition eligibility or erase earlier exposure.

## Earlier public-replica research

The original documentation describes a nonparticipant public-replica investigation using
Energy-Charts rather than the official package. Its methodology includes 1–17 October 2026
price results and uses `prev_day_edge: false`. That record existed before the subsequent
official-data development and configuration change to `true`, according to the audit supplied
by the user. Whether those observations influenced the later work is not established here.

The brief prohibits consulting or using 1–18 October prices before the submission freeze,
including indirect use and public sources. It also restricts training, calibration and
hyperparameter selection to the official package. Offline execution, a September training cutoff,
and official-data retraining do not by themselves resolve prior external-data exposure or selection.
An internal replica freeze is not necessarily the organisers' submission freeze.

The original records are preserved without rewriting their claims in:
- `docs/archive/PUBLIC_REPLICA_METHODOLOGY.md` (including original section 7.7).
- `docs/archive/ORIGINAL_README.md` and `ORIGINAL_CLAUDE.md`.
- `docs/archive/replica_111b489_*.yaml`, retrieved from the historical freeze commit.
- Git history and the original tools, retained in the working repository.

## Current official-data track

The current `artifacts_official/manifest.json` identifies the official training file by hash,
ending 2026-09-30; `artifacts_september_assessment/` ends 2026-08-31. Both use previous-day
boundary features. Official development comparisons support keeping them, but those results
cannot retroactively establish that all methodological choices were independent of the replica.
The authoritative current scores are documented in METHODOLOGY.md.

## Packaging and disclosure

The local packaging tool uses an allowlist and omits downloader, October data, archived research,
explorer and diagnostic tooling. It retains this disclosure and the actual manifest. That makes
the runtime bundle focused; it does NOT undo exposure, remove the need for disclosure, or assert
eligibility. Preserve the complete repository for an organiser audit. Do not delete historical
evidence or describe replica scores as official-data scores.

Before an actual submission, disclose the sequence, external-replica work and possible inherited
choices to the organisers and ask for their interpretation. No organiser decision is recorded here.
