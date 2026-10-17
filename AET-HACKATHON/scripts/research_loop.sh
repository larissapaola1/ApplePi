#!/usr/bin/env bash
# Unattended research loop (RESEARCH.md): one fresh Claude Code session per cycle, state in files.
#
#   scripts/research_loop.sh <tag> [cycles=20] [cycle_timeout_s=3600]
#   touch results/research/<tag>/STOP      # finish after the current cycle
#
# Env: MODEL, EFFORT (claude --model/--effort), BUDGET_USD (per cycle, API-key billing only),
#      BACKOFF_S (wait after a failed cycle, default 1800), MAX_FAILS (consecutive, default 12).
# On a subscription, hitting the usage limit makes cycles fail fast: the loop waits BACKOFF_S and
# retries without spending a cycle, and gives up after MAX_FAILS failures in a row (~6 h).
# Run it from a dedicated worktree so your main checkout stays free:
#   git worktree add -b research/<tag> ../AET-research main && cd ../AET-research
#   mkdir -p data && cp ../AET-HACKATHON/data/train_prices_15min.csv data/
#   uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt
set -euo pipefail

TAG=${1:?usage: scripts/research_loop.sh <tag> [cycles] [cycle_timeout_s]}
CYCLES=${2:-20}
CYCLE_TIMEOUT=${3:-3600}
BACKOFF_S=${BACKOFF_S:-1800}
MAX_FAILS=${MAX_FAILS:-12}
BRANCH="research/$TAG"
STATE="results/research/$TAG"
cd "$(git rev-parse --show-toplevel)"

# Guards: post-freeze data must not be reachable, the tree must be clean, the data must exist.
for forbidden in data/eval results/october; do
  if [ -e "$forbidden" ]; then
    echo "move $forbidden outside the repository before an unattended run (hard rule 1)" >&2; exit 1
  fi
done
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "working tree not clean" >&2; exit 1; }
[ -f data/train_prices_15min.csv ] || { echo "missing data/train_prices_15min.csv" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo "missing .venv (see header)" >&2; exit 1; }

# Setup once: branch, harness incumbent, results.tsv with the baseline row.
if git show-ref --quiet "refs/heads/$BRANCH"; then git checkout -q "$BRANCH"; else git checkout -q -b "$BRANCH"; fi
mkdir -p "$STATE"
if [ ! -f "$STATE/incumbent_per_day.csv" ]; then
  .venv/bin/python -m tools.autoresearch --state "$STATE" --init > "$STATE/run.log" 2>&1
  get() { grep "^$1:" "$STATE/run.log" | cut -d' ' -f2; }
  printf 'commit\tsearch_score\tsearch_vs_pct\tconfirm_vs_pct\tfolds_better\tgate\tstatus\tdescription\n' > "$STATE/results.tsv"
  printf '%s\t%s\t+0.000\t+0.000\t0/8\tref\tkeep\tbaseline\n' "$(get commit)" "$(get search_score)" >> "$STATE/results.tsv"
fi

# Loop agents run without the RTK command rewriter: its rewrites (git -> rtk git, grep -> rtk grep)
# would no longer match the allowlist. A no-op `rtk` placed first on PATH, for these sessions only,
# makes the user's `rtk hook claude` hook return without rewriting; other hooks are unaffected.
SHIM="$PWD/$STATE/bin"
mkdir -p "$SHIM"
printf '#!/bin/sh\n[ "$1" = hook ] && cat >/dev/null\nexit 0\n' > "$SHIM/rtk"
chmod +x "$SHIM/rtk"

# Reads inside the worktree need no rule; nothing outside it is readable without a prompt,
# and --permission-prompts none denies every prompt.
ALLOWED="Edit,Write,WebSearch,WebFetch,\
Bash(.venv/bin/python *),Bash(timeout 600 .venv/bin/python *),\
Bash(git status*),Bash(git log*),Bash(git diff*),Bash(git show*),Bash(git add *),Bash(git commit *),\
Bash(git reset --hard HEAD~1),Bash(grep *),Bash(tail *),Bash(head *),Bash(ls*),Bash(wc *),Bash(touch *)"
DENIED="Bash(git push*),Bash(git checkout*),Bash(git switch*),Bash(git merge*),Bash(git rebase*),\
Bash(pip *),Bash(uv *),Bash(curl *),Bash(wget *),Bash(rm *),Bash(cat *)"

cycle_ok() {   # the session ended normally (JSON result, no error)
  .venv/bin/python -I -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if not d.get("is_error") and d.get("subtype") == "success" else 1)' "$1"
}

done_cycles=0
fails=0
while [ "$done_cycles" -lt "$CYCLES" ]; do
  [ -f "$STATE/STOP" ] && { echo "STOP file found"; break; }
  n=$((done_cycles + 1))
  log="$STATE/cycle_$(date +%Y%m%d_%H%M%S).json"
  echo "$(date '+%F %T') cycle $n/$CYCLES -> $log"
  prompt="Read RESEARCH.md and AUTORESEARCH.md, then run exactly one research cycle for tag $TAG \
(branch $BRANCH, state $STATE). Work autonomously: nobody will answer questions. End the session \
when the cycle is recorded and committed."
  extra=()
  [ -n "${MODEL:-}" ] && extra+=(--model "$MODEL")
  [ -n "${EFFORT:-}" ] && extra+=(--effort "$EFFORT")
  [ -n "${BUDGET_USD:-}" ] && extra+=(--max-budget-usd "$BUDGET_USD")
  started=$(date +%s)
  rc=0
  PATH="$SHIM:$PATH" timeout "$CYCLE_TIMEOUT" claude -p "$prompt" \
    --permission-mode acceptEdits --permission-prompts none \
    --allowedTools "$ALLOWED" --disallowedTools "$DENIED" \
    --output-format json --name "research-$TAG-$n" ${extra[@]+"${extra[@]}"} > "$log" 2> "${log%.json}.err" \
    || rc=$?
  elapsed=$(( $(date +%s) - started ))
  # A cycle that dies mid-experiment leaves an uncommitted change: drop it (the next cycle's orient
  # step reverts an unrecorded exp commit).
  if [ -n "$(git status --porcelain -- aet configs tests research)" ]; then
    echo "discarding uncommitted changes left by cycle $n"
    git checkout -q -- . && git clean -fdq -- aet configs tests research
  fi
  git checkout -q "$BRANCH"
  if [ "$rc" -eq 124 ] || cycle_ok "$log"; then
    done_cycles=$n; fails=0
    echo "$(date '+%F %T') cycle $n finished (rc=$rc, ${elapsed}s)"
  else
    fails=$((fails + 1))
    echo "$(date '+%F %T') cycle $n failed (rc=$rc, ${elapsed}s, $fails in a row; usage limit?) - see ${log%.json}.err"
    [ "$fails" -ge "$MAX_FAILS" ] && { echo "giving up after $fails failures in a row"; break; }
    sleep "$BACKOFF_S"
  fi
done
echo "done: $done_cycles cycles, $(($(wc -l < "$STATE/results.tsv") - 1)) experiment rows in $STATE/results.tsv"
