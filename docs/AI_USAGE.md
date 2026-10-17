# Use of AI coding assistants

> TODO (team): confirm each line below and complete the details before submission.

No pretrained model, external weights or embeddings are part of the solution. The model is
PCHIP interpolation plus three LightGBM regressors trained from scratch on the official data.

AI assistants were used as development tools only:

| Tool | Used for | Human review |
|---|---|---|
| Claude Code | Code drafting and refactoring, tests, documentation, repository restructuring | TODO |
| OpenAI Codex | Feature-engineering review notes | TODO |
| Agent-driven experiment loop | Automated runs of the fixed backtest harness on the official training data to compare candidate configurations | TODO |

All experiment results reported in [results/summary.csv](../results/summary.csv) come from the
code in this repository and can be regenerated with the commands in
[results/README.md](../results/README.md). See also [PROVENANCE.md](PROVENANCE.md).
