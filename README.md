# ApplePi
Hackathon

## Initial data analysis

The analysis uses only `Data/train_prices_15min.csv`. It validates the complete observation
grid, then excludes September 2026 from price statistics and plots to preserve a final holdout.
No models are trained and no external data are accessed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python analysis/initial_analysis.py
python -m unittest analysis.test_initial_analysis
```

Read `analysis/results/initial_analysis.md` for the generated report, with six figures and
CSV tables. Open `notebooks/01_initial_analysis.ipynb` for a guided walkthrough using the same
functions (requires a Jupyter frontend/kernel, e.g. VS Code with notebook support).

The script runs from any working directory and accepts `--input`, `--output-dir`, and
`--eda-end` (inclusive; must precede September). To restrict exploration to initial training:

```bash
python analysis/initial_analysis.py --eda-end 2026-06-30 --output-dir analysis/results_training_only
```

Planned modelling: historical templates, constrained regression/CatBoost, July–August
development, and one untouched September final assessment. Exact calendar-date lag joins
handle the excluded DST dates. Reported flat-baseline MAEs are descriptive, not model-test scores.
