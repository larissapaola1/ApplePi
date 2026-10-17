# Project report

Three-page report (`report.pdf`) built from `main.tex`. It describes the current final model (I3)
and uses the numbers in [results/summary.csv](../results/summary.csv) (September: exported-inference
scores) and the I3 risk-group replay in [docs/experiments/VOLATILITY.md](../docs/experiments/VOLATILITY.md).

Compile from this directory with `pdflatex main.tex` (twice if references change). `images/` must
stay alongside `main.tex`:

- `representative_reconstruction.png`: copy of `analysis/figures/september_report/representative_reconstruction.png`
  (I3 model trained through August; representative day 2026-09-20).
- `dashboard_detail.png`: dashboard capture for ZONE_0, 2026-09-19, 18:00–19:00.
