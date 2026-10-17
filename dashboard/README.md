# Dashboard (optional)

Interactive replay of the reconstruction on the September 2026 hold-out: hourly input, PCHIP
baseline, learned slope/curvature/zigzag corrections, the final 15-minute curve and a heuristic
risk flag per hour. It is not part of the inference path.

## Required resources

- `data/train_prices_15min.csv` (see [data/README.md](../data/README.md)), or pass `-- --train <path>`
  or set `AET_TRAIN_CSV`.
- `artifacts_september_assessment/`: the final configuration trained **through 2026-08-31 only**, so
  September is genuinely held out. It is not committed; regenerate it once:

```bash
python -c "import pandas as pd; d = pd.read_csv('data/train_prices_15min.csv'); \
d[d.delivery_date <= '2026-08-31'].to_csv('data/train_through_august.csv', index=False)"
python -m aet.train --train data/train_through_august.csv --config configs/final.yaml --out artifacts_september_assessment
```

## Launch

From the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-dashboard.txt
PYTHONPATH=. python -m streamlit run dashboard/app.py --server.address localhost
```

Nothing is trained, tuned or written to disk by the dashboard.
