"""The autoresearch harness: search/confirm split and the paired keep rule."""
import pandas as pd

from aet.config import DEFAULT_GATE
from tools.autoresearch import compare, split, verdict


def _per_day(scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for month in range(1, 9):
        for day in (5, 12, 20, 27):
            for z in range(4):
                mae = (1.0 + 0.1 * z + 0.01 * day) * scale
                rows.append({"fold": f"m{month}", "delivery_date": f"2026-{month:02d}-{day:02d}",
                             "zone": f"ZONE_{z}", "mae": mae, "flat_mae": 2.0,
                             **{f"q{q}_mae": mae for q in range(1, 5)},
                             "max_mean_error": 0.0, "fallback_hours": 0})
    return pd.DataFrame(rows)


def test_split_puts_days_from_19_in_the_search_set():
    search, confirm = split(_per_day(), 19)
    assert set(pd.to_datetime(search["delivery_date"]).dt.day) == {20, 27}
    assert set(pd.to_datetime(confirm["delivery_date"]).dt.day) == {5, 12}


def test_identical_runs_are_neither_kept_nor_rejected_as_worse():
    base = _per_day()
    row = compare(base, base, DEFAULT_GATE)
    assert row["vs_ref_pct"] == 0 and row["gate"] == "-"
    assert verdict(row, row) == {"improves": False, "not_worse": True}


def test_keep_needs_the_gate_on_search_and_no_loss_on_confirm():
    base, better, worse = _per_day(), _per_day(0.99), _per_day(1.01)
    s_base, c_base = split(base, 19)
    s_better, _ = split(better, 19)
    _, c_worse = split(worse, 19)
    search = compare(s_better, s_base, DEFAULT_GATE)
    assert search["gate"] == "PASS"
    assert verdict(search, compare(split(better, 19)[1], c_base, DEFAULT_GATE))["improves"]
    assert not verdict(search, compare(c_worse, c_base, DEFAULT_GATE))["improves"]   # confirm set loses
