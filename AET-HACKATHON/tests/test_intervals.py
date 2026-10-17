import numpy as np
import pytest

from aet.intervals import interval_table, load_table, save_table


def test_interval_table_covers_the_requested_share():
    rng = np.random.default_rng(0)
    errors = rng.normal(0, 5, size=(2000, 4, 24, 4))
    lo, hi = interval_table(errors, 0.8)
    assert lo.shape == hi.shape == (4, 24, 4)
    assert (lo < 0).all() and (hi > 0).all()
    inside = ((errors >= lo) & (errors <= hi)).mean()
    assert inside == pytest.approx(0.8, abs=0.01)


def test_interval_table_rejects_bad_level():
    with pytest.raises(ValueError):
        interval_table(np.zeros((10, 1, 24, 4)), 1.2)


def test_table_round_trip(tmp_path):
    lo, hi = -np.ones((2, 24, 4)), np.ones((2, 24, 4))
    save_table(tmp_path / "t.json", ("A", "B"), lo, hi, 0.8, {"candidate": "final"})
    table = load_table(tmp_path / "t.json")
    assert table["level"] == 0.8 and table["zones"] == ["A", "B"]
    np.testing.assert_array_equal(table["lo"], lo)
    np.testing.assert_array_equal(table["hi"], hi)
