"""Focused checks for the challenge's exact constraint and calendar lag semantics."""
import io
import unittest

import numpy as np
import pandas as pd

from analysis.initial_analysis import centre_predictions, lag_diagnostics


class AnalysisChecks(unittest.TestCase):
    def test_constraint_survives_csv_roundtrip(self):
        hourly = np.array([100.0025, -499.9875, 0., 747.1])
        raw = np.random.default_rng(2026).normal(0, 80, (4, 4))
        predicted = centre_predictions(hourly, raw)
        restored = pd.read_csv(io.StringIO(pd.DataFrame(predicted).to_csv(index=False))).to_numpy()
        np.testing.assert_allclose(restored.mean(axis=1), hourly, atol=1e-10, rtol=0)
        np.testing.assert_allclose(centre_predictions(hourly, np.zeros((4, 4))), np.repeat(hourly[:, None], 4, axis=1))

    def test_nonfinite_predictions_rejected(self):
        with self.assertRaises(ValueError):
            centre_predictions(np.array([0.]), np.array([[0., np.nan, 1., 2.]]))

    def test_dst_gap_does_not_become_previous_day(self):
        # The preceding observed date is Oct 25, but Oct 27's yesterday is absent.
        dates = pd.to_datetime(["2025-10-25", "2025-10-27", "2025-10-28"])
        rows = []
        for date in dates:
            for quarter, value in enumerate([-3., -1., 1., 3.], 1):
                rows.append([date, "ZONE_0", 1, quarter, value, abs(value)])
        df = pd.DataFrame(rows, columns=["delivery_date", "market_id", "hour_index", "quarter_index", "deviation", "absolute_deviation"])
        with np.errstate(divide="ignore", invalid="ignore"):
            result = lag_diagnostics(df)
        daily = result[result.calendar_lag_days.eq(1)].iloc[0]
        self.assertAlmostEqual(daily.availability_pct, 100/3)
        self.assertEqual(daily.copy_shape_mae_when_available, 0)


if __name__ == "__main__":
    unittest.main()
