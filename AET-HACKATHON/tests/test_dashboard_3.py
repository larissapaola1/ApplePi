"""The explanatory preview must preserve signed arithmetic and distinguish scoring from prediction."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from aet import dashboard_3 as dashboard


def example_hour():
    rows = pd.DataFrame({
        "delivery_date": ["2026-09-19"] * 4, "market_id": ["ZONE_0"] * 4,
        "hour_index": [1] * 4, "quarter_index": [1, 2, 3, 4], "hourly": [100.] * 4,
        "smooth": [5., -5., 2., -2.], "slope": [-3., 3., -1., 1.],
        "curvature": [2., -2., 1., -1.], "zigzag": [-1., 1., -2., 2.],
        "actual": [108., 90., 102., 100.], "chance": [.7, .6, .8, .9],
    })
    rows["model"] = rows[["hourly", "smooth", "slope", "curvature", "zigzag"]].sum(axis=1)
    rows["baseline"] = rows.hourly + rows.smooth
    rows["range_lo"], rows["range_hi"] = rows.model - 8, rows.model + 12
    rows["mid_lo"], rows["mid_hi"] = rows.model - 2, rows.model + 3
    rows["timestamp"] = pd.date_range("2026-09-19", periods=4, freq="15min")
    hours = pd.DataFrame({"delivery_date": ["2026-09-19"], "market_id": ["ZONE_0"],
                          "hour_index": [1], "risk_label": ["low"], "dominant_signal": ["daily_std"]})
    return rows, hours


def test_within_five_includes_both_boundaries_and_excludes_missing_prices():
    rows = pd.DataFrame({"model": [100., 100., 100., 100., 100., np.nan],
                         "actual": [95., 105., 94.99, 105.01, np.nan, 100.]})
    assert dashboard.tolerance_counts(rows) == (2, 4)


def test_observed_band_coverage_uses_individual_asymmetric_limits():
    rows = pd.DataFrame({"actual": [92., 112., 113., np.nan],
                         "lo": [92., 92., 92., 92.], "hi": [112., 112., 112., 112.]})
    assert dashboard.share_inside(rows, "lo", "hi") == pytest.approx(200 / 3)


@pytest.mark.parametrize("hourly,parts", [(100., [5., -3., 2., -1.]), (-10., [5., -20., 3., -2.]),
                                        (-2., [5., -1., 0., 0.])])
def test_single_bar_stages_preserve_signed_contributions_and_negative_prices(hourly, parts):
    row = pd.Series(dict(zip(["hourly", "smooth", "slope", "curvature", "zigzag"], [hourly, *parts])))
    row["model"], row["actual"], row["quarter_index"] = hourly + sum(parts), 999., 1
    steps = dashboard.construction_steps(row)
    for stage, expected in enumerate(np.cumsum([hourly, *parts])):
        fig = dashboard.single_bar_chart(row, dashboard.PALETTES["light"], stage)
        assert len([trace for trace in fig.data if trace.type == "bar"]) == 1
        assert fig.data[0].base == (0,)
        assert fig.data[0].y[0] == pytest.approx(expected)
        json.loads(fig.to_json())  # all stages can be serialized for Streamlit
    assert steps.iloc[-1]["Running price"] == pytest.approx(row.model)
    # Evaluation truth is a reference marker, never part of the prediction arithmetic.
    row["actual"] = -999.
    pd.testing.assert_frame_equal(steps, dashboard.construction_steps(row))


def test_construction_detects_a_misleading_component_total():
    rows, _ = example_hour()
    rows.loc[0, "model"] += 1
    with pytest.raises(ValueError, match="do not reconstruct"):
        dashboard.construction_steps(rows.iloc[0])


def test_interactive_construction_updates_the_selected_quarter_and_stage():
    rows, hours = example_hour()
    script = """
import streamlit as st
from aet import dashboard_3 as d
d.one_prediction(st.session_state.rows, st.session_state.hours, d.PALETTES['light'])
"""
    at = AppTest.from_string(script, default_timeout=20)
    at.session_state["rows"], at.session_state["hours"] = rows, hours
    at.run()
    assert not at.exception
    assert at.metric[0].value == "3 of 4"
    assert at.dataframe[0].value.iloc[-1]["Running price"] == 103
    # Streamlit 1.45's AppTest treats single-select button groups as multi-select when
    # serializing reruns. Seed their one-element selection lists in this test harness.
    at.button_group(key="explain_zone").set_value(["ZONE_0"])
    at.button_group(key="explain_quarter").set_value([2])
    at.select_slider(key="build_stage").set_value(2).run()
    assert not at.exception
    assert at.dataframe[0].value.iloc[-1]["Running price"] == 98
    assert at.dataframe[0].value.iloc[-1]["Step"] == "Slope"


def test_uncertainty_info_shows_historical_estimate_separately_from_observed_count():
    rows, _ = example_hour()
    at = AppTest.from_string("""
import streamlit as st
from aet import dashboard_3 as d
d.uncertainty_cards(st.session_state.rows)
""")
    at.session_state["rows"] = rows
    at.run()
    assert not at.exception
    assert [(m.label, m.value) for m in at.metric] == [
        ("Average 90% interval width", "20.0 €/MWh"),
        ("Within ±5: historical", "75%"), ("Within ±5: observed", "75%")]
    help_text = "\n".join(m.value for m in at.markdown)
    assert "5th and 95th percentiles" in help_text
    assert "3 of an hour's 4 prices" in help_text


def test_standalone_preview_renders_without_the_original_navigation(monkeypatch):
    rows, hours = example_hour()
    monkeypatch.setattr(dashboard, "train_csv_path", lambda: Path("fixture.csv"))
    # Avoid model replay here: exercise the complete UI, including chart/table controls, on known outputs.
    monkeypatch.setattr(dashboard, "load_assessment", lambda path: (rows, hours))
    at = AppTest.from_string("from aet import dashboard_3 as d\nd.main()", default_timeout=20)
    at.session_state["day"] = pd.Timestamp("2026-09-19").date()
    at.run()
    assert not at.exception
    assert any(s.value == "One prediction, step by step" for s in at.subheader)
    assert len(at.get("plotly_chart")) == 3


def test_clicking_an_interval_does_not_treat_prices_as_a_date():
    at = AppTest.from_string("""
import streamlit as st
from types import SimpleNamespace
from aet import dashboard_3 as d
event = SimpleNamespace(selection=SimpleNamespace(points=[{'customdata': [92., 112.]}]))
assert not d.consume_click(event, 'ZONE_0')
""").run()
    assert not at.exception
