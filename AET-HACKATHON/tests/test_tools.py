"""Offline tests for the public-replica tools (the real API is never called here)."""
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from aet.schema import check_training_window, read_hourly, read_quarter_hourly
from tools import fetch_energy_charts as ec
from tools.score import score

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _payload(start_utc, end_utc, minutes=15):
    idx = pd.date_range(pd.Timestamp(start_utc), pd.Timestamp(end_utc), freq=f"{minutes}min", inclusive="left")
    local = idx.tz_convert(ec.MARKET_TZ)
    hour = local.hour + local.minute / 60
    price = 80 + 30 * np.sin(2 * np.pi * (hour - 8) / 24) + 4 * ((local.minute // 15) - 1.5)   # sawtooth
    return {"unix_seconds": ((idx - EPOCH) // pd.Timedelta(seconds=1)).tolist(), "price": price.round(2).tolist(),
            "unit": "EUR / MWh", "license_info": "CC BY 4.0", "deprecated": False}


def _fake_fetch(hourly_zones=()):
    def fetch(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        minutes = 60 if q["bzn"][0] in hourly_zones else 15
        return _payload(q["start"][0], q["end"][0], minutes)
    return fetch


def test_bounds_and_chunks_follow_market_time():
    assert ec.utc_bounds("2025-10-01", "2025-10-31") == ("2025-09-30T22:00Z", "2025-10-31T23:00Z")
    chunks = [(a.date().isoformat(), b.date().isoformat()) for a, b in ec.month_chunks("2025-10-15", "2025-12-02")]
    assert chunks == [("2025-10-15", "2025-10-31"), ("2025-11-01", "2025-11-30"), ("2025-12-01", "2025-12-02")]


def test_dst_and_hourly_days_are_excluded():
    start, end = ec.utc_bounds("2025-10-24", "2025-10-27")
    raw = pd.DataFrame(_payload(start, end))[["unix_seconds", "price"]]
    rows, report = ec.to_quarter_rows(raw)
    assert sorted(rows["delivery_date"].unique()) == ["2025-10-24", "2025-10-25", "2025-10-27"]
    assert report["dst_days"] == ["2025-10-26"] and report["complete_days"] == 3
    assert (rows.groupby("delivery_date").size() == 96).all()
    hourly, report = ec.to_quarter_rows(pd.DataFrame(_payload(start, end, minutes=60))[["unix_seconds", "price"]])
    assert hourly.empty and report["hourly_days"] == 3


def test_build_writes_challenge_compatible_files(tmp_path):
    zones = ["AA", "BB", "CC", "DD"]
    data, report = ec.build_dataset(zones, "2026-09-20", "2026-10-03", tmp_path / "cache", _fake_fetch(), delay=0)
    meta = ec.write_outputs(data, zones, tmp_path / "data", "2026-09-20", "2026-09-30", ["2026-10-02"])
    train = read_quarter_hourly(tmp_path / "data" / "train_prices_15min.csv", strict=True)
    check_training_window(train)
    assert train["delivery_date"].nunique() == 11 and set(train["market_id"]) == {f"ZONE_{i}" for i in range(4)}
    folder = tmp_path / "data" / "eval" / "2026-10-02"
    hist = read_quarter_hourly(folder / "history_15min.csv", strict=False)
    assert hist["delivery_date"].max() < pd.Timestamp("2026-10-02")
    hourly = read_hourly(folder / "test_hourly.csv")
    truth = pd.read_csv(folder / "truth_15min.csv")
    assert len(hourly) == 4 * 24 and len(truth) == 4 * 96
    assert meta["zones"]["AA"] == "ZONE_0" and json.loads((tmp_path / "data" / "zones.json").read_text())["eval_dates"]
    assert ec.boundary_check(data).idxmax() == 1      # the fake sawtooth jumps at hour boundaries


def test_build_rejects_zones_without_quarter_hours(tmp_path):
    with pytest.raises(SystemExit):
        ec.build_dataset(["AA", "HH"], "2026-09-20", "2026-09-22", tmp_path, _fake_fetch(hourly_zones=("HH",)), delay=0)


def test_score_matches_the_official_metric(tmp_path):
    zones = ["AA", "BB", "CC", "DD"]
    data, _ = ec.build_dataset(zones, "2026-09-29", "2026-10-02", tmp_path / "cache", _fake_fetch(), delay=0)
    ec.write_outputs(data, zones, tmp_path / "data", "2026-09-29", "2026-09-30", ["2026-10-02"])
    truth = pd.read_csv(tmp_path / "data" / "eval" / "2026-10-02" / "truth_15min.csv")
    perfect = truth.rename(columns={"price_eur_mwh": "predicted_price_eur_mwh"})
    _, _, overall = score(perfect, truth)
    assert overall["score"] == pytest.approx(0.0, abs=1e-12) and overall["flat"] > 0
    flat = perfect.assign(predicted_price_eur_mwh=truth.groupby(
        ["market_id", "hour_index"])["price_eur_mwh"].transform("mean"))
    _, _, overall = score(flat, truth)
    assert overall["score"] == pytest.approx(overall["flat"]) and overall["gain_pct"] == pytest.approx(0.0, abs=1e-9)
