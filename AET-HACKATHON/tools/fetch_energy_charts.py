"""Build a public replica of the AET challenge files from Energy-Charts (Fraunhofer ISE).

NOT part of a challenge submission: the brief forbids external data for participants.
This only exists to exercise the pipeline when the official package is unavailable.

Source: https://api.energy-charts.info  endpoint /price, no API key. Day-ahead prices
in EUR/MWh. For the default zones the API states "CC BY 4.0 from Bundesnetzagentur |
SMARD.de"; other zones are for private use only (see the API docs before changing --zones).
Timestamps are taken as the START of each delivery interval.

    python -m tools.fetch_energy_charts --out data
    python -m tools.fetch_energy_charts --out data --eval-date 2026-10-16 --eval-date 2026-10-19

Writes, in the challenge schema and with anonymised zone ids:
    data/train_prices_15min.csv          train window (default 1 Oct 2025 - 30 Sep 2026), DST days excluded
    data/eval/<date>/test_hourly.csv     hourly means of the evaluated day (what the jury hands over)
    data/eval/<date>/history_15min.csv   every quarter-hour strictly before that day
    data/eval/<date>/truth_15min.csv     the hidden quarter-hours, for tools.score
    data/zones.json                      ZONE_i <-> bidding zone, source and licence
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://api.energy-charts.info/price"
MARKET_TZ = "Europe/Brussels"                       # CET/CEST market time, as in the brief
DEFAULT_ZONES = ("DE-LU", "FR", "AT", "IT-North")
QH_KEYS = ["delivery_date", "market_id", "hour_index", "quarter_index"]
QH_COLUMNS = QH_KEYS + ["price_eur_mwh"]


def month_chunks(first_day, last_day):
    """Calendar-month pieces covering [first_day, last_day] (one API call each)."""
    cur, last = pd.Timestamp(first_day), pd.Timestamp(last_day)
    while cur <= last:
        end = min(cur + pd.offsets.MonthEnd(0), last)
        yield cur, end
        cur = end + pd.Timedelta(days=1)


def utc_bounds(first_day, last_day) -> tuple[str, str]:
    """Market-time [first_day 00:00, last_day + 1 day 00:00) as UTC strings for the API."""
    lo = pd.Timestamp(first_day).tz_localize(MARKET_TZ).tz_convert("UTC")
    hi = (pd.Timestamp(last_day) + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
    return lo.strftime("%Y-%m-%dT%H:%MZ"), hi.strftime("%Y-%m-%dT%H:%MZ")


def http_get_json(url: str, retries: int = 5, pause: float = 2.0):
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "aet-neural-wave-replica/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None                                      # no data for that zone/range
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(pause * 2 ** attempt)
    return None


def fetch_zone(bzn: str, first_day, last_day, cache_dir: Path, fetch=http_get_json, delay: float = 1.0) -> pd.DataFrame:
    """Raw (unix_seconds, price) rows; finished months are cached, recent ones always refetched."""
    stable_before = pd.Timestamp.now(tz=MARKET_TZ).normalize().tz_localize(None) - pd.Timedelta(days=2)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    frames = []
    for lo, hi in month_chunks(first_day, last_day):
        cache = Path(cache_dir) / f"{bzn}_{lo:%Y%m%d}_{hi:%Y%m%d}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            start, end = utc_bounds(lo, hi)
            payload = fetch(f"{API}?{urllib.parse.urlencode({'bzn': bzn, 'start': start, 'end': end})}")
            if payload and hi < stable_before:
                cache.write_text(json.dumps(payload), encoding="utf-8")
            if delay:
                time.sleep(delay)
        if payload and payload.get("unix_seconds"):
            frames.append(pd.DataFrame({"unix_seconds": payload["unix_seconds"], "price": payload["price"]}))
    if not frames:
        return pd.DataFrame({"unix_seconds": pd.Series(dtype="int64"), "price": pd.Series(dtype=float)})
    return pd.concat(frames, ignore_index=True)


def to_quarter_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Interval-start timestamps -> challenge rows, keeping only complete 96-quarter market days."""
    raw = raw.dropna(subset=["unix_seconds"]).drop_duplicates("unix_seconds", keep="last")
    ts = pd.to_datetime(raw["unix_seconds"].astype("int64"), unit="s", utc=True).dt.tz_convert(MARKET_TZ)
    rows = pd.DataFrame({
        "delivery_date": ts.dt.strftime("%Y-%m-%d").to_numpy(),
        "hour_index": (ts.dt.hour + 1).to_numpy(),
        "minute": ts.dt.minute.to_numpy(),
        "price_eur_mwh": pd.to_numeric(raw["price"], errors="coerce").to_numpy(dtype=float),
    })
    rows["quarter_index"] = rows["minute"] // 15 + 1
    if rows.empty:
        return rows[["delivery_date", "hour_index", "quarter_index", "price_eur_mwh"]], \
            {"complete_days": 0, "dst_days": [], "hourly_days": 0, "incomplete_days": []}

    days = pd.Index(sorted(rows["delivery_date"].unique()))
    midnight = pd.to_datetime(days)
    quarters_in_day = ((midnight + pd.Timedelta(days=1)).tz_localize(MARKET_TZ)
                       - midnight.tz_localize(MARKET_TZ)) / pd.Timedelta(minutes=15)
    dst = set(days[np.asarray(quarters_in_day) != 96])            # 92 or 100 quarters: excluded

    per_day = rows.groupby("delivery_date").agg(
        n=("price_eur_mwh", "size"),
        finite=("price_eur_mwh", lambda s: bool(np.isfinite(s).all())),
        on_grid=("minute", lambda m: bool((m % 15 == 0).all())),
        hourly=("minute", lambda m: bool((m == 0).all())),
    )
    good = per_day.index[(per_day["n"] == 96) & per_day["finite"] & per_day["on_grid"] & ~per_day.index.isin(dst)]
    hourly = per_day.index[per_day["hourly"] & ~per_day.index.isin(dst)]
    incomplete = per_day.index.difference(good).difference(hourly).difference(pd.Index(sorted(dst)))
    report = {"complete_days": int(len(good)), "dst_days": sorted(dst), "hourly_days": int(len(hourly)),
              "incomplete_days": list(incomplete)}
    keep = rows[rows["delivery_date"].isin(good)]
    return keep[["delivery_date", "hour_index", "quarter_index", "price_eur_mwh"]].reset_index(drop=True), report


def build_dataset(zones, first_day, last_day, cache_dir: Path, fetch=http_get_json, delay: float = 1.0):
    """Long frame with the real zone name in `market`, restricted to days complete in every zone."""
    first, last = f"{pd.Timestamp(first_day):%Y-%m-%d}", f"{pd.Timestamp(last_day):%Y-%m-%d}"
    frames, report = [], {}
    for zone in zones:
        rows, report[zone] = to_quarter_rows(fetch_zone(zone, first_day, last_day, cache_dir, fetch, delay))
        rows = rows[(rows["delivery_date"] >= first) & (rows["delivery_date"] <= last)]
        frames.append(rows.assign(market=zone))
    empty = [zone for zone in zones if report[zone]["complete_days"] == 0]
    if empty:
        raise SystemExit(f"no complete 15-minute days for {empty}: check the zone codes or drop them "
                         f"(report: { {z: report[z] for z in empty} })")
    data = pd.concat(frames, ignore_index=True)
    zones_per_day = data.groupby("delivery_date")["market"].nunique()
    data = data[data["delivery_date"].isin(zones_per_day.index[zones_per_day == len(zones)])]
    return data.reset_index(drop=True), report


def boundary_check(data: pd.DataFrame) -> pd.Series:
    """Mean |price step| into each quarter position. With correct interval-start alignment the
    hourly-block structure usually makes the step into q1 (across the hour boundary) the largest."""
    d = data.sort_values(["market", "delivery_date", "hour_index", "quarter_index"])
    step = d.groupby(["market", "delivery_date"])["price_eur_mwh"].diff().abs()
    return step.groupby(d["quarter_index"]).mean().rename("mean_abs_step_into_quarter")


def write_outputs(data: pd.DataFrame, zones, out: Path, train_start: str, train_end: str, eval_dates) -> dict:
    mapping = {zone: f"ZONE_{i}" for i, zone in enumerate(zones)}
    data = data.assign(market_id=data["market"].map(mapping)).sort_values(QH_KEYS)
    out.mkdir(parents=True, exist_ok=True)
    train = data[(data["delivery_date"] >= train_start) & (data["delivery_date"] <= train_end)]
    train[QH_COLUMNS].to_csv(out / "train_prices_15min.csv", index=False)
    written = []
    for day in eval_dates:
        target = data[data["delivery_date"] == day]
        if target.empty:
            print(f"WARNING no complete data for {day} (day D is auctioned on D-1 around 12:45 CET)")
            continue
        folder = out / "eval" / day
        folder.mkdir(parents=True, exist_ok=True)
        data[data["delivery_date"] < day][QH_COLUMNS].to_csv(folder / "history_15min.csv", index=False)
        target[QH_COLUMNS].to_csv(folder / "truth_15min.csv", index=False)
        (target.groupby(["delivery_date", "market_id", "hour_index"], as_index=False)["price_eur_mwh"].mean()
               .rename(columns={"price_eur_mwh": "hourly_price_eur_mwh"})
               .to_csv(folder / "test_hourly.csv", index=False))
        written.append(day)
    meta = {
        "zones": mapping, "market_time": MARKET_TZ,
        "source": "Energy-Charts.info (Fraunhofer ISE), /price endpoint",
        "licence": "CC BY 4.0 from Bundesnetzagentur | SMARD.de (as stated by the API for these zones)",
        "train_window": [train_start, train_end], "train_dates": int(train["delivery_date"].nunique()),
        "eval_dates": written, "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "zones.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zones", nargs="+", default=list(DEFAULT_ZONES),
                        help="Energy-Charts bidding zones (the configs expect four)")
    parser.add_argument("--train-start", default="2025-10-01")
    parser.add_argument("--train-end", default="2026-09-30")
    parser.add_argument("--eval-date", action="append", default=[], help="YYYY-MM-DD, repeatable")
    parser.add_argument("--out", default="data")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between API calls (be polite)")
    args = parser.parse_args(argv)
    if len(args.zones) != 4:
        print("NOTE the configs list four zones; edit `zones:` in configs/core.yaml to match", args.zones)

    out = Path(args.out)
    cache = out / "raw" / "energy_charts"
    last = max([args.train_end] + args.eval_date)
    data, report = build_dataset(args.zones, args.train_start, last, cache, delay=args.delay)
    for zone, rep in report.items():
        print(f"{zone:>10}: {rep['complete_days']} complete days, DST excluded {rep['dst_days']}, "
              f"hourly-only {rep['hourly_days']}, incomplete {len(rep['incomplete_days'])}")
    # Training window only: 1-18 Oct 2026 must not be consulted before the freeze.
    steps = boundary_check(data[(data["delivery_date"] >= args.train_start) & (data["delivery_date"] <= args.train_end)])
    print("\nAlignment check (mean |step| into each quarter; q1 = across the hour boundary):")
    print(steps.round(3).to_string())
    if steps.idxmax() != 1:
        print("NOTE the largest step is not at the hour boundary: double-check the timestamp convention")
    meta = write_outputs(data, args.zones, out, args.train_start, args.train_end, args.eval_date)
    print(f"\n{meta['train_dates']} training dates -> {out / 'train_prices_15min.csv'}; mapping {meta['zones']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
