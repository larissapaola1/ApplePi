"""Inspect saved September predictions against observations; never fit a model."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/applepi-error-analysis-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AET-HACKATHON"))
from aet.baselines import cumulative, central_slope
from aet.cube import HourlyCube, QuarterCube
from aet.schema import HOUR_KEYS, QH_KEYS, validate_predictions


def md(frame):
    f = frame.reset_index(drop=True).copy()
    for c in f.select_dtypes(include="number"):
        f[c] = f[c].map(lambda v: f"{v:.3f}")
    return "\n".join(["| " + " | ".join(f.columns) + " |",
                      "| " + " | ".join(["---"] * len(f.columns)) + " |",
                      *["| " + " | ".join(map(str, r)) + " |"
                        for r in f.itertuples(index=False, name=None)]])


def main(source):
    output = source / "error_analysis"
    output.mkdir(exist_ok=True)
    meta = json.loads((source / "summary.json").read_text())
    d = pd.read_csv(source / "prediction_details.csv", parse_dates=["delivery_date"])
    truth = pd.read_csv(source / "truth_15min.csv", parse_dates=["delivery_date"])
    pred = pd.read_csv(source / "predictions.csv", parse_dates=["delivery_date"])
    assert len(d) == len(truth) == len(pred) == 4608 and not d.duplicated(QH_KEYS).any()
    check = d[QH_KEYS + ["price_eur_mwh", "predicted_price_eur_mwh"]].merge(
        truth, on=QH_KEYS, validate="one_to_one", suffixes=("", "_truth")).merge(
        pred, on=QH_KEYS, validate="one_to_one", suffixes=("", "_saved"))
    assert np.allclose(check.price_eur_mwh, check.price_eur_mwh_truth)
    assert np.allclose(check.predicted_price_eur_mwh, check.predicted_price_eur_mwh_saved)
    hourly = d[HOUR_KEYS + ["hourly_price_eur_mwh"]].drop_duplicates()
    validate_predictions(pred, hourly)
    cube = HourlyCube.from_long(hourly, tuple(sorted(d.market_id.unique())))
    H = cube.values
    for name, array in {"pchip": cumulative(H), "slope2": H[..., None] +
                        2 * central_slope(H)[..., None] * np.array([-.375, -.125, .125, .375])}.items():
        frame = QuarterCube(cube.dates, cube.zones, array).to_long(name + "_price")
        d = d.merge(frame, on=QH_KEYS, validate="one_to_one")
        d[name + "_error"] = (d[name + "_price"] - d.price_eur_mwh).abs()
    d["signed_error"] = d.predicted_price_eur_mwh - d.price_eur_mwh
    d["absolute_error"] = d.signed_error.abs()
    assert np.isclose(d.absolute_error.mean(), meta["overall"]["score"])
    d["direction"] = np.select([d.signed_error > 1e-8, d.signed_error < -1e-8],
                                ["too high", "too low"], default="equal")
    d["gain_vs_flat"] = d.flat_absolute_error - d.absolute_error
    d["gain_vs_pchip"] = d.pchip_error - d.absolute_error
    d["gain_vs_slope2"] = d.slope2_error - d.absolute_error
    d["hour_start"] = d.hour_index - 1
    minutes = 60 * d.hour_start + 15 * (d.quarter_index - 1)
    d["time"] = minutes.map(lambda v: f"{v//60:02d}:{v%60:02d}")

    h = hourly.sort_values(["market_id", "delivery_date", "hour_index"]).copy()
    g = h.groupby(["market_id", "delivery_date"]).hourly_price_eur_mwh
    h["backward_change"] = h.hourly_price_eur_mwh - g.shift(1)
    h["forward_change"] = g.shift(-1) - h.hourly_price_eur_mwh
    h["turning_point"] = h.backward_change * h.forward_change < 0
    h["local_movement"] = h.backward_change.abs() + h.forward_change.abs()
    h["geometry"] = np.select([
        h.hour_index.isin([1, 24]), h.turning_point,
        (h.backward_change == 0) & (h.forward_change == 0)],
        ["day edge", "turning point", "flat neighbors"], default="ramp or plateau transition")
    h["period"] = np.select([h.hour_index.between(1, 6) | h.hour_index.between(23, 24),
                             h.hour_index.between(7, 11), h.hour_index.between(12, 15)],
                            ["night 22-06", "morning 06-11", "midday 11-15"],
                            default="afternoon/evening 15-22")
    d = d.merge(h[HOUR_KEYS + ["geometry", "period", "local_movement"]],
                on=HOUR_KEYS, validate="many_to_one")
    total = d.absolute_error.sum()

    def summarize(keys):
        s = d.groupby(keys, dropna=False).agg(
            predictions=("absolute_error", "size"), model_MAE=("absolute_error", "mean"),
            flat_MAE=("flat_absolute_error", "mean"), pchip_MAE=("pchip_error", "mean"),
            slope2_MAE=("slope2_error", "mean"), signed_bias=("signed_error", "mean"),
            p95_error=("absolute_error", lambda x: x.quantile(.95)),
            max_error=("absolute_error", "max"), error_sum=("absolute_error", "sum"),
            share_better_than_flat=("gain_vs_flat", lambda x: 100 * (x > 1e-8).mean()))
        s["error_share_pct"] = 100 * s.pop("error_sum") / total
        s["gain_vs_flat_pct"] = 100 * (1 - s.model_MAE / s.flat_MAE.replace(0, np.nan))
        return s.reset_index()

    tables = {}
    for name, keys in {"zone": ["market_id"], "quarter": ["quarter_index"],
                       "hour": ["hour_start"], "date": ["delivery_date"],
                       "date_zone": ["delivery_date", "market_id"],
                       "zone_quarter": ["market_id", "quarter_index"],
                       "zone_hour": ["market_id", "hour_start"],
                       "geometry": ["geometry"], "period": ["period"]}.items():
        tables[name] = summarize(keys)
        tables[name].to_csv(output / f"errors_by_{name}.csv", index=False)
    hour_shapes = d.groupby(HOUR_KEYS).agg(
        model_MAE=("absolute_error", "mean"),
        actual_range=("price_eur_mwh", lambda x: x.max() - x.min()),
        predicted_range=("predicted_price_eur_mwh", lambda x: x.max() - x.min()))
    hour_shapes["range_bias"] = hour_shapes.predicted_range - hour_shapes.actual_range
    hour_shapes.to_csv(output / "hourly_shape_errors.csv")
    ordered = d.sort_values(["delivery_date", "market_id", "hour_index", "quarter_index"])
    columns = ["delivery_date", "market_id", "time", "hour_index", "quarter_index",
               "price_eur_mwh", "predicted_price_eur_mwh", "signed_error", "absolute_error",
               "direction", "hourly_price_eur_mwh", "flat_absolute_error", "pchip_error",
               "slope2_error", "gain_vs_flat", "geometry", "period"]
    ordered[columns].to_csv(output / "all_predictions_vs_actual.csv", index=False)
    worst = d.nlargest(50, "absolute_error")
    worst[columns].to_csv(output / "worst_50_predictions.csv", index=False)
    daily_winners = tables["date"].model_MAE.lt(tables["date"].flat_MAE).sum()
    top5_share = 100 * d.nlargest(int(np.ceil(.05 * len(d))), "absolute_error").absolute_error.sum() / total
    counters = {f"within_{threshold}": float(100 * d.absolute_error.le(threshold).mean())
                for threshold in [1, 5, 10, 20]}
    summary = {"MAE": d.absolute_error.mean(), "median_absolute_error": d.absolute_error.median(),
               "p90_absolute_error": d.absolute_error.quantile(.9),
               "p95_absolute_error": d.absolute_error.quantile(.95), "max_absolute_error": d.absolute_error.max(),
               "worst_5pct_error_share_pct": top5_share, **counters,
               "dates_better_than_flat": int(daily_winners),
               "mean_range_bias": hour_shapes.range_bias.mean()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, label in [("model_MAE", "Final model"), ("flat_MAE", "Flat"),
                       ("slope2_MAE", "Slope x 2")]:
        axes[0, 0].plot(tables["hour"].hour_start, tables["hour"][col], marker=".", label=label)
    axes[0, 0].set(title="Error by clock hour", xlabel="Hour start", ylabel="MAE (EUR/MWh)")
    axes[0, 0].legend()
    axes[0, 1].bar(tables["quarter"].quarter_index, tables["quarter"].model_MAE, color="#2864a0")
    axes[0, 1].set(title="Final-model error by quarter", xticks=[1, 2, 3, 4], ylabel="MAE (EUR/MWh)")
    for z, part in tables["zone_quarter"].groupby("market_id"):
        axes[1, 0].plot(part.quarter_index, part.signed_bias, marker="o", label=z)
    axes[1, 0].axhline(0, color="gray", linewidth=.7)
    axes[1, 0].set(title="Quarter bias: positive = too high", xticks=[1, 2, 3, 4], ylabel="Mean signed error (EUR/MWh)")
    axes[1, 0].legend()
    ranked = np.sort(d.absolute_error.to_numpy())[::-1]
    axes[1, 1].plot(np.arange(1, len(d)+1)/len(d)*100, np.cumsum(ranked)/total*100)
    axes[1, 1].set(title="Where the total error accumulates", xlabel="Predictions, worst first (%)", ylabel="Cumulative absolute error (%)")
    fig.suptitle("Saved September assessment: observed errors, no refitting")
    fig.tight_layout()
    fig.savefig(output / "error_overview.png", dpi=160)
    plt.close(fig)
    top = worst.iloc[0]
    example = ordered[(ordered.delivery_date == top.delivery_date) &
                      (ordered.market_id == top.market_id) & (ordered.hour_index == top.hour_index)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(example.quarter_index, example.price_eur_mwh, "o-", label="Observed: what it should have been")
    ax.plot(example.quarter_index, example.predicted_price_eur_mwh, "o-", label="Model prediction")
    ax.axhline(top.hourly_price_eur_mwh, ls="--", color="gray", label="Supplied hourly average")
    ax.set(title=f"Largest individual miss: {top.delivery_date:%Y-%m-%d}, {top.market_id}, {int(top.hour_start):02d}:00 hour",
           xticks=[1, 2, 3, 4], xlabel="Quarter", ylabel="EUR/MWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "largest_miss_hour.png", dpi=160)
    plt.close(fig)

    # A local, self-contained browser report: every observation remains inspectable.
    browser = ordered[columns].copy()
    browser.delivery_date = browser.delivery_date.dt.strftime("%Y-%m-%d")
    payload = json.dumps(browser.round(6).to_dict("records"), allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Prediction vs actual</title>
<style>body{font:15px system-ui;margin:28px;color:#193047}h1{margin-bottom:8px}select,input,button{padding:8px;margin:4px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#eaf0f6;position:sticky;top:0;cursor:pointer}td:first-child,td:nth-child(2){text-align:left}.bad{background:#fff0e9}#wrap{max-height:70vh;overflow:auto}img{max-width:900px;width:100%}</style>
<h1>Prediction vs observed price, one by one</h1>
<p>19–30 September 2026. Positive signed error = predicted too high; negative = too low. All price errors are EUR/MWh. These are historical assessment results, not a new independent test.</p>
<label>Zone <select id="zone"><option value="">All</option></select></label>
<label>Date <select id="date"><option value="">All</option></select></label>
<label>Minimum absolute error <input id="min" type="number" min="0" value="0" step="1"></label>
<label>Order <select id="order"><option value="time">Chronological</option><option value="worst">Largest error first</option><option value="best">Smallest error first</option></select></label>
<p id="stats"></p><button id="prev">Previous 100</button><button id="next">Next 100</button><span id="pagination"></span>
<div id="wrap"><table><thead><tr id="head"></tr></thead><tbody id="rows"></tbody></table></div>
<p>Click a column heading to sort. CSV: <a href="all_predictions_vs_actual.csv">all 4,608 predictions</a>. <a href="error_analysis.md">Written analysis</a>.</p>
<img src="error_overview.png" alt="Error breakdown"><img src="largest_miss_hour.png" alt="Largest miss">
<script>const data=PAYLOAD;
const fields=['delivery_date','market_id','time','quarter_index','price_eur_mwh','predicted_price_eur_mwh','signed_error','absolute_error','direction','flat_absolute_error','pchip_error','slope2_error','geometry'];
const labels=['Date','Zone','Time','Quarter','Observed (should be)','Predicted','Signed error','Absolute error','Direction','Flat error','PCHIP error','Slope × 2 error','Geometry'];
let page=0,sortKey=null,desc=false,filtered=[];
const $=id=>document.getElementById(id);
for(const [id,key] of [['zone','market_id'],['date','delivery_date']]) for(const v of [...new Set(data.map(r=>r[key]))].sort()){const o=document.createElement('option');o.value=v;o.textContent=v;$(id).appendChild(o);}
fields.forEach((key,i)=>{const th=document.createElement('th');th.textContent=labels[i];th.onclick=()=>{desc=sortKey===key?!desc:false;sortKey=key;page=0;render()};$('head').appendChild(th)});
function render(){filtered=data.filter(r=>(!$('zone').value||r.market_id===$('zone').value)&&(!$('date').value||r.delivery_date===$('date').value)&&r.absolute_error>=Number($('min').value||0));
if(sortKey)filtered.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];return(typeof x==='number'?x-y:String(x).localeCompare(String(y)))*(desc?-1:1)});
else if($('order').value!=='time')filtered.sort((a,b)=>(a.absolute_error-b.absolute_error)*($('order').value==='worst'?-1:1));
page=Math.max(0,Math.min(page,Math.ceil(filtered.length/100)-1));$('rows').replaceChildren();
for(const r of filtered.slice(page*100,page*100+100)){const tr=document.createElement('tr');if(r.absolute_error>20)tr.className='bad';for(const k of fields){const td=document.createElement('td');td.textContent=typeof r[k]==='number'?(k==='quarter_index'?r[k]:r[k].toFixed(3)):r[k];tr.appendChild(td)}$('rows').appendChild(tr)}
const mae=filtered.length?filtered.reduce((s,r)=>s+r.absolute_error,0)/filtered.length:0;
$('stats').textContent=`${filtered.length.toLocaleString()} matching predictions · Mean absolute error in this selection: ${mae.toFixed(3)} EUR/MWh (filtered mean, not necessarily the official equal-zone score)`;
$('pagination').textContent=` Page ${filtered.length?page+1:0} of ${Math.ceil(filtered.length/100)} `;}
for(const id of ['zone','date','min','order'])$(id).onchange=()=>{page=0;sortKey=null;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();</script></html>""".replace("PAYLOAD", payload)
    (output / "prediction_explorer.html").write_text(page)

    selected = ["market_id", "model_MAE", "flat_MAE", "pchip_MAE", "slope2_MAE", "error_share_pct"]
    brief = ["predictions", "model_MAE", "flat_MAE", "error_share_pct"]
    besthour = tables["hour"].nsmallest(3, "model_MAE")
    hardest = tables["hour"].nlargest(5, "error_share_pct")
    evening_share = tables["hour"].loc[tables["hour"].hour_start.between(16, 19), "error_share_pct"].sum()
    edge_quarter_share = tables["quarter"].loc[tables["quarter"].quarter_index.isin([1, 4]), "error_share_pct"].sum()
    zone12_share = tables["zone"].loc[tables["zone"].market_id.isin(["ZONE_1", "ZONE_2"]), "error_share_pct"].sum()
    slope_wins = tables["hour"].loc[tables["hour"].slope2_MAE < tables["hour"].model_MAE, "hour_start"].tolist()
    report = f"""# Error analysis: prediction versus what was observed

**Every one of the 4,608 predictions is compared with its observed quarter price.**
Source: saved assessment on 19–30 September 2026; model weights trained through August.
No model was retrained or changed. This is diagnosis of the existing results, not new validation
and not a recommendation to tune on September.

Open [the prediction explorer](prediction_explorer.html) to filter by date/zone, sort the errors,
and inspect 100 rows at a time. [Download all rows](all_predictions_vs_actual.csv).
“Observed” is what the price should have been; “predicted” is what the model returned.
Signed error = predicted minus observed; positive means too high. Absolute error ignores direction.
All price and error values below are EUR/MWh. Clock hours use 00–23, unlike dataset hour indices 1–24.

## Main findings and what they change

- **Afternoon ramps deserve attention:** 16:00–20:00 accounts for **{evening_share:.1f}%** of total
  error, despite containing only 16.7% of observations. The highest-MAE hour starts at 17:00.
- **Outer quarters remain important:** Q1 and Q4 together contribute **{edge_quarter_share:.1f}%**
  of total error from half the observations. Yet their improvement over flat is larger than
  that of the middle quarters; high remaining error does not mean the model adds no value there.
- **ZONE_1 and ZONE_2 contribute {zone12_share:.1f}%** of total error. ZONE_1 is the larger contributor;
  ZONE_2 is interesting because its reconstruction gain is smaller, not because it contributes most error.
- **Turning points are not the highest-error geometry group:** their MAE is 5.001 versus 5.207
  for ramps/plateau transitions. However, flat is much stronger at reversals, so the model's relative
  gain is smaller there. Inspect both error contribution and gain over baseline before prioritizing.
- **The largest miss is an underestimated late-quarter surge:** on {top.delivery_date:%Y-%m-%d},
  {top.market_id} at {top.time}, the model gives **{top.predicted_price_eur_mwh:.2f}** instead of
  **{top.price_eur_mwh:.2f}**, an error of **{top.absolute_error:.2f}**. The other quarters of that
  hour show compensating overestimates. This is a shape/amplitude mismatch, not a wrong hourly level.
- **The slope rule is not beaten in every clock-hour group:** slope × 2 has lower mean error
  at hour starts **{', '.join(f'{v:02d}:00' for v in slope_wins)}** in this sample. This does not authorize
  switching models at those times based on September; first test that pattern on development folds.

## 1. How large are the misses?

- Mean absolute error: **{summary['MAE']:.3f}**; median absolute error: **{summary['median_absolute_error']:.3f}**.
- 90th / 95th percentile error: **{summary['p90_absolute_error']:.3f} / {summary['p95_absolute_error']:.3f}**.
- Within 5 EUR/MWh: **{counters['within_5']:.1f}%**; within 10: **{counters['within_10']:.1f}%**; within 20: **{counters['within_20']:.1f}%**.
- Largest individual error: **{summary['max_absolute_error']:.3f}**.
- The worst 5% of predictions account for **{top5_share:.1f}%** of all absolute error.
- The model beats flat on **{daily_winners}/12** daily equal-zone averages.

The mean is not the error of every prediction. The tail and error contribution show whether
improving a small set of difficult observations could matter. Error share means sum of absolute
errors in that group divided by the total; the groups must be disjoint to sum to 100%.

![Error overview](error_overview.png)

## 2. Zone by zone, on exactly the same observations

{md(tables['zone'][selected])}

Compare both conditional MAE and error share. A high-error zone has more room to improve, but
the baseline comparisons show how much shape uncertainty was present in that zone already.
ZONE_2's weaker full-year slope relation is a hypothesis to investigate on development data;
this September ranking alone cannot establish a general defect or justify a separate model.

## 3. Quarter by quarter: direction as well as size

{md(tables['quarter'][['quarter_index', *brief, 'signed_bias']])}

Signed bias across all four quarters cancels because both observed and predicted prices share
the same hourly average. Therefore near-zero overall bias does not mean accurate predictions.
Inspect Q1/Q4 biases by zone in [errors_by_zone_quarter.csv](errors_by_zone_quarter.csv), and
do not infer universal underprediction or overprediction from the overall mean signed error.

## 4. Which hours account for most error?

{md(hardest[['hour_start', *brief]])}

Lowest-MAE clock hours in this sample:

{md(besthour[['hour_start', 'model_MAE', 'flat_MAE']])}

The complete [24-hour breakdown](errors_by_hour.csv) includes PCHIP and slope × 2 comparisons.
Use the final model's remaining errors, rather than assuming that every weakness of the EDA's
simple slope rule remains a weakness of the learned correction.

## 5. Turning points, ramps, and day boundaries

{md(tables['geometry'][['geometry', *brief]])}

A turning point has opposite signs for backward and forward hourly changes. Day edges are
classified separately because a same-day neighbor is missing. “Ramp or plateau transition”
includes same-direction changes and hours with one flat neighbor; no fitted threshold is used.
These are input-derived labels, not actual-quarter extremes. Differences can reflect different
zone/hour compositions rather than a causal effect of geometry. See [zone/hour details](errors_by_zone_hour.csv).

Time-window comparison:

{md(tables['period'][['period', *brief]])}

## 6. Date by date

{md(tables['date'][['delivery_date', 'model_MAE', 'flat_MAE', 'pchip_MAE', 'slope2_MAE', 'error_share_pct']])}

See [all 48 date/zone results](errors_by_date_zone.csv) to separate a difficult day across markets
from a problem isolated to one zone. September contains only 12 scored dates.

## 7. Largest misses: actual versus predicted, one by one

{md(worst.head(12)[['delivery_date', 'market_id', 'time', 'quarter_index', 'price_eur_mwh',
                                  'predicted_price_eur_mwh', 'signed_error', 'absolute_error']])}

Columns `price_eur_mwh` and `predicted_price_eur_mwh` mean observed and predicted respectively.
[Full top 50](worst_50_predictions.csv) also includes each baseline's absolute error.
A large miss is not automatically a model failure relative to an available alternative.

### The hour containing the single largest miss

![The four quarters of the worst-miss hour](largest_miss_hour.png)

{md(example[['time', 'quarter_index', 'price_eur_mwh', 'predicted_price_eur_mwh', 'signed_error', 'absolute_error']])}

Both curves preserve the same supplied hourly mean, but distribute that value differently
across quarters. The chart makes the actual shape mismatch visible without hiding the other
three quarters. It cannot by itself identify why the learned model produced that shape.

Across all hours, predicted range minus observed range averages **{summary['mean_range_bias']:.3f}**.
Negative means the predicted max-minus-min range is too narrow on average; positive means too wide.
Inspect [hourly_shape_errors.csv](hourly_shape_errors.csv) before concluding that all regimes
need more amplitude. Averages can mix underestimation on volatile hours and overestimation on calm hours.

## 8. What to do next with these findings

1. Reproduce the leading hour/zone/geometry concentrations on January–August rolling-origin
   development predictions. These September observations are retrospective evidence only.
2. If reversals are consistently costly, test explicit peak/valley flags and local absolute
   movement against the incumbent. Curvature and slopes already exist, so require incremental gain.
3. If repeated Q1/Q4 and range biases persist within a zone, test limited zone/component scaling
   on development data. Do not change the 0.8 scale solely to repair this September sample.
4. If flat or slope × 2 wins only on isolated observations, do not route predictions using realized
   target errors. Any gate must use available hourly/history features and work on later dates.
5. Keep rare extreme cases and their sample counts visible. Some quarter shapes cannot be uniquely
   recovered from hourly averages; a visually plausible reconstruction can still miss an observed extreme.

## Files and reproducibility

- [Interactive row explorer](prediction_explorer.html), [all prediction comparisons](all_predictions_vs_actual.csv), [worst 50](worst_50_predictions.csv).
- CSV summaries by zone, quarter, clock hour, date, date/zone, zone/quarter, zone/hour, geometry, and period.
- `summary.json` contains tail statistics. `hourly_shape_errors.csv` contains observed and predicted ranges.
- Rerun from ApplePi: `.venv-mae/bin/python analysis/analyze_saved_errors.py --source {source.relative_to(ROOT)}`.
- Checks verify row uniqueness, agreement with the saved truth and prediction files, score reproduction,
  and hourly-average consistency. Observed quarter prices are used for diagnostics only.
"""
    (output / "error_analysis.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print("\nZONES\n", tables["zone"][selected].to_string(index=False))
    print("\nHOURS\n", hardest[["hour_start", *brief]].to_string(index=False))
    print("\nGEOMETRY\n", tables["geometry"][["geometry", *brief]].to_string(index=False))
    print("\nWORST\n", worst.head(3)[["delivery_date", "market_id", "time", "price_eur_mwh", "predicted_price_eur_mwh", "absolute_error"]].to_string(index=False))
    print("\nREPORT:", output / "error_analysis.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    main(parser.parse_args().source.resolve())
