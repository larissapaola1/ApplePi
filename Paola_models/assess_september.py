"""Score the original August-cutoff TGNN on the requested Sep 19–30 assessment.

This is reporting only: no training, model selection, or artifact modification.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from . import REPO
from .temporal_graph import GraphReconstructor, ZONES
from .cli import markdown_table
from aet.cube import QuarterCube
from aet.infer import run as run_aet
from aet.schema import HOUR_KEYS, QH_KEYS, read_hourly, read_quarter_hourly, validate_predictions
from tools.score import score


def main(out):
    source=REPO/"analysis/results/mae_evaluation_20261017_175448"
    graph_path=REPO/"Paola_models/artifacts/through_august"
    aet_path=REPO/"AET-HACKATHON/artifacts_september_assessment"
    manifests={name:json.loads((path/"manifest.json").read_text())
               for name,path in [("TGNN",graph_path),("AET",aet_path)]}
    assert all(m["training_range"][1]=="2026-08-31" for m in manifests.values())
    # Also protects the unused full-year incumbent from accidental modification.
    files=[p for folder in [graph_path,aet_path,REPO/"AET-HACKATHON/artifacts_official"]
           for p in folder.iterdir() if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out.mkdir(parents=True,exist_ok=False)
    hourly=read_hourly(source/"test_hourly.csv")
    truth=read_quarter_hourly(source/"truth_15min.csv")
    assert len(truth)==4608 and truth.delivery_date.nunique()==12
    assert truth.delivery_date.min()==pd.Timestamp("2026-09-19")
    assert truth.delivery_date.max()==pd.Timestamp("2026-09-30")
    assert set(truth.market_id)==set(ZONES)
    assert truth.groupby(["delivery_date","market_id"]).size().eq(96).all()
    graph=GraphReconstructor.load(graph_path)
    prior=QuarterCube.from_long(read_quarter_hourly(source/"history_15min.csv",strict=False),ZONES)
    history=graph.training_history.merge(prior)
    predictions=graph.predict_frame(hourly,history)
    validate_predictions(predictions,hourly)
    predictions.to_csv(out/"tgnn_predictions.csv",index=False,float_format="%.10f")
    predictions=pd.read_csv(out/"tgnn_predictions.csv",parse_dates=["delivery_date"])
    checks=validate_predictions(predictions,hourly)
    diagnostics=run_aet(source/"test_hourly.csv",source/"history_15min.csv",aet_path,out/"aet_predictions.csv")
    assert not diagnostics.get("emergency_flat",False),diagnostics
    assert diagnostics["fallback_hours"]==diagnostics["flat_hours"]==0
    aet=pd.read_csv(out/"aet_predictions.csv",parse_dates=["delivery_date"])
    gz,gq,go=score(predictions,truth)
    az,aq,ao=score(aet,truth)
    comparison=pd.DataFrame([{"method":"Flat hourly repetition","MAE_EUR_MWh":go["flat"]},
                            {"method":"AET LightGBM","MAE_EUR_MWh":ao["score"]},
                            {"method":"Original TGNN","MAE_EUR_MWh":go["score"]}])
    comparison.to_csv(out/"method_comparison.csv",index=False)
    zones=gz[["model","flat"]].rename(columns={"model":"TGNN_MAE","flat":"flat_MAE"}).join(az.model.rename("AET_MAE"))
    zones["TGNN_minus_AET"]=zones.TGNN_MAE-zones.AET_MAE
    zones.to_csv(out/"mae_by_zone.csv")
    quarters=gq.rename(columns={"model":"TGNN_MAE","flat":"flat_MAE"}).join(aq.model.rename("AET_MAE"))
    quarters.to_csv(out/"mae_by_quarter.csv")
    rows=truth.merge(predictions,on=QH_KEYS,validate="one_to_one").merge(
        aet.rename(columns={"predicted_price_eur_mwh":"aet_prediction"}),on=QH_KEYS,validate="one_to_one")
    rows["TGNN_signed_error"]=rows.predicted_price_eur_mwh-rows.price_eur_mwh
    rows["TGNN_abs_error"]=rows.TGNN_signed_error.abs()
    rows["AET_abs_error"]=(rows.aet_prediction-rows.price_eur_mwh).abs()
    rows.to_csv(out/"predictions_vs_actual.csv",index=False)
    daily=rows.groupby(["delivery_date","market_id"])[["TGNN_abs_error","AET_abs_error"]].mean()
    daily.to_csv(out/"daily_zone_mae.csv")
    dates=daily.groupby("delivery_date").mean()
    dates.to_csv(out/"daily_mae.csv")
    assert np.isclose(daily.groupby("market_id").TGNN_abs_error.mean().mean(),go["score"])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
    relative=100*(go["score"]/ao["score"]-1)
    direction="higher" if relative>0 else "lower"
    summary={"evaluation_start":"2026-09-19","evaluation_end":"2026-09-30",
        "dates":12,"zones":list(ZONES),"rows":4608,"training_cutoff":"2026-08-31",
        "TGNN":go,"AET":ao,"TGNN_relative_error_change_vs_AET_pct":relative,
        "tgnn_output_validation":checks,"aet_inference_diagnostics":diagnostics,
        "original_artifact_hashes_unchanged":hashes,
        "source_inputs":str(source),"graph_model":str(graph_path),"aet_model":str(aet_path),
        "selection_or_training_performed":False}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    report=f"""# Original temporal graph model: September 19–30 assessment

**TGNN MAE: {go['score']:.4f} EUR/MWh. AET LightGBM: {ao['score']:.4f} EUR/MWh.**
The graph's error is **{abs(relative):.2f}% {direction}** than AET on these same observations.
Both models were trained through 31 August; neither was retrained or tuned for this assessment.
This is the original 80-epoch graph associated with 4.2611 January–August validation MAE,
not the later optimized graph or blend retained separately in the backup.

## Same-date, four-zone comparison

{markdown_table(comparison)}

All scores use September 19–30, 12 dates, four zones, **4,608 quarter-price predictions**.
MAE is calculated per date/zone, averaged over dates within each zone, and then equally over
the four zones (25% each). Units are EUR/MWh, not a percentage accuracy. Lower is better.

## Each zone

{markdown_table(zones.reset_index())}

`TGNN_minus_AET` is positive where graph error is higher.

## Each quarter

{markdown_table(quarters.reset_index())}

## Each date (equal-zone averages)

{markdown_table(dates.reset_index())}

## Interpretation and checks

- The 3.9479 AET / 4.2611 original-graph figures refer to January–August development folds.
  Compare the two September figures above with each other, not with the development figures.
- All 24 hourly prices are supplied inputs. For each target date, quarter-price history is sliced
  strictly before that date. Earlier assessment dates may become history for later dates.
- No October data, optimized-model restoration, hyperparameter selection, or model fitting occurred.
- Both prediction files passed finite-value, complete-key, and hourly-mean checks. TGNN maximum
  hourly-mean error after CSV reload: {checks['max_abs_mean_error']:.3g} EUR/MWh (limit 0.0001).
- Both saved August-cutoff models and the unused full-year incumbent have unchanged hashes.
- September was already EDA-exposed. This is the requested historical assessment, not a new
  independent test or evidence for choosing a new configuration based on September.

Files: `tgnn_predictions.csv`, `aet_predictions.csv`, `predictions_vs_actual.csv`, zone/quarter/date
summaries, `method_comparison.csv`, and `summary.json` for provenance and checks.

Reproduce from ApplePi with a new output directory:

```bash
.venv-mae/bin/python -m Paola_models.assess_september --out Paola_models/results/NEW_september_assessment
```
"""
    (out/"assessment_results.md").write_text(report)
    print(comparison.to_string(index=False))
    print("\nBY ZONE\n"+zones.to_string())
    print(f"\nTGNN error is {abs(relative):.2f}% {direction} than AET.")
    print("REPORT:",out/"assessment_results.md")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=REPO/"Paola_models/results/september_19_30_assessment")
    main(parser.parse_args().out)
