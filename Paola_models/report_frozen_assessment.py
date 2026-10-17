"""Report all frozen September candidates together, without fitting or selection.

Predictions are generated using the documented CLI infer commands. This script
validates and scores the saved files, including the already selected .25 blend.
"""
from pathlib import Path
import hashlib
import json
import argparse

import numpy as np
import pandas as pd

from . import REPO
from .cli import markdown_table
from aet.schema import QH_KEYS, read_hourly, validate_predictions
from tools.score import score


def main(out):
    source=REPO/"analysis/results/mae_evaluation_20261017_175448"
    assessment=REPO/"Paola_models/results/september_19_30_assessment"
    optimization=REPO/"Paola_models/results/aet_informed_optimization"
    selection=json.loads((optimization/"selection_before_confirmation.json").read_text())
    assert selection["graph_weight"]==.25
    assert selection["selection_folds"]==[f"m2026_{m:02d}" for m in range(1,6)]
    truth=pd.read_csv(source/"truth_15min.csv",parse_dates=["delivery_date"])
    hourly=read_hourly(source/"test_hourly.csv")
    assert len(truth)==4608 and truth.delivery_date.nunique()==12
    assert truth.delivery_date.min()==pd.Timestamp("2026-09-19")
    assert truth.delivery_date.max()==pd.Timestamp("2026-09-30")
    files={"AET LightGBM":"aet_predictions.csv","Original TGNN":"tgnn_predictions.csv",
           "Optimized TGNN":"optimized_graph_predictions.csv",
           "Frozen 25% graph / 75% AET blend":"frozen_blend_predictions.csv"}
    out.mkdir(parents=True,exist_ok=False)
    rows=truth.copy();comparisons=[];zone_scores=[];quarter_scores=[];checks={}
    for method,name in files.items():
        frame=pd.read_csv(assessment/name,parse_dates=["delivery_date"])
        checks[method]=validate_predictions(frame,hourly)
        z,q,s=score(frame,truth)
        comparisons.append({"method":method,"MAE_EUR_MWh":s["score"],"reduction_vs_flat_pct":s["gain_pct"]})
        zone_scores.append(z.model.rename(method))
        quarter_scores.append(q.model.rename(method))
        rows=rows.merge(frame.rename(columns={"predicted_price_eur_mwh":method}),on=QH_KEYS,validate="one_to_one")
    # Verify the actual saved hybrid is precisely the weight locked before confirmation.
    blend="Frozen 25% graph / 75% AET blend"
    np.testing.assert_allclose(rows[blend],.25*rows["Optimized TGNN"]+.75*rows["AET LightGBM"],atol=2e-10,rtol=0)
    comparison=pd.DataFrame(comparisons)
    aet=comparison.loc[comparison.method=="AET LightGBM","MAE_EUR_MWh"].iloc[0]
    best=comparison.loc[comparison.method==blend,"MAE_EUR_MWh"].iloc[0]
    comparison["change_vs_AET_pct"]=100*(comparison.MAE_EUR_MWh/aet-1)
    comparison.to_csv(out/"method_comparison.csv",index=False)
    zones=pd.concat(zone_scores,axis=1);zones.to_csv(out/"mae_by_zone.csv")
    quarters=pd.concat(quarter_scores,axis=1);quarters.to_csv(out/"mae_by_quarter.csv")
    methods=list(files)
    for method in methods:
        rows[method+" abs error"]=(rows[method]-rows.price_eur_mwh).abs()
    rows.to_csv(out/"all_predictions_vs_actual.csv",index=False)
    daily=rows.groupby(["delivery_date","market_id"])[[m+" abs error" for m in methods]].mean().groupby("delivery_date").mean()
    daily.to_csv(out/"daily_mae.csv")
    artifact_dirs=[REPO/"AET-HACKATHON/artifacts_official",REPO/"AET-HACKATHON/artifacts_september_assessment",
                   REPO/"Paola_models/artifacts/through_august",optimization/"selected_graph_through_august",
                   optimization/"selected_blend_through_august"]
    artifact_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for directory in artifact_dirs for p in directory.rglob('*') if p.is_file()}
    manifests=[json.loads((directory/"manifest.json").read_text()) for directory in artifact_dirs[1:]]
    assert all(m["training_range"][1]=="2026-08-31" for m in manifests)
    archived=REPO/"Paola_models/results/tgnn_80epochs/run.json"
    old_hashes=json.loads(archived.read_text())["artifact_hashes_unchanged"]
    for oldpath,digest in old_hashes.items():
        local=REPO/('AET-HACKATHON/'+oldpath.split('AET-HACKATHON/',1)[1])
        assert hashlib.sha256(local.read_bytes()).hexdigest()==digest
    meta={"assessment_start":"2026-09-19","assessment_end":"2026-09-30","dates":12,"zones":4,
          "predictions":4608,"training_cutoff":"2026-08-31","score_below_5":bool(best<5),
          "frozen_blend_MAE":best,"margin_below_5":5-best,"AET_MAE":aet,
          "relative_gain_vs_AET_pct":100*(1-best/aet),"graph_weight":.25,
          "selection":selection,"new_fitting_or_tuning_on_September":False,"validation":checks,
          "artifact_hashes":artifact_hashes,
          "input_hashes":{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ['test_hourly.csv','history_15min.csv','truth_15min.csv']}}
    (out/"summary.json").write_text(json.dumps(meta,indent=2)+"\n")
    report=f"""# Frozen hybrid: September MAE below 5

**The frozen 25% optimized graph / 75% AET blend scores {best:.6f} EUR/MWh MAE.**
This is {100*(1-best/aet):.2f}% lower error than AET's {aet:.6f}, and only **{5-best:.6f} EUR/MWh below 5**.
The pure optimized graph does not reach 5. This result is a **hybrid containing a neural network
and LightGBM**, not a pure-neural score or a guarantee for other dates.

## Fair same-date comparison

{markdown_table(comparison)}

All rows use Sep 19–30, four equally weighted zones, 12 dates, and 4,608 predictions. Both
component models were trained through August 31. The official score averages daily MAE within
each zone, then the four zones equally. MAE units are EUR/MWh, not percent accuracy.

## Each zone

{markdown_table(zones.reset_index())}

The blend's pooled improvement need not imply improvement in every zone; inspect all four.

## Each quarter

{markdown_table(quarters.reset_index())}

## What was fixed before this assessment

The optimized architecture and .25 graph weight were selected using January–May development
predictions and recorded in `aet_informed_optimization/selection_before_confirmation.json`.
They were then checked on June–August: blend 4.2057 versus AET 4.2246. That earlier 0.45% gain
failed the 0.5% promotion threshold. Nothing was retuned to turn the September score into a pass.

For this request, the saved optimization was restored from its verified backup directly into
`Paola_models/` on the **Paola branch**. No temporary checkout was recreated. The original README's
September results and all original artifacts were preserved. Both frozen optimized checkpoints
were evaluated exactly as saved. Every model's result is reported, including those above 5.

September had already been explored in EDA and assessed for the original models. This is a
historical assessment of previously fixed recipes, not a wholly untouched test. Repeatedly trying
new settings until this September number falls would make it a tuning score; no such search was
performed for this result. Do not treat the small margin below 5 as a robust performance guarantee.

## Run the hybrid

From the ApplePi root, on Paola:

```bash
.venv-mae/bin/python -m Paola_models.cli infer \\
  --model Paola_models/results/aet_informed_optimization/selected_blend_through_august \\
  --hourly test_hourly.csv --history history_15min.csv \\
  --out Paola_models/NEW_predictions.csv
```

The input dates must follow the August cutoff. Inference does not train, recalibrate, or update
the .25 blend coefficient. The history is sliced strictly before each target date. All four
methods passed complete-key, finite-price, and hourly-mean checks after CSV reload.

Files here include `method_comparison.csv`, `mae_by_zone.csv`, `mae_by_quarter.csv`, `daily_mae.csv`,
`all_predictions_vs_actual.csv`, and `summary.json`. Raw prediction files remain in
`../september_19_30_assessment/`. `summary.json` records the fixed selection, input/weight hashes,
and validation. No AET weights were replaced, no commits were made, and nothing was pushed.
"""
    (out/"below_5_results.md").write_text(report)
    print(comparison.to_string(index=False));print("\nREPORT:",out/"below_5_results.md")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=REPO/"Paola_models/results/frozen_hybrid_september")
    main(parser.parse_args().out)
