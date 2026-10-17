"""Bounded AET-informed study: Jan–May selection, Jun–Aug development confirmation."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import REPO
from .temporal_graph import Config, GraphReconstructor
from .cli import training_cube, markdown_table
from .hybrid import save_hybrid
from aet.config import load_config, folds_from_config
from aet.schema import QH_KEYS, HOUR_KEYS, validate_predictions


def official(frame, prediction="predicted_price_eur_mwh"):
    frame=frame.assign(error=(frame[prediction]-frame.price_eur_mwh).abs())
    return float(frame.groupby(["delivery_date","market_id"]).error.mean().groupby("market_id").mean().mean())


def evaluate(model,cube,fold):
    frames=[]
    for i in np.flatnonzero((cube.dates>=fold.score_start)&(cube.dates<=fold.score_end)):
        day,H=cube.dates[i],cube.values[i].mean(axis=-1)
        value=model.predict_day(day,H,cube)
        if not np.isfinite(value).all() or np.max(np.abs(value.mean(axis=-1)-H))>.0001:
            raise ValueError("invalid graph prediction")
        f=cube._select(cube.dates==day).to_long()
        f["fold"]=fold.name
        f["predicted_price_eur_mwh"]=value.reshape(-1)
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def main(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    base=Config()
    candidates={
        "aet_features":replace(base,cross_zone=True),
        "aet_loss_shrink":replace(base,cross_zone=True,loss="coefficient",residual_scale=.8),
        "compact_local":replace(base,hidden=24,blocks=1,epochs=120,cross_zone=True,
                                 local_skip=True,loss="coefficient",residual_scale=.8)}
    alphas=[0.,.1,.25,.5,1.]
    folds,_=folds_from_config(load_config(REPO/"AET-HACKATHON/configs/final.yaml"),"monthly")
    search,confirm=folds[:5],folds[5:]
    search_names={f.name for f in search}
    plan={"candidates":{k:asdict(v) for k,v in candidates.items()},"graph_blend_weights":alphas,
          "selection_folds":[f.name for f in search],"confirmation_folds":[f.name for f in confirm],
          "rule":"minimum Jan-May official MAE, including original graph; ties prefer original",
          "blend_rule":"minimum Jan-May MAE; ties prefer lower graph weight",
          "no_retuning_after_confirmation":True,
          "limitation":"All development months have been inspected; confirmation is not untouched."}
    (out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    cube=training_cube(args.train,"2026-08-31")
    original=pd.read_csv(REPO/"Paola_models/results/tgnn_80epochs/predictions_vs_actual.csv",parse_dates=["delivery_date"])
    reference=pd.read_csv(args.aet_predictions,parse_dates=["delivery_date"])
    reference=reference[QH_KEYS+["predicted_price_eur_mwh","price_eur_mwh","fold"]].rename(
        columns={"predicted_price_eur_mwh":"aet_prediction","price_eur_mwh":"aet_actual"})
    if reference.delivery_date.max()>=pd.Timestamp("2026-09-01"):
        raise ValueError("reference must contain development dates only")
    joined=original.merge(reference,on=QH_KEYS+["fold"],validate="one_to_one")
    assert len(joined)==len(original)==len(reference)==37632
    assert np.allclose(joined.price_eur_mwh,joined.aet_actual)
    t=joined.merge(cube.to_long(),on=QH_KEYS,validate="one_to_one",suffixes=("","_official"))
    assert np.allclose(t.price_eur_mwh,t.price_eur_mwh_official)
    if not np.isclose(official(joined,"aet_prediction"),3.9479016177112345,atol=1e-8):
        raise ValueError("reference must reproduce the fixed current incumbent")
    paths=[p for folder in ["artifacts_official","artifacts_september_assessment"]
           for p in (REPO/"AET-HACKATHON"/folder).iterdir() if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows={"original":joined[joined.fold.isin(search_names)].copy()}
    ledger=[{"candidate":"original","search_MAE":official(rows["original"]),"seconds":0}]
    for name,cfg in candidates.items():
        frames=[];tick=time.perf_counter()
        for fold in search:
            model=GraphReconstructor(cfg).fit(cube.until(fold.train_end))
            frames.append(evaluate(model,cube,fold))
            print(f"{name} {fold.name} completed",flush=True)
        frame=pd.concat(frames,ignore_index=True).merge(reference,on=QH_KEYS+["fold"],validate="one_to_one")
        frame.to_csv(out/f"search_{name}_predictions.csv",index=False)
        rows[name]=frame
        ledger.append({"candidate":name,"search_MAE":official(frame),"seconds":time.perf_counter()-tick})
        pd.DataFrame(ledger).to_csv(out/"search_ledger.csv",index=False)
        print(f"{name} search MAE: {ledger[-1]['search_MAE']:.6f}",flush=True)
    selected=min(ledger,key=lambda r:r["search_MAE"])["candidate"]
    cfg=base if selected=="original" else candidates[selected]
    blend_trials=[]
    for alpha in alphas:
        f=rows[selected].assign(blend=alpha*rows[selected].predicted_price_eur_mwh+
                               (1-alpha)*rows[selected].aet_prediction)
        blend_trials.append({"graph_weight":alpha,"search_MAE":official(f,"blend")})
    alpha=min(blend_trials,key=lambda r:r["search_MAE"])["graph_weight"]
    selection={"candidate":selected,"config":asdict(cfg),"graph_weight":alpha,
               "selection_folds":sorted(search_names),"blend_trials":blend_trials}
    # Write the exact choice BEFORE confirmation. No choice changes after this point.
    (out/"selection_before_confirmation.json").write_text(json.dumps(selection,indent=2)+"\n")
    print("LOCKED SELECTION",selected,"graph weight",alpha,flush=True)
    frames=[]
    for fold in confirm:
        model=GraphReconstructor(cfg).fit(cube.until(fold.train_end))
        frames.append(evaluate(model,cube,fold))
        print(f"confirmation {fold.name} completed",flush=True)
    conf=pd.concat(frames,ignore_index=True).merge(reference,on=QH_KEYS+["fold"],validate="one_to_one")
    detail=pd.concat([rows[selected],conf],ignore_index=True)
    detail["blend_prediction"]=alpha*detail.predicted_price_eur_mwh+(1-alpha)*detail.aet_prediction
    detail["stage"]=np.where(detail.fold.isin(search_names),"selection_Jan_May","confirmation_Jun_Aug")
    detail.to_csv(out/"selected_predictions.csv",index=False)
    comparisons=[];fold_results=[]
    for stage,part in detail.groupby("stage"):
        old=joined[joined.fold.isin(part.fold.unique())]
        for name,col,f in [("AET incumbent","aet_prediction",part),
                           ("Original TGNN","predicted_price_eur_mwh",old),
                           ("Selected TGNN","predicted_price_eur_mwh",part),
                           ("Selected blend","blend_prediction",part)]:
            comparisons.append({"stage":stage,"method":name,"MAE":official(f,col)})
    for fold,part in detail.groupby("fold"):
        fold_results.append({"fold":fold,"stage":part.stage.iloc[0],"AET":official(part,"aet_prediction"),
                             "graph":official(part),"blend":official(part,"blend_prediction")})
    scores=pd.DataFrame(comparisons);f=pd.DataFrame(fold_results)
    scores.to_csv(out/"stage_scores.csv",index=False);f.to_csv(out/"fold_scores.csv",index=False)
    saved=pd.read_csv(out/"selected_predictions.csv",parse_dates=["delivery_date"])
    hourly=saved.groupby(HOUR_KEYS,as_index=False).price_eur_mwh.mean().rename(columns={"price_eur_mwh":"hourly_price_eur_mwh"})
    validations={}
    for col in ["predicted_price_eur_mwh","blend_prediction"]:
        pred=saved[QH_KEYS+[col]].rename(columns={col:"predicted_price_eur_mwh"})
        validations[col]=validate_predictions(pred,hourly)
    conf=saved[saved.stage=="confirmation_Jun_Aug"]
    zones=conf.assign(graph_error=(conf.predicted_price_eur_mwh-conf.price_eur_mwh).abs(),
                      blend_error=(conf.blend_prediction-conf.price_eur_mwh).abs(),
                      aet_error=(conf.aet_prediction-conf.price_eur_mwh).abs()).groupby(
                          ["delivery_date","market_id"])[["graph_error","blend_error","aet_error"]].mean().groupby("market_id").mean()
    zones.to_csv(out/"confirmation_zone_scores.csv")
    cf=f[f.stage=="confirmation_Jun_Aug"]
    rel=1-official(conf,"blend_prediction")/official(conf,"aet_prediction")
    wins=int((cf.blend<cf.AET).sum())
    worst=float((zones.blend_error/zones.aet_error-1).max())
    promoted=bool(rel>=.005 and wins>=2 and worst<=.02)
    fitted=GraphReconstructor(cfg).fit(cube)
    fitted.save(out/"selected_graph_through_august")
    if 0<alpha<1:
        save_hybrid(out/"selected_blend_through_august",fitted,
                    REPO/"AET-HACKATHON/artifacts_september_assessment",alpha,selection)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    summary={"selection":selection,"confirmation_blend_relative_gain_vs_AET":rel,
             "confirmation_blend_folds_won":wins,"confirmation_blend_worst_zone_degradation":worst,
             "confirmation_gate_passed":promoted,"validation":validations,
             "artifact_hashes_unchanged":hashes,"source_AET_predictions":str(args.aet_predictions),
             "source_AET_predictions_sha256":hashlib.sha256(args.aet_predictions.read_bytes()).hexdigest()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    report=f"""# AET-informed graph optimization

Selection used January–May only. Before scoring June–August, the choice was locked to
**{selected}**, with graph blend weight **{alpha:g}**. The blend combines AET and graph prices;
it is not a pure graph model. Weight zero means AET only, and one means graph only.

## Every selection trial

{markdown_table(pd.DataFrame(ledger))}

{markdown_table(pd.DataFrame(blend_trials))}

## Stage scores (EUR/MWh)

{markdown_table(scores)}

## Monthly scores

{markdown_table(f)}

## June–August zone scores

{markdown_table(zones.reset_index())}

The locked blend changes confirmation MAE by {-100*rel:+.2f}% versus AET, wins {wins}/3 folds,
and has worst-zone degradation {100*worst:+.2f}%. Confirmation gate passed: **{promoted}**
(>=0.5% gain, >=2/3 folds won, no zone >2% worse). No incumbent artifact/config was replaced.
Selection scores are optimistic after candidate comparisons. All development months were inspected
previously: this confirmation is not an untouched test. September and October were excluded.

## Changes tested

- Explicit AET cross-zone spreads/slopes in addition to learned graph messages.
- Coefficient L1 and fixed 0.8 correction shrinkage, assessed by quarter-price MAE.
- A smaller one-block graph with a direct local linear path to preserve magnitude information.
- A predeclared convex blend grid, selected only on Jan–May predictions.

`plan.json` was written before training; `selection_before_confirmation.json` before confirmation.
No candidates or weights were added after viewing confirmation. Each scaler/model uses only
earlier training dates, with strictly earlier-date history. Graph and blend outputs were validated
after CSV reload. Original graph checkpoints stay loadable using the old default configuration.

## Artifacts and reproduction

`selected_graph_through_august/` is the new graph checkpoint. If 0<graph weight<1,
`selected_blend_through_august/` contains the graph plus a copy of the official-data August-cutoff
AET checkpoint. Inference uses no fitting, calibration, or outside-trained weights.

```bash
.venv-mae/bin/python -m Paola_models.cli infer --model PATH_TO_ARTIFACT \\
  --hourly test_hourly.csv --history history_15min.csv --out NEW_OUTPUT.csv
```

Run from the ApplePi root on Paola. The CLI recognizes either artifact type. Row comparisons
are in `selected_predictions.csv`. `summary.json` records validation, gate checks, source data hash,
and unchanged AET artifact hashes. This is research, not automatic promotion or eligibility certification.
"""
    (out/"optimization_results.md").write_text(report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,default=REPO/"Data/train_prices_15min.csv")
    p.add_argument("--aet-predictions",type=Path,default=REPO/"analysis/results/development_error_check_20261017_181534/all_development_predictions.csv")
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
