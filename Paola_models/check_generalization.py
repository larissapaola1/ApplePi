"""Shorter training, longer validation; fixed recipes and no model selection.

Three frozen training cutoffs, shared July–August comparison, in-sample versus
out-of-sample error, and a no-fresh-quarter-history stress check. September is excluded.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from . import REPO
from .temporal_graph import Config, GraphReconstructor
from .cli import training_cube, markdown_table
from aet.config import load_config, final_config
from aet.pipeline import ShapePipeline

METHODS=["Original TGNN","Optimized TGNN","AET","Fixed blend","Flat"]
SCENARIOS=[("train_3_months","2025-12-31","2026-01-01"),
           ("train_6_months","2026-03-31","2026-04-01"),
           ("train_9_months","2026-06-30","2026-07-01")]


def block_interval(differences,repetitions=2000,block=7):
    """Paired daily differences; seven consecutive observed-date circular blocks."""
    values=np.asarray(differences,dtype=float)
    rng=np.random.default_rng(42)
    count=len(values)
    starts=rng.integers(0,count,size=(repetitions,int(np.ceil(count/block))))
    indices=(starts[...,None]+np.arange(block))%count
    samples=values[indices.reshape(repetitions,-1)[:,:count]].mean(axis=1)
    return np.quantile(samples,[.025,.975])


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    selected_path=REPO/"Paola_models/results/aet_informed_optimization/selection_before_confirmation.json"
    selected=json.loads(selected_path.read_text())
    optimized_cfg=Config(**selected["config"])
    original_cfg=Config()
    weight=float(selected["graph_weight"])
    assert weight==.25
    aet_cfg=final_config(load_config(REPO/"AET-HACKATHON/configs/final.yaml"))
    plan={"scenarios":[{"name":n,"train_end":e,"validation_start":s,"validation_end":"2026-08-31"}
                       for n,e,s in SCENARIOS],
          "original_graph":asdict(original_cfg),"optimized_graph":asdict(optimized_cfg),
          "AET":aet_cfg.to_dict(),"graph_blend_weight":weight,
          "fixed_common_window":["2026-07-01","2026-08-31"],
          "history_modes":["strictly earlier dates revealed daily","training data only on common July-Aug window"],
          "no_early_stopping_no_tuning":True,
          "limitation":"Previously inspected official development dates, not an untouched final test."}
    (out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    artifact_roots=[REPO/"AET-HACKATHON/artifacts_official",REPO/"AET-HACKATHON/artifacts_september_assessment",
                    REPO/"Paola_models/artifacts",REPO/"Paola_models/results/aet_informed_optimization/selected_graph_through_august",
                    REPO/"Paola_models/results/aet_informed_optimization/selected_blend_through_august"]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for root in artifact_roots for p in root.rglob('*') if p.is_file()}
    cube=training_cube(REPO/"Data/train_prices_15min.csv","2026-08-31")
    H_all=cube.hourly()
    records=[];details=[];counts=[];max_drift=0.;fit_info=[]
    for scenario,cutoff,start in SCENARIOS:
        tick=time.perf_counter()
        train=cube.until(cutoff)
        validation=(cube.dates>=np.datetime64(start))&(cube.dates<=np.datetime64("2026-08-31"))
        assert train.dates.max()<cube.dates[validation].min()
        counts.append({"scenario":scenario,"training_start":str(train.dates[0]),"training_end":cutoff,
                       "training_dates":len(train.dates),"validation_start":start,"validation_end":"2026-08-31",
                       "validation_dates":int(validation.sum()),"validation_quarter_prices":int(validation.sum()*384)})
        original=GraphReconstructor(original_cfg).fit(train)
        optimized=GraphReconstructor(optimized_cfg).fit(train)
        aet=ShapePipeline(aet_cfg).fit(train)
        print(f"{scenario}: fitted 2 graphs and AET on {len(train.dates)} dates",flush=True)
        stages=[("training",np.flatnonzero(cube.dates<=np.datetime64(cutoff)),cube),
                ("validation_daily_history",np.flatnonzero(validation),cube),
                ("common_window_training_history_only",np.flatnonzero(cube.dates>=np.datetime64("2026-07-01")),train)]
        for stage,indices,history in stages:
            for i in indices:
                day,H,actual=cube.dates[i],H_all[i],cube.values[i]
                g0=original.predict_day(day,H,history)
                g1=optimized.predict_day(day,H,history)
                tree,diag=aet.predict_day(day,H,history)
                assert diag["fallback_hours"].sum()==diag["flat_hours"].sum()==0
                predictions={"Original TGNN":g0,"Optimized TGNN":g1,"AET":tree,
                             "Fixed blend":weight*g1+(1-weight)*tree,
                             "Flat":np.repeat(H[...,None],4,axis=-1)}
                for method,pred in predictions.items():
                    drift=float(np.max(np.abs(pred.mean(axis=-1)-H)))
                    max_drift=max(max_drift,drift)
                    assert np.isfinite(pred).all() and drift<=.0001
                    error=np.abs(pred-actual)
                    for z,zone in enumerate(cube.zones):
                        records.append({"scenario":scenario,"stage":stage,"delivery_date":str(day),
                            "market_id":zone,"method":method,"MAE":error[z].mean(),
                            "Q1_MAE":error[z,:,0].mean(),"Q4_MAE":error[z,:,3].mean()})
                if stage=="validation_daily_history":
                    frame=cube._select(cube.dates==day).to_long()
                    frame["scenario"]=scenario
                    for method,pred in predictions.items():frame[method]=pred.reshape(-1)
                    details.append(frame)
            print(f"{scenario}: {stage}, {len(indices)} dates",flush=True)
        fit_info.append({"scenario":scenario,"seconds":time.perf_counter()-tick,
                         "original_final_logged_training_loss":original.losses[-1],
                         "optimized_final_logged_training_loss":optimized.losses[-1]})
    scores=pd.DataFrame(records)
    scores["delivery_date"]=pd.to_datetime(scores.delivery_date)
    scores.to_csv(out/"daily_zone_scores.csv",index=False)
    per_row=pd.concat(details,ignore_index=True)
    keys=["scenario","delivery_date","market_id","hour_index","quarter_index"]
    assert not per_row.duplicated(keys).any() and per_row.delivery_date.max()<pd.Timestamp("2026-09-01")
    per_row.to_csv(out/"validation_predictions_vs_actual.csv.gz",index=False,compression="gzip")
    summary=scores.groupby(["scenario","stage","method","market_id"]).MAE.mean().groupby(
        ["scenario","stage","method"]).mean().unstack("stage").reset_index()
    summary["validation_minus_training"]=summary.validation_daily_history-summary.training
    summary["validation_training_ratio"]=summary.validation_daily_history/summary.training
    flat=summary[summary.method=="Flat"].set_index("scenario")
    summary["validation_reduction_vs_flat_pct"]=[100*(1-r.validation_daily_history/flat.loc[r.scenario,"validation_daily_history"])
                                                 for r in summary.itertuples()]
    summary.to_csv(out/"training_vs_validation.csv",index=False)
    common=scores[scores.delivery_date>=pd.Timestamp("2026-07-01")]
    common=common[common.stage!="training"]
    common_scores=common.groupby(["scenario","stage","method","market_id"]).MAE.mean().groupby(
        ["scenario","stage","method"]).mean().reset_index()
    common_scores.to_csv(out/"common_july_august_scores.csv",index=False)
    monthly=scores[scores.stage=="validation_daily_history"].assign(month=lambda x:x.delivery_date.dt.strftime('%Y-%m')).groupby(
        ["scenario","month","method","market_id"]).MAE.mean().groupby(["scenario","month","method"]).mean().unstack("method")
    monthly.to_csv(out/"monthly_validation_scores.csv")
    zone=scores[scores.stage=="validation_daily_history"].groupby(["scenario","method","market_id"]).MAE.mean()
    zone.to_csv(out/"validation_zone_scores.csv")
    gains=[]
    for scenario,part in scores[scores.stage=="validation_daily_history"].groupby("scenario"):
        daily=part.groupby(["delivery_date","method"]).MAE.mean().unstack("method")
        diff=daily["Fixed blend"]-daily.AET
        lo,hi=block_interval(diff.to_numpy())
        gains.append({"scenario":scenario,"validation_dates":len(daily),
                      "blend_minus_AET_MAE":diff.mean(),"gain_vs_AET_pct":100*(1-daily["Fixed blend"].mean()/daily.AET.mean()),
                      "fraction_dates_blend_better":float((diff<0).mean()),
                      "paired_daily_block95_low":lo,"paired_daily_block95_high":hi})
    gains=pd.DataFrame(gains);gains.to_csv(out/"blend_gain_uncertainty.csv",index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    meta={"plan":plan,"counts":counts,"validation_rows_total_overlapping_scenarios":len(per_row),
          "unique_validation_dates":per_row.delivery_date.nunique(),"max_hourly_mean_error":max_drift,
          "artifact_hashes_unchanged":hashes,"timings":fit_info,"model_selection_performed":False}
    (out/"summary.json").write_text(json.dumps(meta,indent=2,default=str)+"\n")
    draw(out,scores,summary,common_scores)
    history_table=common_scores.pivot(index=["scenario","method"],columns="stage",values="MAE").reset_index()
    basic=summary[["scenario","method","training","validation_daily_history","validation_minus_training","validation_reduction_vs_flat_pct"]]
    report=f"""# Shorter training and longer validation: generalization check

This test keeps both neural recipes, the AET recipe, and the **.25 graph / .75 AET blend weight fixed**.
No early stopping, hyperparameter search, blending recalibration, September scoring, or model
promotion occurred. Each fitted model stays fixed for its entire validation window.

## Splits declared before fitting

{markdown_table(pd.DataFrame(counts))}

The longest split validates every available January–August day rather than just the 19th onward.
Scenarios overlap; never pool their predictions as independent validation cases. There are
{per_row.delivery_date.nunique()} distinct validation dates overall, and all four zones have equal weight.

## In-sample versus out-of-sample MAE (EUR/MWh)

{markdown_table(basic)}

Training MAE is recomputed in inference/eval mode on fitted observations, with strictly earlier
history for each date. It is not the logged stochastic training loss. Validation is out of sample
relative to that scenario's weight/scaler fit. Day-ahead hourly references remain permitted inputs.

![Training and validation](training_validation.png)

The gap alone does not prove overfitting. The target's flat-baseline error also changes with season,
and the earliest cutoffs see fewer seasonal regimes. Reduced relative improvement over flat,
large train/validation gaps, and sensitivity to cutoff together are warning signs. A good validation
result likewise cannot prove the absence of overfitting, particularly on previously inspected dates.

## Same July–August observations for every training size

{markdown_table(history_table)}

This avoids comparing a winter validation window directly with a summer one. More training here
also means more recent training, so data quantity and recency are not causally isolated.
The `validation_daily_history` mode supplies only past observations, including earlier validation
dates, as frozen-rule inputs. It does not update weights, scalers, or the blend coefficient.
The `common_window_training_history_only` mode withholds all post-cutoff quarter prices; it is a
stress scenario, not the normal jury replay when fresh prior history is supplied.

## Does the blend help consistently?

{markdown_table(gains)}

Negative `blend_minus_AET_MAE` favors the blend. The 95% interval resamples paired daily
differences using seven consecutive observed-date blocks, preserving all zones together.
It is descriptive uncertainty under a block-resampling assumption, not proof of significance;
if the interval includes zero, a stable blend advantage is not established by this diagnostic.

![Validation over time](validation_over_time.png)

## How to interpret this test

- The historical below-5 September score is specific to its original August-trained models and
  that September window. There is no promise of staying below 5 with much less training.
- The pure graph should be compared with AET on the exact same observations, and the blend with
  AET as well as the original graph. A blend can mask a weak neural component through its 75% AET share.
- Weights were frozen for up to eight months: degradation can reflect stale models, seasonal
  distribution shift, or overfitting. This experiment cannot uniquely identify their contributions.
- All months here have already featured in EDA or previous model development. New split boundaries
  do not make them an untouched independent test. Strong final claims still need organizer-held dates.

## Files and reproducibility

`training_vs_validation.csv`, `common_july_august_scores.csv`, `monthly_validation_scores.csv`,
`validation_zone_scores.csv`, and `blend_gain_uncertainty.csv` contain the summaries.
`daily_zone_scores.csv` retains every scenario/date/zone/method error.
`validation_predictions_vs_actual.csv.gz` contains every observed and predicted quarter value;
gzip saves disk space and pandas reads it directly. `plan.json` precedes fitting; `summary.json`
records data counts, timings, and unchanged-artifact hashes. Models were fitted only in memory,
and no existing checkpoint was replaced. Maximum hourly-average error: {max_drift:.3g} EUR/MWh.

```bash
.venv-mae/bin/python -m Paola_models.check_generalization --out Paola_models/results/NEW_generalization_check
```
"""
    (out/"generalization_report.md").write_text(report.replace(
        "## Splits declared before fitting", "[Read the main findings and recommendation](findings.md).\n\n## Splits declared before fitting"))
    interpret(out)
    print("\nCOUNTS\n"+pd.DataFrame(counts).to_string(index=False))
    print("\nTRAIN / VALIDATION\n"+basic.to_string(index=False))
    print("\nCOMMON WINDOW\n"+history_table.to_string(index=False))
    print("\nBLEND GAINS\n"+gains.to_string(index=False))
    print("REPORT",out/"generalization_report.md",flush=True)


def draw(out,scores,summary,common):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/applepi-error-analysis-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,(name,part) in zip(axes,summary.groupby("scenario",sort=False)):
        part=part.set_index("method").loc[METHODS]
        x=np.arange(len(part))
        ax.bar(x-.18,part.training,width=.36,label="Training (in-sample)")
        ax.bar(x+.18,part.validation_daily_history,width=.36,label="Long validation")
        ax.set(xticks=x,xticklabels=METHODS,title=name,ylabel="MAE (EUR/MWh)")
        ax.tick_params(axis='x',rotation=30)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"training_validation.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9))
    for ax,(scenario,part) in zip(axes,scores[scores.stage=="validation_daily_history"].groupby("scenario",sort=False)):
        daily=part.groupby(["delivery_date","method"]).MAE.mean().unstack("method")
        rolling=daily.rolling(14,min_periods=1).mean()
        for method in ["Original TGNN","Optimized TGNN","AET","Fixed blend"]:
            ax.plot(rolling.index,rolling[method],label=method)
        ax.set(title=scenario+": trailing 14-observation-date mean",ylabel="MAE (EUR/MWh)")
        ax.legend(ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig(out/"validation_over_time.png",dpi=160);plt.close(fig)


def interpret(out):
    """Explain the completed run without fitting or selecting anything else."""
    summary=pd.read_csv(out/"training_vs_validation.csv")
    counts=pd.DataFrame(json.loads((out/"summary.json").read_text())["counts"])
    gains=pd.read_csv(out/"blend_gain_uncertainty.csv")
    common=pd.read_csv(out/"common_july_august_scores.csv")
    validation=summary.pivot(index="scenario",columns="method",values="validation_daily_history").reset_index()
    comparison=counts[["scenario","training_dates","validation_dates"]].merge(validation,on="scenario")
    counts.merge(validation,on="scenario").to_csv(out/"split_comparison.csv",index=False)
    short=summary[summary.scenario=="train_3_months"].set_index("method")
    matched=common[common.stage=="validation_daily_history"].pivot(index="scenario",columns="method",values="MAE").reset_index()
    negatives=int((gains.blend_minus_AET_MAE<0).sum())
    text=f"""# Findings: does the model generalize with less training?

**The optimized model generalizes better than the original graph in all three scenarios.
The 25% graph / 75% AET blend improves slightly over AET in {negatives}/3 scenarios.**
This is encouraging evidence across a much longer period, not proof that overfitting is absent.
All four zones retain equal weight, and settings were fixed before this diagnostic.

## All validation results, EUR/MWh

{markdown_table(comparison[['training_dates','validation_dates','Original TGNN','Optimized TGNN','AET','Fixed blend','Flat']])}

The longest run trained on October–December (91 valid dates) and validated every available
January–August date (242 dates, 92,928 quarter-price observations). The older monthly evaluation
covered 98 dates. The shorter validation windows overlap the longer one, so they are not
independent replications and their row counts should not be added to claim extra independent data.

## Overfitting warning in the original graph

For the 91-day training run:

- Original graph: training **{short.loc['Original TGNN','training']:.3f}**, validation **{short.loc['Original TGNN','validation_daily_history']:.3f}**.
- Optimized graph: training **{short.loc['Optimized TGNN','training']:.3f}**, validation **{short.loc['Optimized TGNN','validation_daily_history']:.3f}**.
- AET: training **{short.loc['AET','training']:.3f}**, validation **{short.loc['AET','validation_daily_history']:.3f}**.

The original graph fits training more closely yet predicts later dates less accurately than the
smaller optimized graph. That pattern is consistent with overfitting or less stable extrapolation.
It supports keeping the simpler graph rather than pursuing a lower training loss. It does not
identify overfitting as the sole cause: calendar/price regimes and model age change too.
Even flat-baseline error changes between training and validation, showing different target difficulty.

## More training helps on the same July–August dates

{markdown_table(matched[['scenario','Original TGNN','Optimized TGNN','AET','Fixed blend']])}

All rows here predict identical target dates. The three models use progressively later training
cutoffs, so more data also means fresher data. These effects cannot be separated by this test.
The separate training-history-only replay in the full report shows that fresh earlier-date history
usually helps the optimized model and blend; it is a legitimate input, not a weight update.
The original graph is less stable: fresh lag inputs do not improve it under every cutoff.

## How strong is the blend advantage?

{markdown_table(gains)}

The observed gains are about **{gains.gain_vs_AET_pct.min():.2f}%–{gains.gain_vs_AET_pct.max():.2f}%**,
still below the existing 0.5% promotion requirement. Paired daily seven-observation block-bootstrap
intervals are reported, but depend on resampling assumptions and repeated use of development data.
They cannot certify an independent final-test advantage.

## Recommendation

Keep AET as the main baseline and retain the smaller graph/blend as research candidates.
The blend's small edge is more credible than a single 12-day September number because it repeats
over longer fixed-weight windows, but it does not justify promising a sub-5 score on every period.
Use future organizer-held dates for final evidence; do not reduce training merely because one
historical window happens to score lower. Model age, seasonal coverage, and baseline difficulty matter.

No September/October assessment was run here, no settings were retuned, and no saved model
artifacts changed. All dates had been inspected in prior project work, so these redesigned splits
are diagnostic validation, not a newly untouched holdout.

[Full protocol, history stress tests, charts, and CSVs](generalization_report.md).
"""
    (out/"findings.md").write_text(text)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=REPO/"Paola_models/results/long_validation_check")
    main(p.parse_args().out)
