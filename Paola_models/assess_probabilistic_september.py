"""Frozen Sep 19–30 assessment of the adaptive Bayesian graph; no fitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import REPO
from .cli import markdown_table
from .probabilistic_cli import export
from .probabilistic_graph import ProbabilisticGraph
from .temporal_graph import ZONES
from aet.cube import QuarterCube
from aet.infer import run as run_aet
from aet.schema import QH_KEYS, read_hourly, read_quarter_hourly, validate_predictions
from tools.score import score


def main(out):
    source=REPO/'analysis/results/mae_evaluation_20261017_175448'
    study=REPO/'Paola_models/results/adaptive_probabilistic_study'
    old=REPO/'Paola_models/results/september_19_30_assessment'
    model_path=study/'artifact_clustered_bayesian'
    aet_path=REPO/'AET-HACKATHON/artifacts_september_assessment'
    choice=json.loads((study/'selection_before_confirmation.json').read_text())
    assert choice['point_candidate']=='clustered_bayesian'
    alpha=choice['blend_graph_weight']
    assert alpha==.1  # Recipe locked on January–May before confirmation or September scoring.
    protected=[model_path,aet_path,REPO/'AET-HACKATHON/artifacts_official',
               REPO/'Paola_models/results/aet_informed_optimization/selected_blend_through_august']
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in protected for p in folder.rglob('*') if p.is_file()}
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source/'test_hourly.csv',source/'history_15min.csv',source/'truth_15min.csv']}
    prior_report=json.loads((REPO/'Paola_models/results/frozen_hybrid_september/summary.json').read_text())
    assert inputs==prior_report['input_hashes']
    hourly=read_hourly(source/'test_hourly.csv')
    assert hourly.delivery_date.min()==pd.Timestamp('2026-09-19')
    assert hourly.delivery_date.max()==pd.Timestamp('2026-09-30')
    assert hourly.delivery_date.nunique()==12 and set(hourly.market_id)==set(ZONES)
    model=ProbabilisticGraph.load(model_path)
    assert model.data_cutoff=='2026-08-31' and model.weight_cutoff=='2026-08-17'
    history=model.training_history.merge(QuarterCube.from_long(
        read_quarter_hourly(source/'history_15min.csv',strict=False),ZONES))
    out.mkdir(parents=True,exist_ok=False)
    metadata=export(model,hourly,history,out/'bayesian_inference')
    diagnostics=run_aet(source/'test_hourly.csv',source/'history_15min.csv',aet_path,out/'aet_predictions.csv')
    assert not diagnostics.get('emergency_flat',False)
    assert diagnostics['fallback_hours']==diagnostics['flat_hours']==0
    def read(path):return pd.read_csv(path,parse_dates=['delivery_date'])
    graph=read(out/'bayesian_inference/point_predictions.csv')
    aet=read(out/'aet_predictions.csv')
    merged=graph.merge(aet,on=QH_KEYS,suffixes=('_graph','_aet'),validate='one_to_one')
    blend=merged[QH_KEYS].copy()
    blend['predicted_price_eur_mwh']=alpha*merged.predicted_price_eur_mwh_graph+(1-alpha)*merged.predicted_price_eur_mwh_aet
    blend.to_csv(out/'frozen_10pct_bayesian_blend_predictions.csv',index=False,float_format='%.10f')
    blend=read(out/'frozen_10pct_bayesian_blend_predictions.csv')
    # Truth is used only after frozen predictions have been generated and serialized.
    truth=read_quarter_hourly(source/'truth_15min.csv')
    assert len(truth)==4608 and truth.delivery_date.nunique()==12
    assert truth.groupby(['delivery_date','market_id']).size().eq(96).all()
    frames={'AET':aet,'Previous optimized graph':read(old/'optimized_graph_predictions.csv'),
            'Previous 25% graph blend':read(old/'frozen_blend_predictions.csv'),
            'New Bayesian graph':graph,'New frozen 10% Bayesian blend':blend}
    rows=truth.copy();comparisons=[];zone=[];quarter=[];checks={}
    for name,frame in frames.items():
        checks[name]=validate_predictions(frame,hourly)
        z,q,s=score(frame,truth)
        comparisons.append({'method':name,'MAE_EUR_MWh':s['score']})
        zone.append(z.model.rename(name));quarter.append(q.model.rename(name))
        rows=rows.merge(frame.rename(columns={'predicted_price_eur_mwh':name}),on=QH_KEYS,validate='one_to_one')
        rows[name+' signed error']=rows[name]-rows.price_eur_mwh
        rows[name+' absolute error']=rows[name+' signed error'].abs()
    np.testing.assert_allclose(rows['Previous 25% graph blend'],
        .25*rows['Previous optimized graph']+.75*rows.AET,atol=2e-9,rtol=0)
    comparison=pd.DataFrame(comparisons)
    comparison['change_vs_AET_pct']=100*(comparison.MAE_EUR_MWh/comparison.iloc[0].MAE_EUR_MWh-1)
    comparison.to_csv(out/'method_comparison.csv',index=False)
    zones=pd.concat(zone,axis=1);zones.to_csv(out/'mae_by_zone.csv')
    pd.concat(quarter,axis=1).to_csv(out/'mae_by_quarter.csv')
    error_cols=[name+' absolute error' for name in frames]
    daily=rows.groupby(['delivery_date','market_id'])[error_cols].mean()
    daily.to_csv(out/'daily_zone_mae.csv')
    daily.groupby('delivery_date').mean().to_csv(out/'daily_mae.csv')
    rows.to_csv(out/'predictions_vs_actual.csv',index=False)
    uncertainty=read(out/'bayesian_inference/uncertainty.csv').merge(truth,on=QH_KEYS,validate='one_to_one')
    uncertainty['calibrated90_inside']=uncertainty.price_eur_mwh.between(uncertainty.lower90,uncertainty.upper90)
    uncertainty['interval_width']=uncertainty.upper90-uncertainty.lower90
    uncertainty['interval_score90']=uncertainty.interval_width+20*np.maximum(uncertainty.lower90-uncertainty.price_eur_mwh,0)+20*np.maximum(uncertainty.price_eur_mwh-uncertainty.upper90,0)
    with np.load(out/'bayesian_inference/scenarios.npz',allow_pickle=False) as archive:
        samples=archive['prices'];dates=archive['dates']
    truth_cube=QuarterCube.from_long(truth,ZONES)
    assert np.array_equal(dates,truth_cube.dates)
    assert np.max(np.abs(samples.mean(-1)-truth_cube.hourly()[:,None]))<.0001
    S=samples.shape[1];ordered=np.sort(samples,axis=1)
    crps=np.abs(samples-truth_cube.values[:,None]).mean(1)-np.sum(
        (2*np.arange(1,S+1)-S-1)[None,:,None,None,None]*ordered,axis=1)/S**2
    rawlo,rawhi=np.quantile(samples,[.05,.95],axis=1)
    raw=truth_cube.to_long()
    raw['CRPS']=crps.reshape(-1)
    raw['raw90_inside']=((truth_cube.values>=rawlo)&(truth_cube.values<=rawhi)).reshape(-1)
    uncertainty=uncertainty.merge(raw[QH_KEYS+['CRPS','raw90_inside']],on=QH_KEYS,validate='one_to_one')
    uncertainty.to_csv(out/'uncertainty_vs_actual.csv',index=False)
    uq=uncertainty.groupby('market_id')[['raw90_inside','calibrated90_inside','interval_width','interval_score90','CRPS']].mean()
    uq['whole_zone_day_inside']=uncertainty.groupby(['delivery_date','market_id']).calibrated90_inside.all().groupby('market_id').mean()
    uq.to_csv(out/'uncertainty_by_zone.csv')
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    summary={'period':['2026-09-19','2026-09-30'],'dates':12,'zones':list(ZONES),'rows':4608,
        'model':str(model_path.relative_to(REPO)),'weight_cutoff':model.weight_cutoff,'data_cutoff':model.data_cutoff,
        'locked_selection':choice,'new_training_or_selection':False,'new_calibration':False,
        'scores':comparisons,'uncertainty_equal_zone_average':uq.mean().to_dict(),
        'prediction_checks':checks,'inference_metadata':metadata,'aet_diagnostics':diagnostics,
        'unchanged_artifact_hashes':hashes,'input_hashes':inputs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    report=f'''# Frozen Bayesian graph: September 19–30 assessment

{markdown_table(comparison)}

MAE is EUR/MWh; lower is better. These are the same 12 dates, four equally weighted zones,
and 4,608 quarter-hour predictions as the previous assessment. The new standalone model is
the adaptive clustered Bayesian graph. Its 10% graph / 90% AET blend was selected on January–May,
before June–August confirmation. The previous 25% blend uses the former optimized graph.
No September fitting, calibration, model selection or blend-weight search occurred.

## Zone MAEs

{markdown_table(zones.reset_index())}

## Bayesian graph uncertainty

{markdown_table(uq.reset_index())}

Nominal marginal coverage is 90%. Raw intervals come from Bayesian predictive samples;
calibrated bands use the calibration frozen on August 18–31, with network weights trained
only through August 17. AET's assessment weights end August 31. Calibrated coverage should
be considered together with width and interval score. CRPS evaluates the raw scenarios.
These marginal intervals are not simultaneous whole-day guarantees; raw negative-price
probabilities are not separately calibrated. Uncertainty applies to the graph, not either blend.

## Interpretation and audit

This is the explicitly requested historical assessment. September was already EDA-exposed
and assessed for earlier recipes; the result is not a new independent test. It must not be
used to retune the model or select a new September-optimal blend.

The supplied target-day hourly curves are valid inputs. Quarter-price history is sliced
strictly before each target date; earlier assessment actuals can enter later-date history.
All saved point outputs and raw scenarios preserve the supplied hourly means. No October
data was read. Protected artifacts and the original input files match their recorded hashes.
All files are isolated under Paola_models on branch Paola; no commits or pushes were made.

`predictions_vs_actual.csv` contains every model prediction, actual, signed error and absolute
error. Zone/quarter/date summaries, uncertainty comparisons and frozen graph/scenario exports
are saved alongside this report. Settings, hashes and validation details are in `summary.json`.

Reproduce from ApplePi using a fresh output directory:

```bash
.venv-mae/bin/python -m Paola_models.assess_probabilistic_september --out Paola_models/results/NEW_frozen_bayesian_assessment
```
'''
    (out/'assessment_results.md').write_text(report)
    print(comparison.to_string(index=False),flush=True)
    print('\nUNCERTAINTY\n'+uq.to_string(),flush=True)
    print('\nREPORT',out/'assessment_results.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPO/'Paola_models/results/adaptive_probabilistic_september')
    main(parser.parse_args().out)
