"""Assess the two frozen AET error corrections on Sep 19–30; no fitting or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import REPO
from .cli import markdown_table
from .residual_graph import AETResidualReconstructor
from .temporal_graph import ZONES
from aet.cube import QuarterCube
from aet.infer import run as run_aet
from aet.schema import QH_KEYS,read_hourly,read_quarter_hourly,validate_predictions
from tools.score import score


def main(out):
    source=REPO/'analysis/results/mae_evaluation_20261017_175448'
    study=REPO/'Paola_models/results/aet_residual_graph_study'
    prior=REPO/'Paola_models/results/september_19_30_assessment'
    aet_path=REPO/'AET-HACKATHON/artifacts_september_assessment'
    selection=json.loads((study/'selection_before_confirmation.json').read_text())
    paths={name:study/f'artifact_{name}_through_august' for name in ['graph','no_message']}
    protected=[*paths.values(),aet_path,REPO/'AET-HACKATHON/artifacts_official',
               REPO/'Paola_models/results/aet_informed_optimization/selected_blend_through_august']
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in protected for p in folder.rglob('*') if p.is_file()}
    inputs={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ['test_hourly.csv','history_15min.csv','truth_15min.csv']}
    previous=json.loads((REPO/'Paola_models/results/frozen_hybrid_september/summary.json').read_text())
    assert inputs==previous['input_hashes']
    hourly=read_hourly(source/'test_hourly.csv')
    assert hourly.delivery_date.min()==pd.Timestamp('2026-09-19')
    assert hourly.delivery_date.max()==pd.Timestamp('2026-09-30')
    assert hourly.delivery_date.nunique()==12 and len(hourly)==1152 and set(hourly.market_id)==set(ZONES)
    history=QuarterCube.from_long(read_quarter_hourly(source/'history_15min.csv',strict=False),ZONES)
    out.mkdir(parents=True,exist_ok=False)
    plan={'period':['2026-09-19','2026-09-30'],'variants':['graph','no_message'],
          'selection':selection,'rule':'load both saved artifacts unchanged; no September strength search or fitting',
          'input_hashes':inputs,'original_artifact_hashes':hashes}
    (out/'frozen_assessment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    frames={};model_metadata={}
    labels={'graph':'AET + graph error correction','no_message':'AET + correction without graph messages'}
    for name,path in paths.items():
        model=AETResidualReconstructor.load(path)
        recipe=selection['candidates'][name]
        assert model.training_range==['2025-10-01','2026-08-31']
        assert model.strength==recipe['strength']==.25
        assert model.correction.config.graph==(name=='graph')
        assert model.correction.config.__dict__==recipe['config']
        assert model.correction.oof_range[-1]=='2026-08-31'
        for teacher_file in ['lgbm_coef_0.txt','lgbm_coef_1.txt','lgbm_coef_2.txt','history_store.npz']:
            assert hashlib.sha256((path/'aet'/teacher_file).read_bytes()).hexdigest()==hashlib.sha256((aet_path/teacher_file).read_bytes()).hexdigest()
        pred=model.predict_frame(hourly,model.training_history.merge(history))
        validate_predictions(pred,hourly)
        target=out/f'{name}_correction_predictions.csv'
        pred.to_csv(target,index=False,float_format='%.10f')
        frames[labels[name]]=pd.read_csv(target,parse_dates=['delivery_date'])
        model_metadata[name]={'artifact':str(path.relative_to(REPO)),'strength':model.strength,
            'training_range':model.training_range,'oof_range':model.correction.oof_range,
            'oof_dates':model.correction.oof_days}
        print(f'{name}: frozen predictions saved',flush=True)
    diag=run_aet(source/'test_hourly.csv',source/'history_15min.csv',aet_path,out/'aet_predictions.csv')
    assert not diag.get('emergency_flat',False) and diag['fallback_hours']==diag['flat_hours']==0
    aet=pd.read_csv(out/'aet_predictions.csv',parse_dates=['delivery_date'])
    old_graph=pd.read_csv(prior/'optimized_graph_predictions.csv',parse_dates=['delivery_date'])
    old_blend=pd.read_csv(prior/'frozen_blend_predictions.csv',parse_dates=['delivery_date'])
    # All predictions above precede label loading/scoring; no assessment-dependent adjustment.
    truth=read_quarter_hourly(source/'truth_15min.csv')
    assert len(truth)==4608 and truth.delivery_date.nunique()==12
    assert truth.delivery_date.min()==pd.Timestamp('2026-09-19')
    assert truth.delivery_date.max()==pd.Timestamp('2026-09-30')
    assert set(truth.market_id)==set(ZONES) and truth.groupby(['delivery_date','market_id']).size().eq(96).all()
    methods={'AET':aet,'Previous 25% graph / 75% AET blend':old_blend,**frames}
    rows=truth.copy();comparisons=[];checks={};zone=[];quarter=[]
    for method,frame in methods.items():
        checks[method]=validate_predictions(frame,hourly)
        z,q,s=score(frame,truth)
        comparisons.append({'method':method,'MAE_EUR_MWh':s['score']})
        zone.append(z.model.rename(method));quarter.append(q.model.rename(method))
        rows=rows.merge(frame.rename(columns={'predicted_price_eur_mwh':method}),on=QH_KEYS,validate='one_to_one')
        rows[method+' signed error']=rows[method]-rows.price_eur_mwh
        rows[method+' absolute error']=rows[method+' signed error'].abs()
    old_check=old_graph.merge(aet,on=QH_KEYS,suffixes=('_g','_a'),validate='one_to_one').merge(old_blend,on=QH_KEYS,validate='one_to_one')
    np.testing.assert_allclose(old_check.predicted_price_eur_mwh,
        .25*old_check.predicted_price_eur_mwh_g+.75*old_check.predicted_price_eur_mwh_a,atol=2e-9,rtol=0)
    comparison=pd.DataFrame(comparisons)
    comparison['change_vs_AET_pct']=100*(comparison.MAE_EUR_MWh/comparison.iloc[0].MAE_EUR_MWh-1)
    comparison.to_csv(out/'method_comparison.csv',index=False)
    zones=pd.concat(zone,axis=1);zones.to_csv(out/'mae_by_zone.csv')
    pd.concat(quarter,axis=1).to_csv(out/'mae_by_quarter.csv')
    error_columns=[method+' absolute error' for method in methods]
    daily=rows.groupby(['delivery_date','market_id'])[error_columns].mean()
    daily.to_csv(out/'daily_zone_mae.csv')
    dates=daily.groupby('delivery_date').mean();dates.to_csv(out/'daily_mae.csv')
    rows.to_csv(out/'predictions_vs_actual.csv',index=False)
    by_date=[]
    for name in labels.values():
        by_date.append({'method':name,'dates_better_than_AET':int((dates[name+' absolute error']<dates['AET absolute error']).sum()),
                        'dates':12,'below_5':bool(comparison.loc[comparison.method==name,'MAE_EUR_MWh'].iloc[0]<5)})
    assert all(hashlib.sha256((REPO/path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==digest for name,digest in inputs.items())
    summary={'period':['2026-09-19','2026-09-30'],'dates':12,'zones':list(ZONES),'predictions_per_method':4608,
        'models':model_metadata,'locked_development_selection':selection,'new_training_or_tuning':False,
        'new_calibration':False,'selection_changed':False,'scores':comparison.to_dict('records'),
        'date_comparisons':by_date,'validation':checks,'aet_diagnostics':diag,
        'unchanged_artifact_hashes':hashes,'input_hashes':inputs}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    graph_score=comparison.loc[comparison.method==labels['graph'],'MAE_EUR_MWh'].iloc[0]
    local_score=comparison.loc[comparison.method==labels['no_message'],'MAE_EUR_MWh'].iloc[0]
    report=f'''# Frozen AET error corrections: September 19–30 assessment

**AET + graph error correction: {graph_score:.5f} EUR/MWh.**
**AET + correction without graph messages: {local_score:.5f} EUR/MWh.**

{markdown_table(comparison)}

Lower MAE is better. All methods use the same **12 dates, four equally weighted zones and
4,608 quarter-price predictions**. A negative change versus AET means lower error.
The previous blend combines two full reconstructions; the new models add an estimated AET error.

Both new formulas are **AET + 0.25 × learned correction**, with identical frozen August-cutoff
AET teachers. Both correction networks trained only on chronological out-of-fold AET errors
through August 31. Their architectures and correction strengths were chosen on January–May
and fixed before June–August confirmation. The no-message candidate was the development-selected
variant; this September replay does not change that selection. No fitting, recalibration,
strength search or candidate retuning occurred.

**By zone**

{markdown_table(zones.reset_index())}

**Dates with lower error than AET**

{markdown_table(pd.DataFrame(by_date))}

Date/zone/quarter summaries and every prediction-versus-actual signed and absolute error are saved
alongside this report. The components' weights and original assessment input hashes are unchanged.
All four point outputs passed complete-key, finite-value and hourly-mean checks after CSV reload;
maximum hourly-mean error was {max(c['max_abs_mean_error'] for c in checks.values()):.3g} EUR/MWh,
below the required 0.0001. AET used no fallback hours. No uncertainty intervals are claimed for
these deterministic corrections.

For every date, only supplied target-day hourly curves and strictly earlier quarter-price history
enter prediction features. Earlier assessment actuals may enter later dates' history, matching the
existing daily replay. September was already explored and evaluated for earlier recipes, so this
explicitly requested historical assessment is not an independent final test. No October data was read.
The result must not be used to choose a September-optimal correction strength.

Saved locally under Paola_models on branch Paola. Existing artifacts and the AET dashboard were
not changed by this assessment. No commit or push was made.

Reproduce from ApplePi with a fresh output directory:

```bash
.venv-mae/bin/python -m Paola_models.assess_residual_september --out Paola_models/results/NEW_frozen_residual_assessment
```
'''
    (out/'assessment_results.md').write_text(report)
    print('\n'+comparison.to_string(index=False),flush=True)
    print('\nBY ZONE\n'+zones.to_string(),flush=True)
    print('\nREPORT',out/'assessment_results.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPO/'Paola_models/results/aet_residual_graph_september')
    main(parser.parse_args().out)
