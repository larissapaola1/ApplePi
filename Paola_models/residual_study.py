"""Bounded chronological stacking experiment: AET plus graph error correction."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import REPO
from .cli import training_cube,markdown_table
from .optimize import official
from .residual_graph import (ResidualConfig,ResidualCorrection,OOFPredictions,
                            generate_teacher_block,AETResidualReconstructor)
from .temporal_graph import ZONES
from aet.config import load_config,final_config,folds_from_config
from aet.pipeline import ShapePipeline
from aet.schema import QH_KEYS,HOUR_KEYS,validate_predictions


class TeacherBank:
    """Monthly expanding fits, persisted once and shared across outer-fold candidates."""
    def __init__(self,cube,config,out):
        self.cube,self.config,self.out=cube,config,out
        self.blocks=[];self.ledger=[]

    def ensure(self,through):
        through=pd.Timestamp(through)
        if through>pd.Timestamp('2026-08-31') or through.day!=through.days_in_month:
            raise ValueError('study OOF cutoff must be a month end through August')
        completed={str(b.cube.dates[-1])[:7] for b in self.blocks}
        for period in pd.period_range('2025-11',through.strftime('%Y-%m'),freq='M'):
            if str(period) in completed:continue
            target_end=period.end_time.date().isoformat()
            train_end=(period.start_time-pd.Timedelta(days=1)).date().isoformat()
            tick=time.perf_counter()
            block,teacher,train=generate_teacher_block(self.cube,train_end,target_end,self.config)
            path=self.path(train_end)
            teacher.save(path,train,{'purpose':'chronological OOF teacher; future target labels excluded'})
            self.blocks.append(block)
            self.ledger.append({'teacher_training_start':str(train.dates[0]),'teacher_training_end':str(train.dates[-1]),
                'teacher_training_days':len(train.dates),'target_start':str(block.cube.dates[0]),
                'target_end':str(block.cube.dates[-1]),'target_days':len(block.cube.dates),
                'max_training_date_before_min_target':bool(train.dates[-1]<block.cube.dates[0]),
                'checkpoint':str(path.relative_to(self.out)),'seconds':time.perf_counter()-tick})
            pd.DataFrame(self.ledger).to_csv(self.out/'teacher_oof_ledger.csv',index=False)
            print(f'OOF {period}: teacher through {train_end}, {len(block.cube.dates)} target dates',flush=True)
        bundle=OOFPredictions(
            self.blocks[0].cube.__class__(np.concatenate([b.cube.dates for b in self.blocks]),ZONES,
                                         np.concatenate([b.cube.values for b in self.blocks])),
            np.concatenate([b.teacher_end for b in self.blocks]),np.concatenate([b.teacher_days for b in self.blocks]))
        bundle.save(self.out/f'oof_through_{through.date().isoformat()}.npz')
        bundle.to_frame().to_csv(self.out/f'aet_oof_through_{through.date().isoformat()}.csv.gz',index=False,compression='gzip')
        return bundle

    def path(self,cutoff):
        return self.out/'teacher_models'/f'through_{str(cutoff)}'


def evaluate(model,cube,fold,diagnose=False):
    frames=[];edges=[];ablations=[]
    for i in np.flatnonzero((cube.dates>=fold.score_start)&(cube.dates<=fold.score_end)):
        day,H,truth=cube.dates[i],cube.hourly()[i],cube.values[i]
        point,base,delta,attention=model.predict_day(day,H,cube,return_components=True)
        if not np.isfinite(point).all() or np.max(np.abs(point.mean(-1)-H))>.0001:
            raise ValueError('non-compliant stacked prediction')
        f=cube._select(cube.dates==day).to_long();f['fold']=fold.name
        f['aet_prediction']=base.reshape(-1);f['raw_correction']=delta.reshape(-1)
        f['raw_corrected_prediction']=(base+delta).reshape(-1)
        f['predicted_price_eur_mwh']=point.reshape(-1)
        frames.append(f)
        if diagnose:
            for h in range(24):
                for t in range(4):
                    for s in range(4):edges.append({'delivery_date':str(day),'hour_index':h+1,
                        'target':ZONES[t],'source':ZONES[s],'attention':attention[h,t,s],
                        'correction_strength':model.strength,'fold':fold.name})
            for t in range(4):
                for s in range(4):
                    if s==t:continue
                    masked,_=model.correction.predict_day(day,H,cube,base,blocked_edge=(t,s))
                    without=base+model.strength*masked
                    ablations.append({'delivery_date':str(day),'source':ZONES[s],'target':ZONES[t],
                        'MAE_increase_when_edge_removed':np.abs(without[t]-truth[t]).mean()-np.abs(point[t]-truth[t]).mean(),
                        'mean_prediction_change':np.abs(without[t]-point[t]).mean(),'fold':fold.name})
    return pd.concat(frames,ignore_index=True),edges,ablations


def validation(frame,column):
    hourly=frame.groupby(HOUR_KEYS,as_index=False).price_eur_mwh.mean().rename(columns={'price_eur_mwh':'hourly_price_eur_mwh'})
    return validate_predictions(frame[QH_KEYS+[column]].rename(columns={column:'predicted_price_eur_mwh'}),hourly)


def main(args):
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    cfg=ResidualConfig();candidates={'graph':cfg,'no_message':replace(cfg,graph=False)}
    strengths=[0,.1,.25,.5,1.]
    raw=load_config(REPO/'AET-HACKATHON/configs/final.yaml')
    teacher_cfg=final_config(raw);folds,_=folds_from_config(raw,'monthly')
    search,confirmation=folds[:5],folds[5:]
    plan={'candidate_configs':{k:asdict(v) for k,v in candidates.items()},'correction_strengths':strengths,
          'teacher_config':teacher_cfg.to_dict(),'teacher_oof_start':'2025-11-01',
          'teacher_rule':'fit official AET through preceding month end; predict all valid next-month dates',
          'outer_training_rule':'correction labels and scalers only from OOF dates through the outer training cutoff',
          'selection_folds':[f.name for f in search],'confirmation_folds':[f.name for f in confirmation],
          'selection_rule':'minimum Jan-May price MAE per candidate; ties prefer smaller correction, including zero',
          'no_confirmation_retuning':True,'no_September_or_October_scoring':True,
          'limitations':'previously inspected development dates; one seed; not an independent final test'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    protected=[REPO/'AET-HACKATHON/artifacts_official',REPO/'AET-HACKATHON/artifacts_september_assessment',
               REPO/'Paola_models/artifacts',REPO/'Paola_models/results/aet_informed_optimization/selected_blend_through_august',
               REPO/'Paola_models/results/adaptive_probabilistic_study/artifact_clustered_bayesian']
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in protected for p in folder.rglob('*') if p.is_file()}
    cube=training_cube(args.train,'2026-08-31')
    reference_path=REPO/'Paola_models/results/aet_informed_optimization/selected_predictions.csv'
    reference=pd.read_csv(reference_path,parse_dates=['delivery_date'])
    reference=reference[QH_KEYS+['fold','aet_prediction','blend_prediction','predicted_price_eur_mwh']].rename(
        columns={'aet_prediction':'previous_aet_prediction','blend_prediction':'previous_blend_prediction',
                 'predicted_price_eur_mwh':'previous_graph_prediction'})
    bank=TeacherBank(cube,teacher_cfg,out)
    # No June-August teacher outputs or correction labels are generated until the choice is frozen.
    search_oof=bank.ensure('2026-05-31')
    ledger=[];selection_frames={};trials=[];cutoffs=[]
    for name,config in candidates.items():
        parts=[];tick=time.perf_counter()
        for fold in search:
            training=cube.until(fold.train_end);oof=search_oof.until(fold.train_end)
            correction=ResidualCorrection(config).fit(training,oof)
            teacher,history=ShapePipeline.load(bank.path(fold.train_end))
            model=AETResidualReconstructor(correction,teacher,history,1.)
            part,_,_=evaluate(model,cube,fold);parts.append(part)
            cutoffs.append({'candidate':name,'fold':fold.name,'correction_training_end':str(training.dates[-1]),
                'oof_label_start':correction.oof_range[0],'oof_label_end':correction.oof_range[1],
                'oof_label_dates':correction.oof_days,'base_teacher_training_end':teacher.training_range[-1]})
            print(f'selection {name} {fold.name} completed',flush=True)
        f=pd.concat(parts,ignore_index=True).merge(reference,on=QH_KEYS+['fold'],validate='one_to_one')
        selection_frames[name]=f
        f.to_csv(out/f'selection_{name}_predictions.csv.gz',index=False,compression='gzip')
        ledger.append({'candidate':name,'raw_correction_MAE':official(f),'seconds':time.perf_counter()-tick})
        for strength in strengths:
            trial=f.assign(trial=f.aet_prediction+strength*f.raw_correction)
            trials.append({'candidate':name,'strength':strength,'selection_MAE':official(trial,'trial')})
        pd.DataFrame(trials).to_csv(out/'selection_strength_trials.csv',index=False)
        print(name,'selection raw correction MAE',official(f),flush=True)
    choices={}
    for name,config in candidates.items():
        chosen=min((r for r in trials if r['candidate']==name),key=lambda r:r['selection_MAE'])
        choices[name]={'config':asdict(config),**chosen}
    winner=min(choices,key=lambda k:choices[k]['selection_MAE'])
    selection={'candidates':choices,'selected_candidate':winner,
               'selection_folds':[f.name for f in search],'rule':plan['selection_rule']}
    (out/'selection_before_confirmation.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('LOCKED',selection,flush=True)
    full_oof=bank.ensure('2026-08-31')
    confirmation_frames={};graphs=[];edge_removal=[]
    for name,config in candidates.items():
        parts=[]
        for fold in confirmation:
            training=cube.until(fold.train_end);oof=full_oof.until(fold.train_end)
            correction=ResidualCorrection(config).fit(training,oof)
            teacher,history=ShapePipeline.load(bank.path(fold.train_end))
            model=AETResidualReconstructor(correction,teacher,history,choices[name]['strength'])
            part,edge,ablations=evaluate(model,cube,fold,diagnose=name=='graph')
            parts.append(part);graphs.extend(edge);edge_removal.extend(ablations)
            cutoffs.append({'candidate':name,'fold':fold.name,'correction_training_end':str(training.dates[-1]),
                'oof_label_start':correction.oof_range[0],'oof_label_end':correction.oof_range[1],
                'oof_label_dates':correction.oof_days,'base_teacher_training_end':teacher.training_range[-1]})
            print(f'confirmation {name} {fold.name} completed',flush=True)
        confirmation_frames[name]=pd.concat(parts,ignore_index=True).merge(reference,on=QH_KEYS+['fold'],validate='one_to_one')
    stage_scores=[];fold_scores=[];checks={};zone_tables=[]
    for stage,frames in [('selection_Jan_May',selection_frames),('confirmation_Jun_Aug',confirmation_frames)]:
        base=next(iter(frames.values()))
        for method,col in [('AET','aet_prediction'),('Previous fixed blend','previous_blend_prediction'),
                           ('Previous optimized graph','previous_graph_prediction')]:
            stage_scores.append({'stage':stage,'method':method,'MAE':official(base,col)})
        for name,frame in frames.items():
            frame=frame.copy()
            frame['selected_prediction']=frame.aet_prediction+choices[name]['strength']*frame.raw_correction
            frame['stage']=stage
            frame.to_csv(out/f'{stage}_{name}_predictions.csv.gz',index=False,compression='gzip')
            for method,col in [(name+' raw correction','raw_corrected_prediction'),(name+' selected correction','selected_prediction')]:
                stage_scores.append({'stage':stage,'method':method,'MAE':official(frame,col)})
                checks[stage+' '+method]=validation(frame,col)
            for fold,f in frame.groupby('fold'):
                fold_scores.append({'stage':stage,'fold':fold,'candidate':name,'AET_MAE':official(f,'aet_prediction'),
                                    'raw_MAE':official(f,'raw_corrected_prediction'),'selected_MAE':official(f,'selected_prediction')})
            if stage=='confirmation_Jun_Aug':
                zone=frame.assign(aet_error=(frame.aet_prediction-frame.price_eur_mwh).abs(),
                    corrected_error=(frame.selected_prediction-frame.price_eur_mwh).abs()).groupby(
                        ['delivery_date','market_id'])[['aet_error','corrected_error']].mean().groupby('market_id').mean()
                zone['candidate']=name;zone_tables.append(zone.reset_index())
    scores=pd.DataFrame(stage_scores);scores.to_csv(out/'stage_scores.csv',index=False)
    monthly=pd.DataFrame(fold_scores);monthly.to_csv(out/'fold_scores.csv',index=False)
    zones=pd.concat(zone_tables,ignore_index=True);zones.to_csv(out/'confirmation_zone_scores.csv',index=False)
    pd.DataFrame(ledger).to_csv(out/'candidate_runtime.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(out/'outer_training_ledger.csv',index=False)
    pd.DataFrame(graphs).to_csv(out/'confirmation_hourly_graph.csv',index=False)
    edges=pd.DataFrame(edge_removal)
    edges.to_csv(out/'confirmation_edge_ablation.csv',index=False)
    edge_summary=edges.groupby(['source','target'])[['MAE_increase_when_edge_removed','mean_prediction_change']].mean()
    edge_summary.to_csv(out/'confirmation_edge_summary.csv')
    selected_months=monthly[(monthly.stage=='confirmation_Jun_Aug')&(monthly.candidate==winner)]
    selected_zones=zones[zones.candidate==winner]
    gain=1-official(confirmation_frames[winner])/official(confirmation_frames[winner],'aet_prediction')
    wins=int((selected_months.selected_MAE<selected_months.AET_MAE).sum())
    worst=float((selected_zones.corrected_error/selected_zones.aet_error-1).max())
    promoted=bool(gain>=.005 and wins>=2 and worst<=.02)
    # Both isolated artifacts retain their selection-period strengths, including zero if selected.
    final_teacher_path=REPO/'AET-HACKATHON/artifacts_september_assessment'
    final_teacher,history=ShapePipeline.load(final_teacher_path)
    if final_teacher.config.to_dict()!=teacher_cfg.to_dict():raise ValueError('August teacher recipe differs from OOF recipe')
    if not np.array_equal(history.dates,cube.dates) or not np.allclose(history.values,cube.values,atol=1e-10,rtol=0):
        raise ValueError('August teacher history differs from official correction training data')
    for name,config in candidates.items():
        correction=ResidualCorrection(config).fit(cube,full_oof)
        model=AETResidualReconstructor(correction,final_teacher,history,choices[name]['strength'])
        model.save(out/f'artifact_{name}_through_august',final_teacher_path,selection)
        print('saved',name,'artifact',flush=True)
    assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    summary={'selection':selection,'confirmation_relative_gain_vs_AET':float(gain),'confirmation_folds_won':wins,
             'confirmation_worst_zone_degradation':worst,'confirmation_gate_passed':promoted,
             'September_scored':False,'new_incumbent_replaced':False,'validation':checks,
             'original_artifact_hashes_unchanged':hashes,
             'training_data_sha256':hashlib.sha256(args.train.read_bytes()).hexdigest(),
             'reference_prediction_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
             'max_recomputed_AET_difference':float(max((f.aet_prediction-f.previous_aet_prediction).abs().max() for f in confirmation_frames.values()))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=f'''# AET plus chronological graph residual correction

The formula is **AET + strength × graph-predicted AET error**. Correction labels are actual
quarter prices minus genuinely earlier-fit AET predictions; no AET training-set residuals
are used. Both AET and correction paths preserve the supplied hourly means.

## Scores, EUR/MWh

{markdown_table(scores)}

The January–May selection chooses **{winner}**, with correction strength
**{choices[winner]['strength']:g}**. Each candidate's strength was frozen before June–August.
Zero is an eligible strength and means AET only. Raw correction rows show strength 1, rather
than an additional selected model. The former fixed blend is an accuracy reference.

## Every selection trial

{markdown_table(pd.DataFrame(trials))}

## Monthly scores

{markdown_table(monthly)}

## Confirmation zones

{markdown_table(zones)}

The selected recipe improves confirmation MAE versus AET by **{100*gain:.3f}%**, wins **{wins}/3**
months, and has worst-zone degradation **{100*worst:.3f}%**. Promotion gate passed: **{promoted}**
(>=0.5% gain, >=2/3 months won, no zone >2% worse). The incumbent is not automatically replaced.

## Chronology and interpretation

Teacher models fit the fixed official AET recipe through the preceding month-end and predict
all valid dates in the next month, starting November 2025. For each outer evaluation month,
the correction uses only OOF labels ending at its own preceding month-end. Features include
hourly geometry, earlier quarter-price history, AET shape coefficients and adjustments versus
PCHIP, teacher age and teacher training-day count. Scaling fits these correction-training rows only.
Outer scoring still uses days 19 to month-end, as in prior comparisons. Teacher age is explicit
because OOF training includes earlier days of each month as well.

The graph and matched no-message control have the same inputs, training recipe, local prediction
path and shrinkage grid. AET itself has cross-zone features, so disabling graph messages does not
remove all cross-zone information from the combined system. Attention is a model dependency,
not a physical trading flow or causal effect. Zero correction initialization, magnitude regularization
and small selected strength limit unnecessary changes to an already strong AET prediction.

The teacher recipe and earlier development folds were inspected previously; chronological OOF
prevents target-date fitting leakage but does not make this an untouched final test. One seed was
used, with no retuning after confirmation. September and October were excluded from this study.
Earlier scoring actuals may be used as strictly earlier history on later dates, as in the challenge.

## Artifacts and commands

`artifact_graph_through_august/` and `artifact_no_message_through_august/` are isolated integrated
artifacts. Each contains its frozen AET teacher, correction network, scaler, saved history and the
January–May-selected strength. Both components end August 31. Inference never fits or recalibrates.

```bash
.venv-mae/bin/python -m Paola_models.cli infer \\
  --model {str((out/'artifact_graph_through_august').relative_to(REPO))} \\
  --hourly test_hourly.csv --history history_15min.csv --out NEW_predictions.csv
```

`plan.json`, `selection_before_confirmation.json`, `teacher_oof_ledger.csv`, and
`outer_training_ledger.csv` document all cutoffs and choices. Complete OOF predictions and row-level
outer-fold predictions are saved, plus graph attention and masked-edge diagnostics. Protected
artifacts remain unchanged; all results are local in Paola_models on branch Paola.
'''
    (out/'residual_results.md').write_text(report)
    print('\n'+scores.to_string(index=False),flush=True)
    print('\nPROMOTION',promoted,'REPORT',out/'residual_results.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,default=REPO/'Data/train_prices_15min.csv')
    p.add_argument('--out',type=Path,default=REPO/'Paola_models/results/aet_residual_graph_study')
    main(p.parse_args())
