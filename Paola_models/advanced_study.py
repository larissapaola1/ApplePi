"""Bounded graph/clustering/Bayesian study; no September or October scoring."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import REPO
from .probabilistic_graph import AdvancedConfig,ProbabilisticGraph
from .cli import training_cube,markdown_table
from .optimize import official
from aet.config import folds_from_config,load_config
from aet.schema import QH_KEYS,HOUR_KEYS,validate_predictions


def evaluate(model,cube,fold):
    frames=[]
    for i in np.flatnonzero((cube.dates>=fold.score_start)&(cube.dates<=fold.score_end)):
        pred=model.predict_day(cube.dates[i],cube.hourly()[i],cube)
        assert np.isfinite(pred).all() and np.max(np.abs(pred.mean(-1)-cube.hourly()[i]))<.0001
        frame=cube._select(cube.dates==cube.dates[i]).to_long();frame['fold']=fold.name
        frame['predicted_price_eur_mwh']=pred.reshape(-1);frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def inspect(model,cube,fold):
    graph_rows=[];ablation=[];sensitivity=[];prob=[];distribution_scores=[]
    for i in np.flatnonzero((cube.dates>=fold.score_start)&(cube.dates<=fold.score_end)):
        day,H,truth=cube.dates[i],cube.hourly()[i],cube.values[i]
        point,edges,regime=model.predict_day(day,H,cube,return_graph=True)
        averaged=edges.mean(0)
        for h in range(24):
            for target in range(4):
                for source in range(4):
                    graph_rows.append({'fold':fold.name,'delivery_date':str(day),'hour_index':h+1,
                        'target':f'ZONE_{target}','source':f'ZONE_{source}','attention':averaged[h,target,source],
                        'regime_1_probability':regime[-1],
                        'source_cluster':int(model.zone_clusters[source]),'target_cluster':int(model.zone_clusters[target])})
        for source in range(4):
            shifted=H.copy();shifted[source]+=10
            changed=model.predict_day(day,shifted,cube)
            for target in range(4):
                if source==target:continue
                without=model.predict_day(day,H,cube,blocked_edge=(target,source))
                ablation.append({'fold':fold.name,'delivery_date':str(day),'source':f'ZONE_{source}',
                    'target':f'ZONE_{target}','MAE_increase_when_edge_removed':
                    np.abs(without[target]-truth[target]).mean()-np.abs(point[target]-truth[target]).mean(),
                    'mean_prediction_change':np.abs(without[target]-point[target]).mean()})
                sensitivity.append({'fold':fold.name,'delivery_date':str(day),'source':f'ZONE_{source}',
                    'target':f'ZONE_{target}','mean_absolute_target_shape_change':np.abs(changed[target]-point[target]).mean(),
                    'source_hourly_shift_EUR_MWh':10})
        samples,epistemic=model.sample_day(day,H,cube)
        assert np.max(np.abs(samples.mean(-1)-H[None]))<.0001
        std=samples.std(0).clip(1)
        lower=point-model.calibration_radius[:,None,None]*std
        upper=point+model.calibration_radius[:,None,None]*std
        rawlo,rawhi=np.quantile(samples,[.05,.95],axis=0)
        inside=(truth>=lower)&(truth<=upper)
        rawinside=(truth>=rawlo)&(truth<=rawhi)
        sorted_samples=np.sort(samples,axis=0);S=len(samples)
        crps=np.abs(samples-truth).mean(0)-np.sum((2*np.arange(1,S+1)-S-1)[:,None,None,None]*sorted_samples,axis=0)/S**2
        interval_score=upper-lower+20*np.maximum(lower-truth,0)+20*np.maximum(truth-upper,0)
        for z,zone in enumerate(cube.zones):
            distribution_scores.append({'fold':fold.name,'delivery_date':str(day),'market_id':zone,
                'raw90_coverage':rawinside[z].mean(),'calibrated90_coverage':inside[z].mean(),
                'mean_interval_width':(upper[z]-lower[z]).mean(),'interval_score90':interval_score[z].mean(),
                'CRPS':crps[z].mean(),'whole_zone_day_inside':bool(inside[z].all()),
                'weight_training_end':model.weight_cutoff,'data_and_calibration_end':model.data_cutoff})
        frame=cube._select(cube.dates==day).to_long();frame['fold']=fold.name
        for name,value in {'point':point,'lower90':lower,'upper90':upper,'raw_lower90':rawlo,'raw_upper90':rawhi,
                           'epistemic_std':epistemic,'predictive_std':std,
                           'raw_probability_negative':(samples<0).mean(0)}.items():frame[name]=value.reshape(-1)
        prob.append(frame)
    return {'hourly_graph':pd.DataFrame(graph_rows),'edge_ablation':pd.DataFrame(ablation),
            'neighbor_sensitivity':pd.DataFrame(sensitivity),'probabilistic_predictions':pd.concat(prob,ignore_index=True),
            'distribution_scores':pd.DataFrame(distribution_scores)}


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    base=AdvancedConfig()
    candidates={
        'adaptive_point':replace(base,clustered=False,bayesian=False,calibration_days=0),
        'clustered_point':replace(base,bayesian=False,calibration_days=0),
        'clustered_bayesian':base,
        'no_message_bayesian':replace(base,graph=False)}
    folds,_=folds_from_config(load_config(REPO/'AET-HACKATHON/configs/final.yaml'),'monthly')
    search,confirm=folds[:5],folds[5:];names={f.name for f in search}
    plan={'candidates':{k:asdict(v) for k,v in candidates.items()},'selection_folds':sorted(names),
        'confirmation_folds':[f.name for f in confirm],'blend_graph_weights':[0,.1,.25,.5,1],
        'selection_rule':'minimum selection MAE among graph-enabled models; no-message is a control',
        'uncertainty_model':'clustered_bayesian regardless of which point candidate wins',
        'no_retuning_after_confirmation':True,'no_September_or_October':True,
        'probabilistic_calibration':'last 14 calendar days within each training cutoff; excluded from weights, scalers, clustering and graph priors',
        'limitation':'development dates were inspected before; this is not a new independent test'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    cube=training_cube(REPO/'Data/train_prices_15min.csv','2026-08-31')
    baseline_path=REPO/'Paola_models/results/aet_informed_optimization/selected_predictions.csv'
    baseline=pd.read_csv(baseline_path,parse_dates=['delivery_date']).rename(columns={'predicted_price_eur_mwh':'previous_graph_prediction',
                                                                             'blend_prediction':'previous_blend_prediction'})
    reference=baseline[QH_KEYS+['aet_prediction','previous_graph_prediction','previous_blend_prediction','fold']]
    original_paths=[p for folder in [REPO/'AET-HACKATHON/artifacts_official',REPO/'AET-HACKATHON/artifacts_september_assessment',
                                   REPO/'Paola_models/artifacts',REPO/'Paola_models/results/aet_informed_optimization/selected_blend_through_august']
                    for p in folder.rglob('*') if p.is_file()]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in original_paths}
    rows={};ledger=[];cutoffs=[]
    for name,cfg in candidates.items():
        parts=[];start=time.perf_counter()
        for fold in search:
            model=ProbabilisticGraph(cfg).fit(cube.until(fold.train_end))
            parts.append(evaluate(model,cube,fold))
            cutoffs.append({'candidate':name,'fold':fold.name,'weight_training_end':model.weight_cutoff,
                            'data_and_calibration_end':model.data_cutoff})
            print(f'{name} {fold.name} completed',flush=True)
        frame=pd.concat(parts,ignore_index=True).merge(reference,on=QH_KEYS+['fold'],validate='one_to_one')
        frame.to_csv(out/f'selection_{name}.csv.gz',index=False,compression='gzip')
        rows[name]=frame;ledger.append({'candidate':name,'selection_MAE':official(frame),'seconds':time.perf_counter()-start})
        pd.DataFrame(ledger).to_csv(out/'selection_ledger.csv',index=False)
        print(f'{name}: MAE {ledger[-1]["selection_MAE"]:.6f}',flush=True)
    best=min((r for r in ledger if r['candidate']!='no_message_bayesian'),key=lambda r:r['selection_MAE'])['candidate']
    weights=[]
    for alpha in plan['blend_graph_weights']:
        f=rows[best].assign(blend=alpha*rows[best].predicted_price_eur_mwh+(1-alpha)*rows[best].aet_prediction)
        weights.append({'graph_weight':alpha,'selection_MAE':official(f,'blend')})
    alpha=min(weights,key=lambda r:r['selection_MAE'])['graph_weight']
    choice={'point_candidate':best,'probabilistic_candidate':'clustered_bayesian','blend_graph_weight':alpha,'blend_trials':weights}
    (out/'selection_before_confirmation.json').write_text(json.dumps(choice,indent=2)+'\n')
    print('LOCKED',choice,flush=True)
    confirmations={};inspection={};fixed_nodes=[]
    needed=list(dict.fromkeys([best,'clustered_bayesian']))
    for name in needed:
        parts=[];info={}
        for fold in confirm:
            model=ProbabilisticGraph(candidates[name]).fit(cube.until(fold.train_end))
            parts.append(evaluate(model,cube,fold))
            cutoffs.append({'candidate':name,'fold':fold.name,'weight_training_end':model.weight_cutoff,
                            'data_and_calibration_end':model.data_cutoff})
            if name=='clustered_bayesian':
                diagnostics=inspect(model,cube,fold)
                for k,v in diagnostics.items():info.setdefault(k,[]).append(v)
                for z,zone in enumerate(cube.zones):fixed_nodes.append({'fold':fold.name,'zone':zone,'cluster':int(model.zone_clusters[z])})
            print(f'confirmation {name} {fold.name} completed',flush=True)
        confirmations[name]=pd.concat(parts,ignore_index=True).merge(reference,on=QH_KEYS+['fold'],validate='one_to_one')
        for k,values in info.items():inspection[k]=pd.concat(values,ignore_index=True)
    stage_rows=[]
    for stage,frames in [('selection_Jan_May',rows),('confirmation_Jun_Aug',confirmations)]:
        ref=reference[reference.fold.isin(names if stage.startswith('selection') else {f.name for f in confirm})]
        truth=baseline[baseline.fold.isin(ref.fold.unique())]
        for label,column in [('AET','aet_prediction'),('Previous optimized graph','previous_graph_prediction'),('Previous fixed blend','previous_blend_prediction')]:
            stage_rows.append({'stage':stage,'method':label,'MAE':official(truth,column)})
        for name,f in frames.items():stage_rows.append({'stage':stage,'method':name,'MAE':official(f)})
        selected=frames[best].copy()
        selected['new_blend']=alpha*selected.predicted_price_eur_mwh+(1-alpha)*selected.aet_prediction
        stage_rows.append({'stage':stage,'method':'New selected blend','MAE':official(selected,'new_blend')})
    stages=pd.DataFrame(stage_rows);stages.to_csv(out/'stage_scores.csv',index=False)
    pd.DataFrame(cutoffs).to_csv(out/'training_and_calibration_cutoffs.csv',index=False)
    pd.DataFrame(fixed_nodes).to_csv(out/'zone_clusters_by_fold.csv',index=False)
    for name,f in confirmations.items():
        f.to_csv(out/f'confirmation_{name}.csv',index=False)
        hourly=f.groupby(HOUR_KEYS,as_index=False).price_eur_mwh.mean().rename(columns={'price_eur_mwh':'hourly_price_eur_mwh'})
        validate_predictions(f[QH_KEYS+['predicted_price_eur_mwh']],hourly)
    for name,f in inspection.items():f.to_csv(out/f'{name}.csv',index=False)
    edge_summary=inspection['edge_ablation'].groupby(['source','target'])[['MAE_increase_when_edge_removed','mean_prediction_change']].mean().reset_index()
    edge_summary.to_csv(out/'edge_ablation_summary.csv',index=False)
    inspection['neighbor_sensitivity'].groupby(['source','target']).mean(numeric_only=True).to_csv(out/'neighbor_sensitivity_summary.csv')
    uncertainty=inspection['distribution_scores'].groupby('market_id').mean(numeric_only=True)
    uncertainty.to_csv(out/'uncertainty_by_zone.csv')
    for name in needed:
        final=ProbabilisticGraph(candidates[name]).fit(cube)
        final.save(out/f'artifact_{name}')
        if name=='clustered_bayesian':
            posterior=[]
            for layer,block in enumerate(final.network.blocks):
                mu=block.edge_mu.detach().numpy();std=torch_softplus(block.edge_rho.detach().numpy())
                for regime in range(len(mu)):
                    for target in range(4):
                        for source in range(4):posterior.append({'layer':layer,'regime':regime,'target':f'ZONE_{target}',
                            'source':f'ZONE_{source}','logit_offset_mean':mu[regime,target,source],
                            'logit_offset_posterior_std':std[regime,target,source]})
            pd.DataFrame(posterior).to_csv(out/'final_edge_posterior.csv',index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    summary={'choice':choice,'stage_scores':stage_rows,'uncertainty_by_zone':uncertainty.reset_index().to_dict('records'),
             'original_artifacts_unchanged':hashes,'September_scored':False,'new_model_promoted':False,
             'attention_orientation':'row=target/receiver, column=source/sender',
             'uncertainty_scope':'variational output weights and graph offsets; deterministic feature encoder; Student-t coefficient likelihood plus auxiliary MAE',
             'interval_scope':'calibrated marginal intervals; no time-series coverage guarantee'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    report=f"""# Adaptive probabilistic graph: bounded research results

## Selection and confirmation MAE, EUR/MWh

{markdown_table(stages)}

Selected graph: **{best}**. Selected graph blend weight: **{alpha:g}**. These choices were
written before June–August evaluation. All models use the same zone/date scoring keys.
The no-message Bayesian variant is a control, not eligible for graph-model selection.
The former graph and former AET blend are reported alongside all new trials.

## Probabilistic performance on June–August

{markdown_table(uncertainty.reset_index())}

Raw Bayesian predictive intervals and calibrated intervals are reported separately. Nominal
marginal coverage is 90%; compare measured coverage with interval width, interval score and CRPS.
Bayesian models reserve the last 14 calendar days within each training cutoff for interval calibration;
those dates do not fit network weights, scalers, clusters or graph priors. Thus their effective weight
training ends earlier than the point models. Exact dates are in `training_and_calibration_cutoffs.csv`.
Calibration is frozen before scoring and at inference. These intervals have no guaranteed coverage
under arbitrary temporal distribution shift. They are not simultaneous whole-curve confidence sets.

## Which neighbor connections matter to this model?

{markdown_table(edge_summary.sort_values('MAE_increase_when_edge_removed',ascending=False))}

Positive removal delta means removing that source→target graph edge worsened held-out MAE.
Negative means removal helped. This is model-reliance evidence, not causation or measured trade flow.
Attention weights alone are not evidence of influence. The +10 EUR/MWh source-curve perturbations
are hypothetical input sensitivities, not interventions on a physical power system.
Nodes remain ZONE_0–ZONE_3: the official package does not identify countries or their real borders.

## What is implemented

GATv2-style query-dependent attention, learned directed regime-specific edge offsets, two-hop messages,
gated dilated temporal convolutions, direct local coefficient prediction, and a zero-sum output basis.
Training-only two-component diagonal Gaussian EM clusters daily price regimes. A separate training-only
hierarchical grouping of historical residual-tilt similarity supplies zone clusters and graph priors.
No external geographic adjacency or price data is used. All weights start from scratch.

The probabilistic model uses mean-field Gaussian variational distributions on output weights and
graph offsets, Student-t latent-coefficient likelihood (df=4), KL regularization, and an auxiliary MAE
term. This is partial, approximate Bayesian inference, not a fully Bayesian encoder or a posterior
guarantee. Every sampled four-quarter scenario preserves its supplied hourly mean; marginal interval
endpoints need not themselves form a valid scenario. Probabilistic outputs are separate from the
required point-prediction CSV.

The methods are research-inspired, not a claim of state-of-the-art performance on this challenge.
There are only four anonymous nodes and one training year. Complexity must earn its place in the
comparison above. No settings were retuned after confirmation; September and October were excluded.
All development dates were previously inspected, so this is not an independent final test.

## Outputs

`hourly_graph.csv`, `zone_clusters_by_fold.csv`, `edge_ablation.csv`, `neighbor_sensitivity.csv`,
`probabilistic_predictions.csv`, `distribution_scores.csv`, and `final_edge_posterior.csv` are
available for visualization and audits. Original artifacts remain unchanged. New isolated artifacts
are `artifact_{best}/` and `artifact_clustered_bayesian/` (same folder if the Bayesian candidate wins).
The latter's manifest explicitly separates weight-training end from the data/calibration cutoff.
"""
    (out/'advanced_results.md').write_text(report)
    print('\n'+stages.to_string(index=False),flush=True)
    print('\nUNCERTAINTY\n'+uncertainty.to_string(),flush=True)
    print('REPORT',out/'advanced_results.md',flush=True)


def torch_softplus(x):
    return np.logaddexp(0,x)+1e-6


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=REPO/'Paola_models/results/adaptive_probabilistic_study')
    main(p.parse_args().out)
