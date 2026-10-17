"""Actual constraints and information boundaries for the advanced research model."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from .probabilistic_graph import AdvancedConfig,ProbabilisticGraph,gaussian_kl
from .temporal_graph import ZONES
from aet.cube import QuarterCube


@pytest.fixture
def data():
    dates=np.arange(np.datetime64('2025-10-01'),np.datetime64('2025-12-01'))
    dates=dates[dates!=np.datetime64('2025-10-26')]
    d=np.arange(len(dates))[:,None,None];z=np.arange(4)[None,:,None];h=np.arange(24)[None,None,:]
    H=80+z*8+12*np.sin(h/24*6.28)+d*.1
    shape=(2+np.cos(h/24*6.28)+z/3+d*0)[...,None]*np.array([-3,-1,1,3])
    return QuarterCube(dates,ZONES,H[...,None]+shape)


@pytest.fixture
def model(data):
    return ProbabilisticGraph(AdvancedConfig(hidden=8,epochs=2,threads=1,draws=8,calibration_days=3)).fit(data)


def test_posterior_scenarios_preserve_all_hourly_means(model,data):
    H=data.hourly()[-1]-150
    samples,epistemic=model.sample_day('2025-12-01',H,data)
    assert samples.shape==(8,4,24,4) and np.isfinite(samples).all()
    np.testing.assert_allclose(samples.mean(-1),np.broadcast_to(H,samples.shape[:-1]),atol=1e-10)
    assert epistemic.max()>0
    lower,upper,_,_=model.interval_day('2025-12-01',H,data)
    assert (upper>=lower).all()


def test_future_target_poisoning_has_no_effect(model,data):
    H=data.hourly()[-1]
    poison=QuarterCube(np.array([np.datetime64('2025-12-01'),np.datetime64('2025-12-02')]),ZONES,np.full((2,4,24,4),999999.))
    a=model.predict_day('2025-12-01',H,data)
    b=model.predict_day('2025-12-01',H,data.merge(poison))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(model.sample_day('2025-12-01',H,data)[0],
                                  model.sample_day('2025-12-01',H,data.merge(poison))[0])


def test_edge_orientation_mask_and_attention_normalization(model,data):
    _,edges,r=model.predict_day('2025-12-01',data.hourly()[-1],data,blocked_edge=(1,0),return_graph=True)
    np.testing.assert_allclose(edges.sum(-1),1,atol=1e-6)
    assert (edges[:,:,1,0]==0).all() # zone 0 sending to zone 1
    assert (edges[:,:,0,1]>0).all()
    assert np.isclose(r.sum(),1)


def test_calibration_data_excluded_from_weights_scalers_and_cluster_fit(data):
    cfg=AdvancedConfig(hidden=8,epochs=1,threads=1,draws=8,calibration_days=3)
    a=ProbabilisticGraph(cfg).fit(data)
    changed=data.values.copy();changed[-3:]+=10000
    b=ProbabilisticGraph(cfg).fit(QuarterCube(data.dates,ZONES,changed))
    assert a.weight_cutoff=='2025-11-27' and a.data_cutoff=='2025-11-30'
    np.testing.assert_array_equal(a.mean,b.mean)
    np.testing.assert_array_equal(a.centers,b.centers)
    np.testing.assert_array_equal(a.prior,b.prior)
    for k,v in a.network.state_dict().items():assert torch.equal(v,b.network.state_dict()[k])
    assert not np.array_equal(a.calibration_radius,b.calibration_radius)


def test_roundtrip_missing_history_and_no_inference_updates(model,data,tmp_path):
    model.save(tmp_path/'m');restored=ProbabilisticGraph.load(tmp_path/'m')
    H=data.hourly()[-1];empty=QuarterCube.empty(ZONES)
    np.testing.assert_array_equal(model.predict_day('2025-12-01',H,empty),restored.predict_day('2025-12-01',H,empty))
    np.testing.assert_array_equal(model.sample_day('2025-12-01',H,data)[0],restored.sample_day('2025-12-01',H,data)[0])
    before={k:v.clone() for k,v in restored.network.state_dict().items()}
    restored.interval_day('2025-12-01',H,data)
    for k,v in before.items():assert torch.equal(v,restored.network.state_dict()[k])


def test_local_control_point_mode_and_period_checks(data):
    cfg=AdvancedConfig(hidden=8,epochs=1,threads=1,graph=False,clustered=False,bayesian=False)
    model=ProbabilisticGraph(cfg).fit(data)
    pred,edges,r=model.predict_day('2025-12-01',data.hourly()[-1],data,return_graph=True)
    np.testing.assert_allclose(pred.mean(-1),data.hourly()[-1],atol=1e-10)
    np.testing.assert_array_equal(edges,np.broadcast_to(np.eye(4),edges.shape))
    assert model.weight_cutoff==model.data_cutoff
    with pytest.raises(ValueError):model.sample_day('2025-12-01',data.hourly()[-1],data)
    with pytest.raises(ValueError):model.predict_day('2025-12-01',np.full((4,24),np.nan),data)
    with pytest.raises(ValueError):ProbabilisticGraph(cfg).fit(QuarterCube(data.dates+np.timedelta64(365,'D'),ZONES,data.values))


def test_kl_is_finite_and_nonnegative():
    assert gaussian_kl(torch.zeros(3),torch.full((3,),-4.)).item()>0


def test_diagnostic_export_matches_point_predictions_and_coherent_samples(model,data,tmp_path):
    from .probabilistic_cli import export
    import pandas as pd
    from aet.schema import HOUR_KEYS
    future=QuarterCube(np.array([np.datetime64('2025-12-01')]),ZONES,data.values[-1:]).to_long()
    hourly=future.groupby(HOUR_KEYS,as_index=False).price_eur_mwh.mean().rename(columns={'price_eur_mwh':'hourly_price_eur_mwh'})
    export(model,hourly,data,tmp_path/'export')
    out=pd.read_csv(tmp_path/'export/point_predictions.csv')
    diagnostic=pd.read_csv(tmp_path/'export/uncertainty.csv')
    np.testing.assert_allclose(out.predicted_price_eur_mwh,diagnostic.point,atol=1e-9)
    with np.load(tmp_path/'export/scenarios.npz') as sample:
        assert sample['prices'].shape==(1,8,4,24,4)
        np.testing.assert_allclose(sample['prices'].mean(-1),np.broadcast_to(data.hourly()[-1],(1,8,4,24)),atol=1e-10)
    with pytest.raises(FileExistsError):export(model,hourly,data,tmp_path/'export')
