"""Chronological stacking boundaries and the actual integrated inference path."""
from dataclasses import replace
from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
import torch

from .residual_graph import (ResidualConfig,OOFPredictions,generate_teacher_block,
                            ResidualCorrection,AETResidualReconstructor)
from .temporal_graph import ZONES
from aet.cube import QuarterCube
from aet.pipeline import PipelineConfig,ShapePipeline
from aet.schema import HOUR_KEYS,validate_predictions


@pytest.fixture(scope='module')
def data():
    dates=np.arange(np.datetime64('2025-10-01'),np.datetime64('2026-01-01'))
    dates=dates[dates!=np.datetime64('2025-10-26')]
    d=np.arange(len(dates))[:,None,None];z=np.arange(4)[None,:,None];h=np.arange(24)[None,None,:]
    H=50+z*5+d*.1+10*np.sin(h*2*np.pi/24)
    shape=(1+z*.3+.3*np.cos(h*2*np.pi/24)+d*.002)[...,None]*np.array([-3,-1,1,3])
    return QuarterCube(dates,ZONES,H[...,None]+shape)


@pytest.fixture(scope='module')
def teacher_config():
    return PipelineConfig(residual_scale=.8,num_boost_round=3,model_params={'num_threads':1})


@pytest.fixture(scope='module')
def oof(data,teacher_config):
    a,_,_=generate_teacher_block(data,'2025-10-31','2025-11-30',teacher_config)
    b,_,_=generate_teacher_block(data,'2025-11-30','2025-12-31',teacher_config)
    return OOFPredictions(a.cube.merge(b.cube),np.concatenate([a.teacher_end,b.teacher_end]),np.concatenate([a.teacher_days,b.teacher_days]))


@pytest.fixture
def correction(data,oof):
    return ResidualCorrection(ResidualConfig(hidden=8,epochs=2,threads=1)).fit(data,oof)


def test_teacher_parameters_and_first_day_ignore_future_labels(data,teacher_config):
    original,teacher,_=generate_teacher_block(data,'2025-10-31','2025-11-01',teacher_config)
    values=data.values.copy();values[data.dates>=np.datetime64('2025-11-02')]=999999
    changed,other,_=generate_teacher_block(QuarterCube(data.dates,ZONES,values),'2025-10-31','2025-11-01',teacher_config)
    assert teacher.training_range[-1]=='2025-10-31'
    assert (original.teacher_end<original.cube.dates).all()
    np.testing.assert_array_equal(original.cube.values,changed.cube.values)
    for a,b in zip(teacher.model.boosters,other.model.boosters):assert a.model_to_string()==b.model_to_string()


def test_invalid_oof_and_outer_cutoffs_are_rejected(data,oof):
    with pytest.raises(ValueError,match='strictly before'):
        OOFPredictions(oof.cube,oof.cube.dates,oof.teacher_days)
    with pytest.raises(ValueError,match='outer training cutoff'):
        ResidualCorrection(ResidualConfig(epochs=1)).fit(data.until('2025-11-30'),oof)
    altered=oof.cube.values.copy();altered+=1
    wrong=OOFPredictions(QuarterCube(oof.cube.dates,ZONES,altered),oof.teacher_end,oof.teacher_days)
    with pytest.raises(ValueError,match='hourly means'):
        ResidualCorrection(ResidualConfig(epochs=1)).fit(data,wrong)


def test_perfect_teacher_needs_zero_correction(data,oof):
    exact=data._select(np.isin(data.dates,oof.cube.dates))
    perfect=OOFPredictions(exact,oof.teacher_end,oof.teacher_days)
    model=ResidualCorrection(ResidualConfig(hidden=8,epochs=2,threads=1)).fit(data,perfect)
    delta,_=model.predict_day('2026-01-01',data.hourly()[-1],data,data.values[-1])
    np.testing.assert_array_equal(delta,np.zeros_like(delta))
    assert model.losses==[0.,0.]


def test_negative_hourly_means_and_directed_graph_mask(correction,data):
    H=data.hourly()[-1]-150;P=data.values[-1]-150
    delta,attention=correction.predict_day('2026-01-01',H,data,P,blocked_edge=(1,0))
    np.testing.assert_allclose(delta.mean(-1),0,atol=1e-12)
    np.testing.assert_allclose((P+delta).mean(-1),H,atol=1e-10)
    np.testing.assert_allclose(attention.sum(-1),1,atol=1e-6)
    assert (attention[:,1,0]==0).all() and (attention[:,0,1]>0).all()


def test_no_message_control_and_oof_serialization(data,oof,tmp_path):
    path=tmp_path/'oof.npz';oof.save(path);loaded=OOFPredictions.load(path)
    np.testing.assert_array_equal(oof.cube.values,loaded.cube.values)
    model=ResidualCorrection(ResidualConfig(hidden=8,epochs=1,threads=1,graph=False)).fit(data,loaded)
    _,attention=model.predict_day('2026-01-01',data.hourly()[-1],data,data.values[-1])
    np.testing.assert_array_equal(attention,np.broadcast_to(np.eye(4),attention.shape))


def test_integrated_cli_roundtrip_future_poisoning_and_frozen_inference(correction,data,teacher_config,tmp_path):
    from .cli import infer
    teacher=ShapePipeline(teacher_config).fit(data)
    teacher.save(tmp_path/'teacher',data)
    model=AETResidualReconstructor(correction,teacher,data,.25)
    model.save(tmp_path/'model',tmp_path/'teacher',{'test':True})
    loaded=AETResidualReconstructor.load(tmp_path/'model')
    future=QuarterCube(np.array([np.datetime64('2026-01-01')]),ZONES,data.values[-1:])
    hourly=future.to_long().groupby(HOUR_KEYS,as_index=False).price_eur_mwh.mean().rename(columns={'price_eur_mwh':'hourly_price_eur_mwh'})
    expected=model.predict_frame(hourly,data)
    poison=QuarterCube(np.array([np.datetime64('2026-01-01'),np.datetime64('2026-01-02')]),ZONES,np.full((2,4,24,4),999999.))
    before={k:v.clone() for k,v in loaded.correction.network.state_dict().items()}
    other=loaded.predict_frame(hourly,data.merge(poison))
    np.testing.assert_array_equal(expected.predicted_price_eur_mwh,other.predicted_price_eur_mwh)
    for k,v in before.items():assert torch.equal(v,loaded.correction.network.state_dict()[k])
    hourly.to_csv(tmp_path/'hourly.csv',index=False)
    infer(Namespace(model=tmp_path/'model',hourly=tmp_path/'hourly.csv',history=None,out=tmp_path/'out.csv'))
    validate_predictions(pd.read_csv(tmp_path/'out.csv'),hourly)
    np.testing.assert_allclose(pd.read_csv(tmp_path/'out.csv').predicted_price_eur_mwh,expected.predicted_price_eur_mwh,atol=1e-9)


def test_zero_strength_and_mismatched_teacher(correction,data,teacher_config):
    teacher=ShapePipeline(teacher_config).fit(data)
    model=AETResidualReconstructor(correction,teacher,data,0)
    point,base,delta,_=model.predict_day('2026-01-01',data.hourly()[-1],data,return_components=True)
    np.testing.assert_allclose(point,base,atol=1e-10)
    with pytest.raises(ValueError,match='training ranges'):
        AETResidualReconstructor(correction,ShapePipeline(teacher_config).fit(data.until('2025-12-30')),data,1)
    with pytest.raises(ValueError,match='cutoff'):
        model.predict_day('2025-12-31',data.hourly()[-1],data)
