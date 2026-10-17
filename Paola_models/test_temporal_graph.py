"""Challenge invariants and persistence, using synthetic prices only for tests."""
import numpy as np
import pandas as pd
import pytest
import torch

from .temporal_graph import (Config, GraphReconstructor, GraphTemporalBlock,
                             TemporalGraphNetwork, ZONES, feature_frame)
from aet.cube import QuarterCube


@pytest.fixture
def cube():
    dates = np.arange(np.datetime64("2025-10-20"), np.datetime64("2025-11-01"))
    dates = dates[dates != np.datetime64("2025-10-26")]
    d = np.arange(len(dates))[:,None,None]
    z = np.arange(4)[None,:,None]
    h = np.arange(24)[None,None,:]
    H = 30+3*d+12*z+40*np.sin((h-6)*np.pi/12)
    residual = (5+z+0*d+0*h)[...,None]*np.array([-3,-1,1,3])
    return QuarterCube(dates, ZONES, H[...,None]+residual)


@pytest.fixture
def fitted(cube):
    return GraphReconstructor(Config(hidden=8, blocks=1, epochs=2, dropout=0, threads=1)).fit(cube)


def test_graph_and_temporal_paths_transfer_information():
    torch.manual_seed(7)
    block = GraphTemporalBlock(8, 0).eval()
    x = torch.randn(1,4,24,8)
    y = x.clone()
    y[:,1,12,:] += torch.arange(8)
    # Another zone's information reaches zone 0 at the same hour through graph messages.
    assert not torch.allclose(block(x)[:,0,12], block(y)[:,0,12])
    y = x.clone()
    y[:,0,13,:] += torch.arange(8)
    temporal = GraphTemporalBlock(8,0,graph=False).eval()
    assert not torch.allclose(temporal(x)[:,0,12], temporal(y)[:,0,12])


def test_basis_and_final_output_preserve_means_with_negative_prices(fitted, cube):
    H = cube.hourly()[-1]-150
    prediction = fitted.predict_day("2025-11-01", H, cube)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.mean(axis=-1), H, atol=1e-10)
    # Even a nonzero/random head cannot change the mean algebraically except rounding.
    net = fitted.network
    torch.nn.init.normal_(net.head.weight)
    with torch.no_grad():
        residual = net(torch.randn(2,4,24,2*len(fitted.feature_names)))
    assert residual.sum(dim=-1).abs().max().item() < .0001


def test_current_and_future_quarters_cannot_change_prediction(fitted, cube):
    H = cube.hourly()[-1]
    extra = QuarterCube(np.array([np.datetime64("2025-11-01"),np.datetime64("2025-11-02")]),
                        ZONES, np.full((2,4,24,4), 999999.))
    expected = fitted.predict_day("2025-11-01",H,cube)
    actual = fitted.predict_day("2025-11-01",H,cube.merge(extra))
    np.testing.assert_array_equal(actual, expected)


def test_calendar_gap_remains_missing(cube):
    frame, _ = feature_frame(np.array([np.datetime64("2025-10-27")]), cube.hourly()[0:1], cube)
    assert frame.lag1_c1.isna().all()
    assert frame.lag7_c1.notna().all()


def test_empty_history_is_supported(fitted,cube):
    prediction = fitted.predict_day("2025-11-01",cube.hourly()[-1],QuarterCube.empty(ZONES))
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction.mean(axis=-1),cube.hourly()[-1],atol=1e-10)


def test_save_load_and_inference_do_not_update_parameters(fitted,cube,tmp_path):
    fitted.save(tmp_path/"model")
    restored = GraphReconstructor.load(tmp_path/"model")
    before = {k:v.clone() for k,v in restored.network.state_dict().items()}
    expected = fitted.predict_day("2025-11-01",cube.hourly()[-1],cube)
    actual = restored.predict_day("2025-11-01",cube.hourly()[-1],cube)
    np.testing.assert_array_equal(actual,expected)
    for k,v in restored.network.state_dict().items():
        assert torch.equal(v,before[k])
    with pytest.raises(FileExistsError):
        restored.save(tmp_path/"model")


def test_seeded_training_is_repeatable(cube):
    cfg = Config(hidden=8,blocks=1,epochs=2,dropout=0,threads=1)
    a,b = GraphReconstructor(cfg).fit(cube),GraphReconstructor(cfg).fit(cube)
    np.testing.assert_array_equal(a.predict_day("2025-11-01",cube.hourly()[-1],cube),
                                  b.predict_day("2025-11-01",cube.hourly()[-1],cube))


def test_malformed_or_in_sample_inference_is_rejected(fitted,cube):
    bad = cube.hourly()[-1].copy()
    bad[0,0]=np.nan
    with pytest.raises(ValueError):
        fitted.predict_day("2025-11-01",bad,cube)
    hourly = cube.to_long().groupby(["delivery_date","market_id","hour_index"],as_index=False).price_eur_mwh.mean().rename(columns={"price_eur_mwh":"hourly_price_eur_mwh"})
    with pytest.raises(ValueError, match="cutoff"):
        fitted.predict_frame(hourly,cube)


def test_training_rejects_october_2026(cube):
    bad = QuarterCube(cube.dates+np.timedelta64(365,"D"),ZONES,cube.values)
    with pytest.raises(ValueError,match="official"):
        GraphReconstructor(Config(epochs=1)).fit(bad)


def test_aet_variant_and_shrinkage_roundtrip(cube,tmp_path):
    from dataclasses import replace
    from aet.baselines import cumulative
    cfg=Config(hidden=8,blocks=1,epochs=2,threads=1,dropout=0,cross_zone=True,
               local_skip=True,loss="coefficient",residual_scale=1)
    model=GraphReconstructor(cfg).fit(cube)
    assert "z1_spread" in model.feature_names
    H=cube.hourly()[-1]
    full=model.predict_day("2025-11-01",H,cube)
    model.config=replace(cfg,residual_scale=.5)
    half=model.predict_day("2025-11-01",H,cube)
    np.testing.assert_allclose(half,cumulative(H)+.5*(full-cumulative(H)),atol=1e-10)
    model.save(tmp_path/"aet_variant")
    restored=GraphReconstructor.load(tmp_path/"aet_variant")
    np.testing.assert_array_equal(restored.predict_day("2025-11-01",H,cube),half)


def test_old_manifest_without_new_options_is_loadable(fitted,cube,tmp_path):
    import json
    fitted.save(tmp_path/"old")
    p=tmp_path/"old/manifest.json"
    meta=json.loads(p.read_text())
    for key in ["cross_zone","loss","residual_scale","local_skip"]:
        meta["config"].pop(key)
    p.write_text(json.dumps(meta))
    restored=GraphReconstructor.load(tmp_path/"old")
    np.testing.assert_array_equal(restored.predict_day("2025-11-01",cube.hourly()[-1],cube),
                                  fitted.predict_day("2025-11-01",cube.hourly()[-1],cube))


def test_hybrid_roundtrip_and_no_future_information(fitted,cube,tmp_path):
    from .hybrid import HybridReconstructor,save_hybrid
    from aet.pipeline import ShapePipeline,PipelineConfig
    from aet.schema import validate_predictions
    tree=ShapePipeline(PipelineConfig(num_boost_round=3)).fit(cube)
    tree.save(tmp_path/"aet",cube)
    save_hybrid(tmp_path/"hybrid",fitted,tmp_path/"aet",.25,{"test":True})
    restored=HybridReconstructor.load(tmp_path/"hybrid")
    actual=QuarterCube(np.array([np.datetime64("2025-11-01")]),ZONES,cube.values[-1:])
    hourly=actual.to_long().groupby(["delivery_date","market_id","hour_index"],as_index=False).price_eur_mwh.mean().rename(columns={"price_eur_mwh":"hourly_price_eur_mwh"})
    output=restored.predict_frame(hourly,cube)
    validate_predictions(output,hourly)
    poison=QuarterCube(np.array([np.datetime64("2025-11-01")]),ZONES,np.full((1,4,24,4),999999.))
    other=restored.predict_frame(hourly,cube.merge(poison))
    np.testing.assert_array_equal(output.predicted_price_eur_mwh,other.predicted_price_eur_mwh)
    g=fitted.predict_frame(hourly,cube)
    a,_=tree.predict_frame(hourly,cube)
    np.testing.assert_allclose(output.predicted_price_eur_mwh,
        .25*g.predicted_price_eur_mwh.to_numpy()+.75*a.predicted_price_eur_mwh.to_numpy(),atol=1e-10)


def test_cli_can_train_selected_recipe_and_infer(cube,tmp_path):
    from argparse import Namespace
    from dataclasses import asdict
    import json
    from .cli import train,infer
    path=tmp_path/"training.csv"
    cube.to_long().to_csv(path,index=False)
    cfg=Config(hidden=8,blocks=1,epochs=1,threads=1,cross_zone=True,local_skip=True,
               loss="coefficient",residual_scale=.8)
    recipe=tmp_path/"recipe.json"
    recipe.write_text(json.dumps({"config":asdict(cfg)}))
    train(Namespace(train=path,train_end="2025-10-31",recipe=recipe,epochs=None,
                    no_graph=False,out=tmp_path/"model"))
    meta=json.loads((tmp_path/"model/manifest.json").read_text())
    assert meta["config"]==asdict(cfg)
    future=QuarterCube(np.array([np.datetime64("2025-11-01")]),ZONES,cube.values[-1:])
    hourly=future.to_long().groupby(["delivery_date","market_id","hour_index"],as_index=False).price_eur_mwh.mean().rename(columns={"price_eur_mwh":"hourly_price_eur_mwh"})
    hourly.to_csv(tmp_path/"hourly.csv",index=False)
    infer(Namespace(model=tmp_path/"model",history=None,hourly=tmp_path/"hourly.csv",out=tmp_path/"pred.csv"))
    assert len(pd.read_csv(tmp_path/"pred.csv"))==384
