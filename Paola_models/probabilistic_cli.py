"""Train research graph artifacts or export frozen probabilistic diagnostics offline."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import REPO
from .cli import training_cube
from .probabilistic_graph import AdvancedConfig, ProbabilisticGraph
from .temporal_graph import ZONES
from aet.cube import HourlyCube, QuarterCube
from aet.schema import read_hourly, read_quarter_hourly, validate_predictions


def export(model, hourly, history, out):
    """Required point CSV, separate uncertainty CSV, and coherent scenario tensor."""
    if out.exists():
        raise FileExistsError('choose a new output directory')
    point = model.predict_frame(hourly, history)
    validate_predictions(point, hourly)
    cube = HourlyCube.from_long(hourly, ZONES)
    records, scenarios, edges = [], [], []
    for i, day in enumerate(cube.dates):
        H = cube.values[i]
        lo, hi, samples, epistemic = model.interval_day(day, H, history)
        pred, graph, regimes = model.predict_day(day, H, history, return_graph=True)
        if np.max(np.abs(samples.mean(-1)-H[None])) > .0001:
            raise ValueError('incoherent sampled scenarios')
        frame = QuarterCube(np.array([day]), ZONES, pred[None]).to_long('point')
        for name, value in {'lower90':lo, 'upper90':hi, 'epistemic_std':epistemic,
                            'predictive_std':samples.std(0),
                            'raw_probability_negative':(samples<0).mean(0)}.items():
            frame[name] = value.reshape(-1)
        records.append(frame)
        scenarios.append(samples)
        for h in range(24):
            for t, target in enumerate(ZONES):
                for s, source in enumerate(ZONES):
                    edges.append({'delivery_date':str(day), 'hour_index':h+1,
                        'source':source, 'target':target, 'attention':graph.mean(0)[h,t,s],
                        'regime_1_probability':regimes[-1]})
    out.mkdir(parents=True)
    point.to_csv(out/'point_predictions.csv', index=False, float_format='%.10f')
    checks = validate_predictions(pd.read_csv(out/'point_predictions.csv'), hourly)
    pd.concat(records, ignore_index=True).to_csv(out/'uncertainty.csv', index=False)
    pd.DataFrame(edges).to_csv(out/'graph.csv', index=False)
    np.savez_compressed(out/'scenarios.npz', dates=cube.dates, zones=np.array(ZONES),
                        prices=np.stack(scenarios), dimensions=np.array(['date','draw','zone','hour','quarter']))
    metadata = {'data_cutoff':model.data_cutoff, 'weight_cutoff':model.weight_cutoff,
        'checks':checks, 'intervals':'empirically calibrated marginal 90%; no coverage guarantee under shift',
        'scenarios':'raw Bayesian predictive draws; hourly means preserved; not conformally adjusted',
        'negative_probabilities':'raw model probabilities, not separately calibrated',
        'graph':'learned attention, source to target; not causal trade flows'}
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--train', type=Path, default=REPO/'Data/train_prices_15min.csv')
    train.add_argument('--train-end', default='2026-08-31')
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=100)
    infer = sub.add_parser('infer')
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--hourly', type=Path, required=True)
    infer.add_argument('--history', type=Path)
    infer.add_argument('--out', type=Path, required=True, help='new directory, not a CSV path')
    args = p.parse_args()
    if args.command == 'train':
        if args.out.exists():
            raise FileExistsError('choose a new output directory')
        model = ProbabilisticGraph(replace(AdvancedConfig(), epochs=args.epochs)).fit(training_cube(args.train,args.train_end))
        model.save(args.out)
        print(json.dumps({'artifact':str(args.out), 'weight_cutoff':model.weight_cutoff,
                          'data_cutoff':model.data_cutoff}, indent=2))
    else:
        model = ProbabilisticGraph.load(args.model)
        if not model.cfg.bayesian:
            raise ValueError('uncertainty export requires a Bayesian artifact; use Paola_models.cli for point-only artifacts')
        history = model.training_history
        if args.history:
            history = history.merge(QuarterCube.from_long(read_quarter_hourly(args.history,strict=False),ZONES))
        print(json.dumps(export(model, read_hourly(args.hourly), history, args.out),indent=2))


if __name__ == '__main__':
    main()
