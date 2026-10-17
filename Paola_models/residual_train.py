"""Retrain the correction from saved chronological OOF data and a frozen AET checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import REPO
from .cli import training_cube
from .residual_graph import OOFPredictions,ResidualConfig,ResidualCorrection,AETResidualReconstructor
from aet.pipeline import ShapePipeline


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,default=REPO/'Data/train_prices_15min.csv')
    p.add_argument('--train-end',default='2026-08-31')
    p.add_argument('--oof',type=Path,required=True)
    p.add_argument('--aet-model',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True,help='selection_before_confirmation.json from residual_study')
    p.add_argument('--candidate',choices=['graph','no_message'],help='default: development-selected candidate')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('choose a new output directory')
    selection=json.loads(args.recipe.read_text())
    name=args.candidate or selection['selected_candidate']
    recipe=selection['candidates'][name]
    cube=training_cube(args.train,args.train_end)
    aet,history=ShapePipeline.load(args.aet_model)
    if not np.array_equal(cube.dates,history.dates) or not np.allclose(cube.values,history.values,atol=1e-10,rtol=0):
        raise ValueError('frozen teacher history differs from requested correction training data')
    oof=OOFPredictions.load(args.oof).until(args.train_end)
    correction=ResidualCorrection(ResidualConfig(**recipe['config'])).fit(cube,oof)
    model=AETResidualReconstructor(correction,aet,history,recipe['strength'])
    provenance={'development_selection':selection,'selected_artifact_candidate':name,
                'OOF_sha256':hashlib.sha256(args.oof.read_bytes()).hexdigest(),
                'training_data_sha256':hashlib.sha256(args.train.read_bytes()).hexdigest()}
    model.save(args.out,args.aet_model,provenance)
    print(json.dumps({'artifact':str(args.out),'candidate':name,'strength':model.strength,
                      'training_range':model.training_range,'OOF_range':correction.oof_range},indent=2))


if __name__=='__main__':
    main()
