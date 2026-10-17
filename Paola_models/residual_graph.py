"""A small graph trained on chronological out-of-fold AET reconstruction errors."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .temporal_graph import ZONES, feature_frame, seed_everything
from aet.basis import encode, get_basis
from aet.cube import QuarterCube, HourlyCube
from aet.pipeline import ShapePipeline
from aet.schema import PRED_COLUMNS, validate_predictions


@dataclass(frozen=True)
class ResidualConfig:
    hidden: int = 16
    epochs: int = 80
    batch_days: int = 16
    learning_rate: float = .001
    weight_decay: float = .01
    dropout: float = .15
    correction_penalty: float = .02
    target_scale: float = 5.
    seed: int = 42
    threads: int = 4
    graph: bool = True

    def __post_init__(self):
        if min(self.hidden,self.epochs,self.batch_days,self.threads)<1:
            raise ValueError('positive dimensions and training counts required')
        if not 0<=self.dropout<1 or self.learning_rate<=0 or self.weight_decay<0:
            raise ValueError('invalid optimizer settings')
        if self.target_scale<=0 or self.correction_penalty<0:
            raise ValueError('invalid correction settings')


@dataclass(frozen=True)
class OOFPredictions:
    """Predictions plus the teacher's strictly earlier training cutoff for every date."""
    cube: QuarterCube
    teacher_end: np.ndarray
    teacher_days: np.ndarray

    def __post_init__(self):
        n=len(self.cube.dates)
        if self.cube.zones!=ZONES or not n or not np.isfinite(self.cube.values).all():
            raise ValueError('complete finite four-zone OOF predictions required')
        if self.teacher_end.shape!=(n,) or self.teacher_days.shape!=(n,):
            raise ValueError('one teacher cutoff and training-day count per OOF date required')
        if not (self.teacher_end<self.cube.dates).all():
            raise ValueError('OOF teacher must train strictly before each target date')
        if not np.isfinite(self.teacher_days).all() or (self.teacher_days<1).any():
            raise ValueError('invalid teacher training-day counts')
        if self.cube.dates[0]<np.datetime64('2025-10-01') or self.cube.dates[-1]>np.datetime64('2026-09-30'):
            raise ValueError('OOF data must stay inside the official package period')

    def until(self,day):
        mask=self.cube.dates<=np.datetime64(day,'D')
        return OOFPredictions(self.cube._select(mask),self.teacher_end[mask],self.teacher_days[mask])

    def save(self,path):
        np.savez_compressed(path,dates=self.cube.dates,values=self.cube.values,
                            teacher_end=self.teacher_end,teacher_days=self.teacher_days)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as x:
            return cls(QuarterCube(x['dates'].copy(),ZONES,x['values'].copy()),
                       x['teacher_end'].copy(),x['teacher_days'].copy())

    def to_frame(self):
        f=self.cube.to_long('aet_oof_prediction')
        f['teacher_training_end']=np.repeat(self.teacher_end,384)
        f['teacher_training_days']=np.repeat(self.teacher_days,384)
        return f


def generate_teacher_block(cube,teacher_end,target_end,config):
    """Expanding-window teacher; labels after teacher_end never fit its parameters."""
    end=np.datetime64(teacher_end,'D');target_end=np.datetime64(target_end,'D')
    if cube.zones!=ZONES or not len(cube.dates) or cube.dates[0]<np.datetime64('2025-10-01') or target_end>np.datetime64('2026-09-30'):
        raise ValueError('teacher data must stay within the official four-zone package')
    train=cube.until(end)
    targets=cube._select((cube.dates>end)&(cube.dates<=target_end))
    if len(train.dates)<28 or not len(targets.dates):
        raise ValueError('teacher needs at least 28 earlier days and later targets')
    if not np.isfinite(train.values).all() or not np.isfinite(targets.values).all():
        raise ValueError('complete official training and target days required')
    teacher=ShapePipeline(config).fit(train)
    values=[]
    for i,day in enumerate(targets.dates):
        pred,diag=teacher.predict_day(day,targets.hourly()[i],cube.before(day))
        if diag['fallback_hours'].sum() or diag['flat_hours'].sum():
            raise RuntimeError('OOF teacher fell back')
        if not np.isfinite(pred).all() or np.max(np.abs(pred.mean(-1)-targets.hourly()[i]))>.0001:
            raise ValueError('invalid OOF teacher output')
        values.append(pred)
    oof=OOFPredictions(QuarterCube(targets.dates,ZONES,np.stack(values)),
                      np.full(len(values),train.dates[-1],dtype='datetime64[D]'),
                      np.full(len(values),len(train.dates)))
    return oof,teacher,train


class ResidualNetwork(nn.Module):
    """One temporal block and optional directed messages; starts at zero correction."""
    def __init__(self,features,cfg):
        super().__init__()
        c=cfg.hidden
        self.cfg=cfg
        self.input=nn.Linear(features,c)
        self.temporal=nn.Conv1d(c,c,3,padding=1)
        self.norm=nn.LayerNorm(c)
        self.q=nn.Linear(c,c,bias=False);self.k=nn.Linear(c,c,bias=False)
        self.v=nn.Linear(c,c,bias=False);self.mix=nn.Linear(c,c)
        self.graph_gate=nn.Parameter(torch.tensor(-2.))
        self.drop=nn.Dropout(cfg.dropout)
        self.head=nn.Linear(c+features,3)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('basis',torch.tensor(get_basis('poly'),dtype=torch.float32))

    def forward(self,x,blocked_edge=None):
        hidden=F.gelu(self.input(x));b,z,h,c=hidden.shape
        temporal=self.temporal(hidden.reshape(b*z,h,c).transpose(1,2)).transpose(1,2).reshape(b,z,h,c)
        hidden=self.norm(hidden+self.drop(F.gelu(temporal)))
        attention=torch.eye(4,device=x.device).expand(b,h,4,4)
        if self.cfg.graph:
            nodes=hidden.permute(0,2,1,3)
            logits=self.q(nodes)@self.k(nodes).transpose(-1,-2)/c**.5
            if blocked_edge is not None:
                t,s=blocked_edge;logits=logits.clone();logits[:,:,t,s]=-1e9
            attention=logits.softmax(-1)
            messages=self.mix(attention@self.v(nodes)).permute(0,2,1,3)
            hidden=hidden+torch.sigmoid(self.graph_gate)*self.drop(F.gelu(messages))
        coef=self.head(torch.cat([x,hidden],dim=-1))
        return (coef@self.basis.T)*self.cfg.target_scale,attention


class ResidualCorrection:
    def __init__(self,config=ResidualConfig()):
        self.config=config
        self.network=None

    def frame(self,dates,H,history,aet_values,teacher_end,teacher_days):
        f,base=feature_frame(dates,H,history,cross_zone=False)
        basis=get_basis('poly')
        shape=encode(aet_values-H[...,None],basis)
        adjustment=encode(aet_values-base,basis)
        for k in range(3):
            f[f'aet_shape_c{k}']=shape[...,k].reshape(-1)
            f[f'aet_vs_pchip_c{k}']=adjustment[...,k].reshape(-1)
        ages=(dates-teacher_end).astype('timedelta64[D]').astype(float)
        f['teacher_age_days']=np.repeat(ages,96)
        f['teacher_training_days']=np.repeat(teacher_days,96)
        return f

    def tensor(self,frame):
        if list(frame.columns)!=self.feature_names:
            raise ValueError('correction feature schema mismatch')
        v=frame.to_numpy(float);missing=~np.isfinite(v)
        x=np.clip((np.where(missing,self.mean,v)-self.mean)/self.scale,-8,8)
        return torch.tensor(np.concatenate([x,missing.astype(float)],axis=-1).reshape(-1,4,24,2*len(self.feature_names)),dtype=torch.float32)

    def fit(self,training,oof):
        if training.zones!=ZONES or not len(training.dates) or not np.isfinite(training.values).all():
            raise ValueError('complete official training cube required')
        if training.dates[0]<np.datetime64('2025-10-01') or training.dates[-1]>np.datetime64('2026-09-30'):
            raise ValueError('training must stay within the official package period')
        if len(oof.cube.dates)<20 or oof.cube.dates[-1]>training.dates[-1]:
            raise ValueError('at least 20 OOF dates, all within the outer training cutoff, required')
        indexes=np.searchsorted(training.dates,oof.cube.dates)
        if (indexes>=len(training.dates)).any() or not np.array_equal(training.dates[indexes],oof.cube.dates):
            raise ValueError('OOF dates absent from training labels')
        truth=training.values[indexes];H=truth.mean(-1)
        if np.max(np.abs(oof.cube.hourly()-H))>.0001:
            raise ValueError('OOF hourly means disagree with official labels')
        seed_everything(self.config)
        f=self.frame(oof.cube.dates,H,training,oof.cube.values,oof.teacher_end,oof.teacher_days)
        self.feature_names=list(f.columns)
        v=f.to_numpy(float);finite=np.isfinite(v);count=finite.sum(0)
        self.mean=np.divide(np.where(finite,v,0).sum(0),count,out=np.zeros(v.shape[1]),where=count>0)
        variance=np.divide((np.where(finite,v-self.mean,0)**2).sum(0),count,out=np.zeros(v.shape[1]),where=count>0)
        self.scale=np.sqrt(variance);self.scale[self.scale<1e-6]=1
        X=self.tensor(f)
        target=torch.tensor(truth-oof.cube.values,dtype=torch.float32)
        self.network=ResidualNetwork(X.shape[-1],self.config)
        optimizer=torch.optim.AdamW(self.network.parameters(),lr=self.config.learning_rate,weight_decay=self.config.weight_decay)
        generator=torch.Generator().manual_seed(self.config.seed)
        self.losses=[]
        for _ in range(self.config.epochs):
            self.network.train();total=0.
            for ids in torch.randperm(len(X),generator=generator).split(self.config.batch_days):
                optimizer.zero_grad(set_to_none=True)
                correction,_=self.network(X[ids])
                mae=(correction-target[ids]).abs().mean()
                loss=mae/self.config.target_scale+self.config.correction_penalty*(correction/self.config.target_scale).square().mean()
                loss.backward();nn.utils.clip_grad_norm_(self.network.parameters(),1.)
                optimizer.step();total+=mae.item()*len(ids)
            self.losses.append(total/len(X))
        self.network.eval()
        self.training_range=[str(training.dates[0]),str(training.dates[-1])]
        self.oof_range=[str(oof.cube.dates[0]),str(oof.cube.dates[-1])]
        self.oof_days=len(oof.cube.dates)
        self.teacher_training_days=len(training.dates)
        return self

    def predict_day(self,day,H,history,aet_values,blocked_edge=None):
        day=np.datetime64(day,'D');H=np.asarray(H,float);aet_values=np.asarray(aet_values,float)
        if H.shape!=(4,24) or aet_values.shape!=(4,24,4) or not np.isfinite(H).all() or not np.isfinite(aet_values).all() or history.zones!=ZONES:
            raise ValueError('complete finite four-zone inputs required')
        if np.max(np.abs(aet_values.mean(-1)-H))>.0001:
            raise ValueError('AET baseline violates hourly means')
        if day<=np.datetime64(self.training_range[1]):
            raise ValueError('target date must follow correction training cutoff')
        f=self.frame(np.array([day]),H[None],history.before(day),aet_values[None],
                     np.array([np.datetime64(self.training_range[1])]),np.array([self.teacher_training_days]))
        self.network.eval()
        with torch.no_grad():
            correction,attention=self.network(self.tensor(f),blocked_edge)
        value=correction.numpy()[0].astype(float)
        value-=value.mean(-1,keepdims=True)
        if not np.isfinite(value).all():raise FloatingPointError('nonfinite residual correction')
        return value,attention.numpy()[0]

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        torch.save(self.network.state_dict(),path/'weights.pt')
        np.savez(path/'scaler.npz',mean=self.mean,scale=self.scale)
        m={'config':asdict(self.config),'feature_names':self.feature_names,'training_range':self.training_range,
           'oof_range':self.oof_range,'oof_days':self.oof_days,'teacher_training_days':self.teacher_training_days,
           'training_MAE':self.losses,'parameters':sum(p.numel() for p in self.network.parameters()),
           'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__}
        (path/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path);m=json.loads((path/'manifest.json').read_text());obj=cls(ResidualConfig(**m['config']))
        seed_everything(obj.config)
        for key in ['feature_names','training_range','oof_range','oof_days','teacher_training_days']:setattr(obj,key,m[key])
        obj.losses=m['training_MAE']
        with np.load(path/'scaler.npz',allow_pickle=False) as x:obj.mean,obj.scale=x['mean'].copy(),x['scale'].copy()
        obj.network=ResidualNetwork(2*len(obj.feature_names),obj.config)
        obj.network.load_state_dict(torch.load(path/'weights.pt',map_location='cpu',weights_only=True));obj.network.eval()
        return obj


class AETResidualReconstructor:
    def __init__(self,correction,aet,history,strength):
        if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('strength must be in [0,1]')
        if tuple(aet.zones)!=ZONES or list(aet.training_range)!=correction.training_range:
            raise ValueError('teacher and correction training ranges must match')
        if len(history.dates)!=correction.teacher_training_days or str(history.dates[-1])!=correction.training_range[-1]:
            raise ValueError('teacher history does not match correction training data')
        self.correction,self.aet,self.training_history=correction,aet,history
        self.strength=float(strength);self.training_range=correction.training_range

    def predict_day(self,day,H,history,return_components=False,blocked_edge=None):
        base,diag=self.aet.predict_day(day,H,history.before(day))
        if diag['fallback_hours'].sum() or diag['flat_hours'].sum():raise RuntimeError('AET baseline fell back')
        delta,attention=self.correction.predict_day(day,H,history,base,blocked_edge)
        point=base+self.strength*delta
        point+=(H-point.mean(-1))[...,None]
        return (point,base,delta,attention) if return_components else point

    def predict_frame(self,hourly,history):
        if set(hourly.market_id)!=set(ZONES):raise ValueError('all four zones required')
        cube=HourlyCube.from_long(hourly,ZONES)
        if not len(cube.dates) or not np.isfinite(cube.values).all():raise ValueError('complete finite hourly curves required')
        if (cube.dates<=np.datetime64(self.training_range[1])).any():raise ValueError('target dates must follow training cutoff')
        values=np.stack([self.predict_day(day,cube.values[i],history) for i,day in enumerate(cube.dates)])
        out=QuarterCube(cube.dates,ZONES,values).to_long('predicted_price_eur_mwh')[PRED_COLUMNS]
        validate_predictions(out,hourly)
        return out

    def save(self,path,aet_checkpoint,selection):
        # Copy the exact frozen teacher; never refit or overwrite incumbent artifacts.
        other,history=ShapePipeline.load(aet_checkpoint)
        AETResidualReconstructor(self.correction,other,history,self.strength)
        if other.config.to_dict()!=self.aet.config.to_dict():raise ValueError('teacher config mismatch')
        for i in range(3):
            if other.model.boosters[i].model_to_string()!=self.aet.model.boosters[i].model_to_string():
                raise ValueError('teacher weights mismatch')
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.correction.save(path/'correction');shutil.copytree(aet_checkpoint,path/'aet')
        meta={'kind':'aet_residual_graph','strength':self.strength,'training_range':self.training_range,
              'selection':selection,'formula':'AET + strength * GNN(AET_OOF_errors)',
              'fitting_at_inference':False}
        (path/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path);meta=json.loads((path/'manifest.json').read_text())
        if meta.get('kind')!='aet_residual_graph':raise ValueError('unsupported residual artifact')
        correction=ResidualCorrection.load(path/'correction');aet,history=ShapePipeline.load(path/'aet')
        return cls(correction,aet,history,meta['strength'])
