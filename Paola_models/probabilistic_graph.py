"""Adaptive directed temporal graphs with regime mixtures and partial Bayesian inference.

Original research implementation, not a reproduction or a claim of benchmark SOTA.
No country identities, physical links, outside data, or pretrained parameters.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.special import logsumexp

from .temporal_graph import feature_frame, seed_everything, ZONES
from aet.baselines import cumulative
from aet.basis import get_basis
from aet.cube import QuarterCube, HourlyCube
from aet.schema import PRED_COLUMNS


@dataclass(frozen=True)
class AdvancedConfig:
    hidden: int = 24
    epochs: int = 100
    batch_days: int = 16
    learning_rate: float = .001
    weight_decay: float = .002
    dropout: float = .1
    seed: int = 42
    threads: int = 4
    graph: bool = True
    clustered: bool = True
    bayesian: bool = True
    calibration_days: int = 14
    residual_scale: float = .8
    target_scale: float = 10.
    draws: int = 64

    def __post_init__(self):
        if min(self.hidden,self.epochs,self.batch_days,self.threads,self.draws)<1:
            raise ValueError("positive sizes required")
        if not 0<=self.dropout<1 or self.calibration_days<0 or self.target_scale<=0:
            raise ValueError("invalid configuration")


def gaussian_kl(mu,rho,prior_std=.5):
    sigma=F.softplus(rho)+1e-6
    return (torch.log(prior_std/sigma)+(sigma.square()+mu.square())/(2*prior_std**2)-.5).sum()


class VariationalLinear(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.mu=nn.Parameter(torch.zeros(outputs,inputs+1))
        self.rho=nn.Parameter(torch.full((outputs,inputs+1),-4.5))

    def forward(self,x,stochastic=False):
        weight=self.mu
        if stochastic:weight=weight+(F.softplus(self.rho)+1e-6)*torch.randn_like(weight)
        return F.linear(F.pad(x,(0,1),value=1),weight)

    def kl(self):
        return gaussian_kl(self.mu,self.rho)


class AdaptiveBlock(nn.Module):
    def __init__(self,hidden,dilation,dropout,regimes,graph,bayesian):
        super().__init__()
        self.temporal=nn.Conv1d(hidden,2*hidden,3,padding=dilation,dilation=dilation)
        self.norm=nn.LayerNorm(hidden)
        self.q=nn.Linear(hidden,hidden,bias=False)
        self.k=nn.Linear(hidden,hidden,bias=False)
        self.v=nn.Linear(hidden,hidden,bias=False)
        self.attention=nn.Parameter(torch.randn(hidden)/hidden**.5)
        self.edge_mu=nn.Parameter(torch.zeros(regimes,4,4))
        self.edge_rho=nn.Parameter(torch.full((regimes,4,4),-4.5),requires_grad=bayesian)
        self.mix=nn.Linear(hidden,hidden)
        self.gate=nn.Parameter(torch.tensor(-1.))
        self.drop=nn.Dropout(dropout)
        self.graph,self.bayesian=graph,bayesian

    def forward(self,x,regime,prior,stochastic=False,blocked_edge=None):
        b,z,h,c=x.shape
        y=self.temporal(x.reshape(b*z,h,c).transpose(1,2)).transpose(1,2).reshape(b,z,h,2*c)
        a,g=y.chunk(2,dim=-1)
        x=self.norm(x+self.drop(torch.tanh(a)*torch.sigmoid(g)))
        nodes=x.permute(0,2,1,3)
        if not self.graph:
            return x,torch.eye(4,device=x.device).expand(b,h,4,4)
        pair=self.q(nodes).unsqueeze(-2)+self.k(nodes).unsqueeze(-3)
        logits=(F.leaky_relu(pair,.2)*self.attention).sum(-1)/c**.5
        edges=self.edge_mu
        if stochastic and self.bayesian:
            edges=edges+(F.softplus(self.edge_rho)+1e-6)*torch.randn_like(edges)
        logits=logits+torch.einsum('br,rij->bij',regime,prior+edges)[:,None]
        if blocked_edge is not None:
            target,source=blocked_edge
            logits=logits.clone();logits[:,:,target,source]=-1e9
        adjacency=logits.softmax(-1)  # row=receiving zone, column=sending zone
        value=self.v(nodes)
        one=adjacency@value
        message=self.mix(.5*one+.5*(adjacency@one)).permute(0,2,1,3)
        return x+torch.sigmoid(self.gate)*self.drop(F.gelu(message)),adjacency

    def kl(self):
        return gaussian_kl(self.edge_mu,self.edge_rho) if self.graph and self.bayesian else self.edge_mu.new_tensor(0.)


class AdaptiveNetwork(nn.Module):
    def __init__(self,features,prior,cfg):
        super().__init__()
        regimes=len(prior)
        self.cfg=cfg
        self.input=nn.Linear(features,cfg.hidden)
        self.blocks=nn.ModuleList([AdaptiveBlock(cfg.hidden,d,cfg.dropout,regimes,cfg.graph,cfg.bayesian) for d in (1,2)])
        context=features+cfg.hidden+regimes
        self.head=VariationalLinear(context,3*regimes) if cfg.bayesian else nn.Linear(context,3*regimes)
        if not cfg.bayesian:
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.scale=nn.Linear(context,3)
        nn.init.zeros_(self.scale.weight);nn.init.constant_(self.scale.bias,-1.)
        self.register_buffer('prior',torch.tensor(prior,dtype=torch.float32))
        self.register_buffer('basis',torch.tensor(get_basis('poly'),dtype=torch.float32))

    def forward(self,x,regime,stochastic=False,blocked_edge=None):
        hidden=F.gelu(self.input(x));edges=[]
        for block in self.blocks:
            hidden,adj=block(hidden,regime,self.prior,stochastic,blocked_edge);edges.append(adj)
        r=regime[:,None,None].expand(*x.shape[:3],regime.shape[-1])
        context=torch.cat([x,hidden,r],dim=-1)
        values=self.head(context,stochastic) if self.cfg.bayesian else self.head(context)
        experts=values.reshape(*x.shape[:3],regime.shape[-1],3)
        location=(experts*regime[:,None,None,:,None]).sum(-2)
        scale=F.softplus(self.scale(context))+.05
        return location,scale,edges

    def kl(self):
        return (self.head.kl() if self.cfg.bayesian else self.prior.new_tensor(0.))+sum(b.kl() for b in self.blocks)


def descriptors(H):
    return np.concatenate([H.mean(-1),H.std(-1),np.ptp(H,axis=-1),np.abs(np.diff(H,axis=-1)).mean(-1)],axis=-1)


def diagonal_mixture(x,seed):
    """Small two-component diagonal Gaussian EM; no native clustering runtime required."""
    rng=np.random.default_rng(seed);best=None
    for _ in range(3):
        means=x[rng.choice(len(x),size=2,replace=False)].copy()
        cov=np.broadcast_to(x.var(0).clip(.01),(2,x.shape[1])).copy();weights=np.full(2,.5)
        for _ in range(60):
            logp=-.5*(((x[:,None]-means)**2/cov).sum(-1)+np.log(2*np.pi*cov).sum(-1))+np.log(weights)
            resp=np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
            mass=resp.sum(0).clip(1e-6)
            means=resp.T@x/mass[:,None]
            cov=(resp[:,:,None]*(x[:,None]-means)**2).sum(0)/mass[:,None]+1e-3
            weights=mass/mass.sum()
        logp=-.5*(((x[:,None]-means)**2/cov).sum(-1)+np.log(2*np.pi*cov).sum(-1))+np.log(weights)
        score=logsumexp(logp,axis=1).mean()
        if best is None or score>best[0]:best=(score,means.copy(),cov.copy(),weights.copy())
    return best[1:]


class ProbabilisticGraph:
    def __init__(self,cfg=AdvancedConfig()):
        self.cfg=cfg
        self.network=None
        self.calibration_radius=np.ones(4)

    def regimes(self,H):
        if not self.cfg.clustered:return np.ones((len(H),1))
        x=np.clip((descriptors(H)-self.regime_mean)/self.regime_scale,-8,8)
        logp=-.5*(((x[:,None]-self.centers)**2/self.covariances).sum(-1)+np.log(2*np.pi*self.covariances).sum(-1))
        logp+=np.log(self.mixture_weights)
        return np.exp(logp-logsumexp(logp,axis=1,keepdims=True))

    def _features(self,dates,H,history):
        frame,base=feature_frame(dates,H,history,cross_zone=False)
        # No explicit peer columns: graph-message ablations have a defined pathway.
        values=frame[self.feature_names].to_numpy(float)
        missing=~np.isfinite(values)
        x=np.clip((np.where(missing,self.mean,values)-self.mean)/self.std,-8,8)
        x=np.concatenate([x,missing.astype(float)],axis=-1)
        return torch.tensor(x.reshape(len(dates),4,24,-1),dtype=torch.float32),base,torch.tensor(self.regimes(H),dtype=torch.float32)

    def fit(self,cube):
        if cube.zones!=ZONES or not len(cube.dates) or not np.isfinite(cube.values).all():
            raise ValueError('complete official four-zone training days required')
        if cube.dates[0]<np.datetime64('2025-10-01') or cube.dates[-1]>np.datetime64('2026-09-30'):
            raise ValueError('training outside official package period')
        seed_everything(self.cfg)
        cal_start=cube.dates[-1]-np.timedelta64(self.cfg.calibration_days-1,'D') if self.cfg.bayesian and self.cfg.calibration_days else cube.dates[-1]+np.timedelta64(1,'D')
        proper=cube.before(cal_start)
        if len(proper.dates)<30:raise ValueError('at least 30 proper-training days required')
        self.data_cutoff=str(cube.dates[-1]);self.weight_cutoff=str(proper.dates[-1]);self.training_history=cube
        H=proper.hourly()
        description=descriptors(H)
        self.regime_mean=description.mean(0);self.regime_scale=description.std(0).clip(1)
        if self.cfg.clustered:
            centers,covariances,weights=diagonal_mixture((description-self.regime_mean)/self.regime_scale,self.cfg.seed)
            order=np.argsort(centers[:,8:12].mean(-1))
            self.centers=centers[order];self.covariances=covariances[order];self.mixture_weights=weights[order]
        else:
            self.centers=np.zeros((1,16));self.covariances=np.ones((1,16));self.mixture_weights=np.ones(1)
        residual=proper.values-cumulative(H)
        tilt=residual[...,3]-residual[...,0]
        flat=tilt.transpose(1,0,2).reshape(4,-1)
        # Similarity prior and zone clustering are estimated exclusively from proper-training labels.
        correlation=np.nan_to_num(np.corrcoef(flat),nan=0);np.fill_diagonal(correlation,1)
        distance=np.clip(1-correlation,0,2);distance=(distance+distance.T)/2;np.fill_diagonal(distance,0)
        self.zone_clusters=fcluster(linkage(squareform(distance),method='average'),t=2,criterion='maxclust')-1
        responsibilities=self.regimes(H)
        priors=[]
        for k in range(responsibilities.shape[1]):
            weights=np.repeat(responsibilities[:,k],24)
            mean=np.average(flat,axis=1,weights=weights)
            centered=flat-mean[:,None]
            covariance=(centered*weights)@centered.T/weights.sum()
            denom=np.sqrt(np.maximum(np.diag(covariance),1e-8))
            corr=covariance/(denom[:,None]*denom[None,:])
            affinity=.25+np.clip(corr,0,1)+.15*(self.zone_clusters[:,None]==self.zone_clusters[None,:])
            affinity/=affinity.sum(-1,keepdims=True)
            priors.append(np.log(affinity))
        self.prior=np.asarray(priors)
        frame,_=feature_frame(proper.dates,H,proper,cross_zone=False)
        self.feature_names=list(frame.columns)
        v=frame.to_numpy(float);finite=np.isfinite(v);count=finite.sum(0)
        self.mean=np.divide(np.where(finite,v,0).sum(0),count,out=np.zeros(v.shape[1]),where=count>0)
        self.std=np.sqrt(np.divide((np.where(finite,v-self.mean,0)**2).sum(0),count,out=np.zeros(v.shape[1]),where=count>0)).clip(.01)
        X,base,R=self._features(proper.dates,H,proper)
        target=torch.tensor((proper.values-base)/self.cfg.target_scale,dtype=torch.float32)
        self.network=AdaptiveNetwork(X.shape[-1],self.prior,self.cfg)
        optimizer=torch.optim.AdamW(self.network.parameters(),lr=self.cfg.learning_rate,weight_decay=self.cfg.weight_decay)
        generator=torch.Generator().manual_seed(self.cfg.seed)
        self.training_losses=[]
        for epoch in range(self.cfg.epochs):
            self.network.train();loss_sum=0.
            for ids in torch.randperm(len(X),generator=generator).split(self.cfg.batch_days):
                optimizer.zero_grad(set_to_none=True)
                loc,scale,_=self.network(X[ids],R[ids],stochastic=self.cfg.bayesian)
                decoded=loc@self.network.basis.T
                mae=(decoded-target[ids]).abs().mean()
                if self.cfg.bayesian:
                    coef=target[ids]@self.network.basis
                    nll=-torch.distributions.StudentT(4.,loc=loc,scale=scale).log_prob(coef).mean()
                    loss=nll+.25*mae+self.network.kl()/(len(X)*4*24*3)
                else:loss=mae
                loss.backward();nn.utils.clip_grad_norm_(self.network.parameters(),1.)
                optimizer.step();loss_sum+=mae.item()*len(ids)*self.cfg.target_scale
            self.training_losses.append(loss_sum/len(X))
        self.network.eval()
        if self.cfg.bayesian and self.cfg.calibration_days:
            errors=[]
            for i in np.flatnonzero(cube.dates>=cal_start):
                point=self.predict_day(cube.dates[i],cube.hourly()[i],cube)
                samples,_=self.sample_day(cube.dates[i],cube.hourly()[i],cube)
                spread=samples.std(0).clip(1)
                errors.append(np.abs(cube.values[i]-point)/spread)
            residuals=np.asarray(errors).transpose(1,0,2,3).reshape(4,-1)
            q=min(1,np.ceil((residuals.shape[1]+1)*.9)/residuals.shape[1])
            self.calibration_radius=np.quantile(residuals,q,axis=1,method='higher')
        return self

    def inputs(self,day,H,history):
        H=np.asarray(H,dtype=float)
        if H.shape!=(4,24) or not np.isfinite(H).all() or history.zones!=ZONES:
            raise ValueError('complete finite hourly curves for all four zones required')
        return self._features(np.array([np.datetime64(day,'D')]),H[None],history.before(day))

    def predict_day(self,day,H,history,blocked_edge=None,return_graph=False):
        x,base,r=self.inputs(day,H,history)
        self.network.eval()
        with torch.no_grad():
            loc,_,edges=self.network(x,r,blocked_edge=blocked_edge)
            correction=(loc@self.network.basis.T).numpy()[0].astype(float)
        point=base[0]+self.cfg.residual_scale*self.cfg.target_scale*correction
        point+=(H-point.mean(-1))[...,None]
        return (point,np.stack([a.numpy()[0] for a in edges]),r.numpy()[0]) if return_graph else point

    def sample_day(self,day,H,history,draws=None):
        if not self.cfg.bayesian:raise ValueError('predictive sampling requires Bayesian model')
        x,base,r=self.inputs(day,H,history);draws=draws or self.cfg.draws
        self.network.eval();scenarios=[];means=[]
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            day_seed=int(np.datetime64(day,'D').astype(int))%1000000
            torch.manual_seed(self.cfg.seed+day_seed)
            for _ in range(draws):
                loc,scale,_=self.network(x,r,stochastic=True)
                coef=torch.distributions.StudentT(4.,loc=loc,scale=scale).sample()
                scenarios.append((coef@self.network.basis.T).numpy()[0])
                means.append((loc@self.network.basis.T).numpy()[0])
        samples=base[0]+self.cfg.residual_scale*self.cfg.target_scale*np.asarray(scenarios,dtype=float)
        samples+=(H[None]-samples.mean(-1))[...,None]
        epistemic=self.cfg.residual_scale*self.cfg.target_scale*np.asarray(means,dtype=float).std(0)
        return samples,epistemic

    def interval_day(self,day,H,history):
        point=self.predict_day(day,H,history)
        samples,epistemic=self.sample_day(day,H,history)
        spread=samples.std(0).clip(1)
        radius=self.calibration_radius[:,None,None]*spread
        return point-radius,point+radius,samples,epistemic

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        torch.save(self.network.state_dict(),path/'weights.pt')
        np.savez(path/'transforms.npz',mean=self.mean,std=self.std,prior=self.prior,
                 regime_mean=self.regime_mean,regime_scale=self.regime_scale,centers=self.centers,
                 covariances=self.covariances,mixture_weights=self.mixture_weights,
                 zone_clusters=self.zone_clusters,calibration_radius=self.calibration_radius)
        self.training_history.save(path/'history.npz')
        metadata={'kind':'adaptive_probabilistic_graph','config':asdict(self.cfg),'feature_names':self.feature_names,
                  'data_cutoff':self.data_cutoff,'weight_cutoff':self.weight_cutoff,'zones':list(ZONES),
                  'parameters':sum(p.numel() for p in self.network.parameters()),'training_losses':self.training_losses,
                  'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__}
        (path/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path);m=json.loads((path/'manifest.json').read_text())
        if m['kind']!='adaptive_probabilistic_graph' or m['zones']!=list(ZONES):raise ValueError('unsupported artifact')
        obj=cls(AdvancedConfig(**m['config']));seed_everything(obj.cfg)
        with np.load(path/'transforms.npz',allow_pickle=False) as arrays:
            for key in arrays.files:setattr(obj,key,arrays[key].copy())
        obj.feature_names=m['feature_names'];obj.data_cutoff=m['data_cutoff'];obj.weight_cutoff=m['weight_cutoff']
        obj.training_history=QuarterCube.load(path/'history.npz');obj.training_losses=m['training_losses']
        obj.network=AdaptiveNetwork(2*len(obj.feature_names),obj.prior,obj.cfg)
        obj.network.load_state_dict(torch.load(path/'weights.pt',map_location='cpu',weights_only=True));obj.network.eval()
        return obj

    def predict_frame(self,hourly,history):
        if set(hourly.market_id)!=set(ZONES):raise ValueError('all four zones required')
        cube=HourlyCube.from_long(hourly,ZONES)
        if not len(cube.dates) or (cube.dates<=np.datetime64(self.data_cutoff)).any():raise ValueError('target dates must follow data cutoff')
        values=np.stack([self.predict_day(day,cube.values[i],history) for i,day in enumerate(cube.dates)])
        return QuarterCube(cube.dates,ZONES,values).to_long('predicted_price_eur_mwh')[PRED_COLUMNS]
