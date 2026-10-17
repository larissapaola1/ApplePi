"""Small spatiotemporal graph network for mean-preserving price reconstruction.

Four zones form a learned, fully connected graph at each supplied hour.
Temporal convolutions operate on the 24-hour input curve, without crossing days.
All weights start from scratch. No graph geography or external observations.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from aet.baselines import cumulative
from aet.basis import encode, get_basis
from aet.cube import QuarterCube, HourlyCube
from aet.features import FeatureConfig, build_features
from aet.history import HistoryStore
from aet.schema import HOUR_KEYS, PRED_COLUMNS, QH_KEYS

ZONES = ("ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3")


@dataclass(frozen=True)
class Config:
    hidden: int = 32
    blocks: int = 2
    epochs: int = 80
    batch_days: int = 16
    learning_rate: float = .001
    weight_decay: float = .001
    dropout: float = .1
    seed: int = 42
    threads: int = 4
    graph: bool = True
    target_scale: float = 10.
    cross_zone: bool = False
    loss: str = "price"
    residual_scale: float = 1.
    local_skip: bool = False

    def __post_init__(self):
        if min(self.hidden, self.blocks, self.epochs, self.batch_days, self.threads) < 1:
            raise ValueError("network dimensions and training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.target_scale <= 0:
            raise ValueError("invalid optimization settings")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.loss not in {"price", "coefficient"}:
            raise ValueError("loss must be price or coefficient")
        if not np.isfinite(self.residual_scale) or not 0 <= self.residual_scale <= 2:
            raise ValueError("residual_scale must be finite and in [0,2]")


def seed_everything(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)


class GraphTemporalBlock(nn.Module):
    def __init__(self, hidden, dropout, graph=True):
        super().__init__()
        self.graph = graph
        self.temporal = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.mix = nn.Linear(hidden, hidden)
        self.norm_t = nn.LayerNorm(hidden)
        self.norm_g = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: batch days, zones, hours, hidden; no cross-date convolution.
        b, z, h, c = x.shape
        temporal = self.temporal(x.reshape(b*z, h, c).transpose(1, 2))
        temporal = temporal.transpose(1, 2).reshape(b, z, h, c)
        x = self.norm_t(x + self.dropout(torch.nn.functional.gelu(temporal)))
        if not self.graph:
            return x
        nodes = x.permute(0, 2, 1, 3)  # day, hour, zone, hidden
        logits = self.query(nodes) @ self.key(nodes).transpose(-1, -2) / c**.5
        attention = logits.softmax(dim=-1)
        message = self.mix(attention @ self.value(nodes)).permute(0, 2, 1, 3)
        return self.norm_g(x + self.dropout(torch.nn.functional.gelu(message)))


class TemporalGraphNetwork(nn.Module):
    def __init__(self, input_features, config):
        super().__init__()
        self.input = nn.Linear(input_features, config.hidden)
        self.blocks = nn.ModuleList([GraphTemporalBlock(config.hidden, config.dropout, config.graph)
                                     for _ in range(config.blocks)])
        self.head = nn.Linear(config.hidden, 3)
        # Begin exactly at PCHIP instead of adding arbitrary random price corrections.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.local_head = nn.Linear(input_features, 3) if config.local_skip else None
        if self.local_head is not None:
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
        self.register_buffer("basis", torch.tensor(get_basis("poly"), dtype=torch.float32))
        self.target_scale = config.target_scale

    def forward(self, x):
        local = self.local_head(x) if self.local_head is not None else 0
        x = torch.nn.functional.gelu(self.input(x))
        for block in self.blocks:
            x = block(x)
        return ((self.head(x) + local) @ self.basis.T) * self.target_scale


def feature_frame(dates, hourly, history, cross_zone=False):
    """Shared train/inference path: history lookups exclude each target date."""
    basis = get_basis("poly")
    base = cumulative(hourly)
    store = HistoryStore.build(history, "pchip", basis)
    # Original recipe uses graph messages alone; AET-inspired variants add explicit spreads/slopes.
    features = build_features(dates, hourly, encode(base-hourly[..., None], basis), store,
                              FeatureConfig(history=True, cross_zone=cross_zone, prev_day_edge=True))
    zone = features.pop("zone").to_numpy(dtype=int)
    for j in range(4):
        features[f"is_zone_{j}"] = (zone == j).astype(float)
    features["hour_sin"] = np.sin(2*np.pi*features.hour/24)
    features["hour_cos"] = np.cos(2*np.pi*features.hour/24)
    return features, base


class GraphReconstructor:
    def __init__(self, config=Config()):
        self.config = config
        self.network = None
        self.feature_names = []
        self.mean = self.scale = None
        self.training_range = None
        self.training_history = None
        self.losses = []

    def _tensor(self, frame, days):
        if list(frame.columns) != self.feature_names:
            raise ValueError("feature schema differs from fitted model")
        values = frame.to_numpy(float)
        missing = ~np.isfinite(values)
        normalized = (np.where(missing, self.mean, values)-self.mean)/self.scale
        normalized = np.clip(normalized, -8, 8)
        values = np.concatenate([normalized, missing.astype(float)], axis=-1)
        return torch.tensor(values.reshape(days, 4, 24, -1), dtype=torch.float32)

    def fit(self, cube):
        if cube.zones != ZONES or not len(cube.dates) or not np.isfinite(cube.values).all():
            raise ValueError("training requires complete official four-zone days")
        if cube.dates[0] < np.datetime64("2025-10-01") or cube.dates[-1] > np.datetime64("2026-09-30"):
            raise ValueError("training must stay within official Oct 2025–Sep 2026 package")
        seed_everything(self.config)
        frame, baseline = feature_frame(cube.dates, cube.hourly(), cube, self.config.cross_zone)
        self.feature_names = list(frame.columns)
        values = frame.to_numpy(float)
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        self.mean = np.divide(np.where(finite, values, 0).sum(axis=0), count,
                              out=np.zeros(values.shape[1]), where=count>0)
        deviations = np.where(finite, values-self.mean, 0)
        self.scale = np.sqrt(np.divide((deviations**2).sum(axis=0), count,
                                       out=np.zeros(values.shape[1]), where=count>0))
        self.scale[self.scale < 1e-6] = 1
        X = self._tensor(frame, len(cube.dates))
        target = torch.tensor(cube.values-baseline, dtype=torch.float32)
        self.network = TemporalGraphNetwork(X.shape[-1], self.config)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        generator = torch.Generator().manual_seed(self.config.seed)
        self.losses = []
        self.network.train()
        # Fixed epoch count; no scoring-date targets used for early stopping or selection.
        for epoch in range(self.config.epochs):
            order = torch.randperm(len(X), generator=generator)
            total = 0.
            for ids in order.split(self.config.batch_days):
                optimizer.zero_grad(set_to_none=True)
                # Direct L1 in decoded quarter-price space, not a coefficient surrogate.
                difference = self.network(X[ids])-target[ids]
                error = difference.abs().mean()
                objective = ((difference @ self.network.basis).abs().mean()
                             if self.config.loss == "coefficient" else error)
                (objective/self.config.target_scale).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
                optimizer.step()
                total += error.item()*len(ids)
            self.losses.append(total/len(X))
        self.network.eval()
        self.training_range = [str(cube.dates[0]), str(cube.dates[-1])]
        self.training_history = cube
        return self

    def predict_day(self, day, hourly, history):
        if self.network is None:
            raise RuntimeError("fit or load a model first")
        H = np.asarray(hourly, dtype=float)
        if H.shape != (4, 24) or not np.isfinite(H).all() or history.zones != ZONES:
            raise ValueError("expected complete finite [4 zones,24 hours] hourly input")
        # Cut history before building features even if a caller supplies extra dates.
        dates = np.array([np.datetime64(day, "D")])
        frame, base = feature_frame(dates, H[None], history.before(day), self.config.cross_zone)
        self.network.eval()
        with torch.no_grad():
            correction = self.network(self._tensor(frame, 1)).numpy()[0].astype(np.float64)
        if not np.isfinite(correction).all():
            raise FloatingPointError("non-finite graph correction")
        prediction = base[0]+self.config.residual_scale*correction
        # Float64 projection removes float32 decoding drift, without changing hourly input.
        prediction += (H-prediction.mean(axis=-1))[..., None]
        return prediction

    def predict_frame(self, hourly, history):
        if set(hourly.market_id) != set(ZONES):
            raise ValueError("graph inference requires all four anonymized zones")
        cube = HourlyCube.from_long(hourly, ZONES)
        if not np.isfinite(cube.values).all():
            raise ValueError("each date requires all 24 hours of all four zones")
        if (cube.dates <= np.datetime64(self.training_range[1])).any():
            raise ValueError("inference dates must follow the model training cutoff")
        values = np.stack([self.predict_day(day, cube.values[i], history)
                           for i, day in enumerate(cube.dates)])
        return QuarterCube(cube.dates, ZONES, values).to_long("predicted_price_eur_mwh")[PRED_COLUMNS]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save(self.network.state_dict(), path/"weights.pt")
        np.savez(path/"scaler.npz", mean=self.mean, scale=self.scale)
        self.training_history.save(path/"history.npz")
        manifest = {"config": asdict(self.config), "feature_names": self.feature_names,
                    "training_range": self.training_range, "training_losses": self.losses,
                    "torch": str(torch.__version__), "numpy": np.__version__,
                    "pandas": pd.__version__, "parameters": sum(p.numel() for p in self.network.parameters()),
                    "baseline": "cumulative_pchip", "basis": "poly", "zones": list(ZONES)}
        (path/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path/"manifest.json").read_text())
        if meta["zones"] != list(ZONES) or meta["basis"] != "poly":
            raise ValueError("unsupported model schema")
        obj = cls(Config(**meta["config"]))
        seed_everything(obj.config)
        obj.feature_names = meta["feature_names"]
        with np.load(path/"scaler.npz", allow_pickle=False) as a:
            obj.mean, obj.scale = a["mean"].copy(), a["scale"].copy()
        obj.training_range = meta["training_range"]
        obj.training_history = QuarterCube.load(path/"history.npz")
        obj.network = TemporalGraphNetwork(2*len(obj.feature_names), obj.config)
        obj.network.load_state_dict(torch.load(path/"weights.pt", map_location="cpu", weights_only=True))
        obj.network.eval()
        obj.losses = meta["training_losses"]
        return obj
