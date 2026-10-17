"""ShapePipeline: baseline + residual model + fallback chain + exact mean projection.

P(h, q) = baseline(h, q) + residual_scale * residual(h, q)

The baseline preserves each hourly mean algebraically and the residual is
zero-sum, so the constraint holds by construction; the final float64
projection only removes rounding noise.
"""
from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .baselines import get_baseline
from .basis import decode, encode, get_basis
from .cube import HourlyCube, QuarterCube
from .features import FeatureConfig, build_features
from .history import AnalogueMemory, HistoryStore
from .models import MODELS
from .schema import HOUR_KEYS, HOURS, PRED_COLUMNS, QH_KEYS, QUARTERS

log = logging.getLogger(__name__)

ZONES = ("ZONE_0", "ZONE_1", "ZONE_2", "ZONE_3")
MANIFEST = "manifest.json"
HISTORY_STORE = "history_store.npz"


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "candidate"
    baseline: str = "pchip"
    basis: str = "poly"
    model: str = "lgbm_coef"
    model_params: dict = field(default_factory=dict)
    num_boost_round: int = 500
    features: FeatureConfig = field(default_factory=FeatureConfig)
    residual_scale: float = 1.0                 # global shrinkage of the learned residual
    recency_halflife_days: float | None = None  # optional recency weighting of training rows
    memory: dict | None = None                  # E4 analogue memory, e.g. {k: 16, window_days: 180}
    shrinkage: dict | None = None               # E6 cross-zone graph shrinkage, e.g. {lambda: 0.1}
    zones: tuple = ZONES
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        data = {k: v for k, v in dict(data).items() if k != "final"}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown candidate keys {sorted(unknown)}")
        if "features" in data:
            data["features"] = FeatureConfig.from_dict(data["features"])
        if "zones" in data:
            data["zones"] = tuple(data["zones"])
        if "model_params" in data:
            data["model_params"] = dict(data["model_params"] or {})
        return cls(**data)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["zones"] = list(self.zones)
        return out


def runtime_versions() -> dict:
    import lightgbm
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "lightgbm": lightgbm.__version__,
    }


def finalize(P: np.ndarray, H: np.ndarray, base: np.ndarray):
    """Fallback chain (learned -> baseline -> flat), then the exact hourly-mean projection.
    Returns P and the [.., hour] masks of hours that fell back."""
    P = np.array(P, dtype=float, copy=True)
    has_input = np.isfinite(H)
    to_baseline = ~np.isfinite(P).all(axis=-1) & has_input
    P[to_baseline] = base[to_baseline]
    to_flat = ~np.isfinite(P).all(axis=-1) & has_input
    P[to_flat] = np.repeat(H[..., None], QUARTERS, axis=-1)[to_flat]
    P += (H - P.mean(axis=-1))[..., None]
    return P, to_baseline, to_flat


def flat_predictions(hourly: pd.DataFrame) -> pd.DataFrame:
    """Emergency output: the official flat baseline, always compliant."""
    out = hourly.loc[hourly.index.repeat(QUARTERS), HOUR_KEYS + ["hourly_price_eur_mwh"]].reset_index(drop=True)
    out["quarter_index"] = np.tile(np.arange(1, QUARTERS + 1), len(hourly))
    out = out.rename(columns={"hourly_price_eur_mwh": "predicted_price_eur_mwh"})
    out["delivery_date"] = pd.to_datetime(out["delivery_date"]).dt.strftime("%Y-%m-%d")
    return out[PRED_COLUMNS]


class ShapePipeline:
    def __init__(self, config: PipelineConfig):
        if config.model not in MODELS:
            raise ValueError(f"unknown model {config.model!r}; choose from {sorted(MODELS)}")
        self.config = config
        self.basis = get_basis(config.basis)
        self.baseline = get_baseline(config.baseline)
        self.model = MODELS[config.model]({**config.model_params, "seed": config.seed}, config.num_boost_round)
        self.training_range: tuple[str, str] | None = None
        self.memory: AnalogueMemory | None = None   # scaler fitted on the training slice, then frozen
        self.shrink_scale: float | None = None      # graph similarity scale from training spreads

    @property
    def zones(self) -> tuple:
        return tuple(self.config.zones)

    @property
    def learned(self) -> bool:
        return self.config.model != "none"

    def _features(self, dates, H, base, store) -> pd.DataFrame:
        memory = self.memory.lookup(store, dates, H) if self.memory is not None else None
        return build_features(dates, H, encode(base - H[..., None], self.basis), store, self.config.features, memory)

    # ------------------------------------------------------------------ training
    def fit(self, cube: QuarterCube) -> "ShapePipeline":
        if cube.zones != self.zones:
            raise ValueError(f"training zones {cube.zones} differ from config zones {self.zones}")
        if len(cube.dates) == 0:
            raise ValueError("no training dates")
        if self.learned:
            H = cube.hourly()
            base = self.baseline(H)
            R = (cube.values - base).reshape(-1, QUARTERS)
            # Each row only sees history strictly before its own delivery date.
            store = HistoryStore.build(cube, self.config.baseline, self.basis)
            if self.config.memory is not None:
                self.memory = AnalogueMemory.fit(store, **self.config.memory)
            X = self._features(cube.dates, H, base, store)
            keep = np.isfinite(R).all(axis=1)
            weight = self._recency_weights(cube.dates)
            self.model.fit(
                X.loc[keep].reset_index(drop=True), R[keep], self.basis,
                None if weight is None else weight[keep],
            )
        if self.config.shrinkage is not None:
            self.shrink_scale = self._fit_shrink_scale(cube.hourly(), self.config.shrinkage.get("floor", 1.0))
        self.training_range = (str(cube.dates[0]), str(cube.dates[-1]))
        return self

    @staticmethod
    def _fit_shrink_scale(H: np.ndarray, floor: float) -> float:
        """Median |H_z - H_k| over training zone pairs and hours, with a fixed floor (EUR/MWh)."""
        spread = np.abs(H[:, :, None, :] - H[:, None, :, :])
        upper = np.triu_indices(H.shape[1], k=1)
        return float(max(np.nanmedian(spread[:, upper[0], upper[1], :]), floor))

    def _shrink(self, P: np.ndarray, H: np.ndarray) -> np.ndarray:
        """E6 graph shrinkage of each hour's [zone, k] deviation coefficients toward zones whose
        hourly prices are similar: C_s = solve(I + lambda L, C). Every zone stays zero-sum."""
        lam = float(self.config.shrinkage["lambda"])
        C = encode(P - H[..., None], self.basis)                       # [zone, 24, k]
        ok = np.isfinite(C).all(axis=(0, 2)) & np.isfinite(H).all(axis=0)
        if lam == 0.0 or not ok.any():
            return P
        Hh = H[:, ok].T                                                # [hour, zone]
        A = np.exp(-np.abs(Hh[:, :, None] - Hh[:, None, :]) / self.shrink_scale)
        A[:, np.arange(len(self.zones)), np.arange(len(self.zones))] = 0.0
        L = np.einsum("hz,zk->hzk", A.sum(-1), np.eye(len(self.zones))) - A
        Cs = np.linalg.solve(np.eye(len(self.zones)) + lam * L, C[:, ok].transpose(1, 0, 2))
        out = P.copy()
        out[:, ok] = H[:, ok, None] + decode(Cs.transpose(1, 0, 2), self.basis)
        return out

    def _recency_weights(self, dates: np.ndarray) -> np.ndarray | None:
        half_life = self.config.recency_halflife_days
        if not half_life:
            return None
        age = (dates[-1] - dates).astype(np.float64)
        w = 0.5 ** (age / float(half_life))
        return np.broadcast_to(w[:, None, None], (len(dates), len(self.zones), HOURS)).reshape(-1)

    # ----------------------------------------------------------------- inference
    def predict_day(self, day, H_day: np.ndarray, history: QuarterCube):
        """H_day: [zone, 24] -> P: [zone, 24, 4]. Sees only history strictly before `day`."""
        day = np.datetime64(day, "D")
        H = np.asarray(H_day, dtype=float)[None]
        base = self.baseline(H)
        residual = np.zeros_like(base)
        if self.learned:
            store = HistoryStore.build(history.before(day), self.config.baseline, self.basis)
            X = self._features(np.array([day]), H, base, store)
            residual = self.model.predict(X, self.basis).reshape(base.shape)
        P = base + self.config.residual_scale * residual
        if self.shrink_scale is not None:
            P = self._shrink(P[0], H[0])[None]
        P, to_baseline, to_flat = finalize(P, H, base)
        info = {"fallback_hours": to_baseline[0].sum(axis=-1), "flat_hours": to_flat[0].sum(axis=-1)}
        if self.learned and getattr(self.model, "last_bracketed", None) is not None:
            info["bracket_fail_hours"] = (~self.model.last_bracketed).reshape(H.shape[1:]).sum(axis=-1)
        return P[0], info

    def predict_frame(self, hourly: pd.DataFrame, history: QuarterCube):
        """Clean hourly input frame -> prediction frame in the official schema."""
        known = hourly["market_id"].isin(self.zones).to_numpy()
        frames = []
        diag = {"dates": [], "fallback_hours": 0, "flat_hours": 0, "unknown_zone_hours": int((~known).sum())}
        if known.any():
            cube = HourlyCube.from_long(hourly[known], self.zones)
            P = np.full(cube.values.shape + (QUARTERS,), np.nan)
            for i, day in enumerate(cube.dates):
                P[i], info = self.predict_day(day, cube.values[i], history)
                diag["dates"].append(str(day))
                diag["fallback_hours"] += int(info["fallback_hours"].sum())
                diag["flat_hours"] += int(info["flat_hours"].sum())
            frames.append(QuarterCube(cube.dates, cube.zones, P).to_long("predicted_price_eur_mwh"))
        if (~known).any():
            other = hourly[~known]
            log.warning("zones unseen in training %s get the parameter-free baseline",
                        sorted(other["market_id"].unique()))
            cube = HourlyCube.from_long(other, tuple(sorted(other["market_id"].unique())))
            base = self.baseline(cube.values)
            P, _, _ = finalize(base, cube.values, base)
            frames.append(QuarterCube(cube.dates, cube.zones, P).to_long("predicted_price_eur_mwh"))
        pred = pd.concat(frames, ignore_index=True).merge(hourly[HOUR_KEYS], on=HOUR_KEYS, how="inner")
        pred = pred.sort_values(QH_KEYS).reset_index(drop=True)
        pred["delivery_date"] = pred["delivery_date"].dt.strftime("%Y-%m-%d")
        return pred[PRED_COLUMNS], diag

    # --------------------------------------------------------------- persistence
    def save(self, directory, training_cube: QuarterCube, extra: dict | None = None) -> dict:
        """Everything inference needs, including the training prices as history store,
        so a short jury history never changes how lag features are computed."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        training_cube.save(directory / HISTORY_STORE)
        manifest = {
            "package_version": __version__,
            "config": self.config.to_dict(),
            "model": self.model.save(directory),
            "basis_matrix": self.basis.tolist(),
            "memory": self.memory.to_dict() if self.memory is not None else None,
            "shrink_scale": self.shrink_scale,
            "training_range": self.training_range,
            "versions": runtime_versions(),
            **(extra or {}),
        }
        (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, directory) -> tuple["ShapePipeline", QuarterCube]:
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        pipe = cls(PipelineConfig.from_dict(manifest["config"]))
        if not np.allclose(pipe.basis, np.asarray(manifest["basis_matrix"])):
            raise RuntimeError("basis defined in code differs from the trained artifact")
        pipe.model = MODELS[pipe.config.model].load(
            directory, manifest["model"], pipe.model.params, pipe.config.num_boost_round
        )
        pipe.shrink_scale = manifest.get("shrink_scale")
        if manifest.get("memory"):
            pipe.memory = AnalogueMemory.from_dict(manifest["memory"])
        if manifest.get("training_range"):
            pipe.training_range = tuple(manifest["training_range"])
        current = runtime_versions()
        for lib, version in manifest.get("versions", {}).items():
            if current.get(lib) != version:
                log.warning("%s %s differs from the training environment (%s)", lib, current.get(lib), version)
        return pipe, QuarterCube.load(directory / HISTORY_STORE)
