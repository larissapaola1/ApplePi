"""Frozen convex blend of AET and graph predictions; no inference-time fitting."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

from .temporal_graph import GraphReconstructor, ZONES
from aet.pipeline import ShapePipeline
from aet.schema import HOUR_KEYS, QH_KEYS, PRED_COLUMNS, validate_predictions


class HybridReconstructor:
    def __init__(self, graph, aet, history, graph_weight):
        if not np.isfinite(graph_weight) or not 0 <= graph_weight <= 1:
            raise ValueError("graph weight must be finite and in [0,1]")
        if tuple(aet.zones) != ZONES or list(aet.training_range) != graph.training_range:
            raise ValueError("component zones and training cutoffs must match")
        self.graph, self.aet = graph, aet
        self.training_history = history.merge(graph.training_history)
        self.training_range = graph.training_range
        self.graph_weight = float(graph_weight)

    def predict_frame(self, hourly, history):
        graph = self.graph.predict_frame(hourly, history)
        aet, diag = self.aet.predict_frame(hourly, history)
        if diag["fallback_hours"] or diag["flat_hours"]:
            raise RuntimeError("AET fell back; refusing to silently change the blend")
        aet["delivery_date"] = pd.to_datetime(aet.delivery_date)
        merged = graph.merge(aet, on=QH_KEYS, validate="one_to_one", suffixes=("_g", "_a"))
        merged["predicted_price_eur_mwh"] = (
            self.graph_weight*merged.predicted_price_eur_mwh_g +
            (1-self.graph_weight)*merged.predicted_price_eur_mwh_a)
        merged = merged.merge(hourly, on=HOUR_KEYS, validate="many_to_one")
        avg = merged.groupby(HOUR_KEYS).predicted_price_eur_mwh.transform("mean")
        merged["predicted_price_eur_mwh"] += merged.hourly_price_eur_mwh-avg
        result = merged[PRED_COLUMNS].sort_values(QH_KEYS).reset_index(drop=True)
        validate_predictions(result, hourly)
        return result

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path/"manifest.json").read_text())
        if meta.get("kind") != "aet_graph_blend":
            raise ValueError("not a hybrid artifact")
        graph = GraphReconstructor.load(path/"graph")
        aet, history = ShapePipeline.load(path/"aet")
        return cls(graph, aet, history, meta["graph_weight"])


def save_hybrid(path, graph, aet_checkpoint, weight, selection):
    aet, history = ShapePipeline.load(aet_checkpoint)
    HybridReconstructor(graph, aet, history, weight)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    graph.save(path/"graph")
    shutil.copytree(aet_checkpoint, path/"aet")
    (path/"manifest.json").write_text(json.dumps({"kind":"aet_graph_blend",
        "graph_weight":float(weight), "training_range":graph.training_range,
        "selection":selection}, indent=2)+"\n")
