"""Small inference adapter for the evaluated chargeback evidence classifier."""

from __future__ import annotations

import json
import math
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2] / "ml" / "model_artifact.json"


class RiskModel:
    def __init__(self, path: Path = ARTIFACT):
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.path = path

    def predict(self, features: dict) -> dict:
        values = [float(features[n]) for n in self.data["feature_names"]]
        if self.data["model_type"] == "logistic_regression":
            z = self.data["intercept"] + sum(
                a * b for a, b in zip(self.data["coefficients"], values)
            )
            probability = 1 / (1 + math.exp(-max(-40, min(40, z))))
        elif self.data["model_type"] == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(self.path.parent / "xgboost_model.json")
            probability = float(model.predict_proba([values])[0, 1])
        else:
            # Selected non-linear estimators are serialized as shallow JSON trees.
            votes = [self._tree(tree, values) for tree in self.data["trees"]]
            probability = sum(votes) / len(votes)
        threshold = self.data["threshold"]
        return {
            "label": "SUFFICIENT_EVIDENCE" if probability >= threshold else "INSUFFICIENT_EVIDENCE",
            "probability_sufficient": round(probability, 4),
            "threshold": threshold,
            "model": self.data["selected_model"],
            "disclaimer": "Demonstration classifier trained only on synthetic feature data; it does not validate evidence authenticity.",
        }

    def _tree(self, nodes, values, index=0):
        node = nodes[index]
        if "value" in node:
            return node["value"]
        return self._tree(
            nodes,
            values,
            node["left"] if values[node["feature"]] <= node["threshold"] else node["right"],
        )
