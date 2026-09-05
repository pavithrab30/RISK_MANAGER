import json

import numpy as np

from app.services.risk_model import ARTIFACT, RiskModel
from app.services.risk_service import FEATURE_NAMES
from ml.train import metrics, synthetic_data


def test_synthetic_generator_reproducible_without_label_feature():
    a, labels = synthetic_data(40)
    b, other_labels = synthetic_data(40)
    assert np.array_equal(a, b) and np.array_equal(labels, other_labels)
    assert a.shape == (40, len(FEATURE_NAMES))
    assert "label" not in FEATURE_NAMES


def test_costs_and_confusion_convention():
    actual = metrics([0, 0, 1, 1], [0, 1, 0, 1], 100, 25)
    assert actual["confusion_matrix"] == [[1, 1], [1, 1]]
    assert actual["false_positive_rate"] == actual["false_negative_rate"] == 0.5
    assert actual["financial_cost"] == 125


def test_shipped_model_and_split_manifest():
    import csv

    model = RiskModel()
    result = model.predict(dict(zip(FEATURE_NAMES, [1, 1, 5, 0, 0, 0, 0, 1])))
    assert 0 <= result["probability_sufficient"] <= 1
    assert "synthetic" in result["disclaimer"]
    report = json.loads((ARTIFACT.parent / "evaluation_report.json").read_text())
    assert report["selected_model"] == model.data["selected_model"]
    with (ARTIFACT.parent / "synthetic_demonstration.csv").open() as f:
        rows = list(csv.DictReader(f))
    groups = {
        name: {r["synthetic_case_id"] for r in rows if r["split"] == name}
        for name in report["split_sizes"]
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["held_out_test"]
    assert not groups["validation"] & groups["held_out_test"]
    assert {k: len(v) for k, v in groups.items()} == report["split_sizes"]
