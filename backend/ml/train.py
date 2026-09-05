"""Run with python -m ml.train. No merchant documents or network access used."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

OUT = Path(__file__).parent
SEED = 42
FEATURE_NAMES = [
    "coverage",
    "critical_coverage",
    "matched_chunks",
    "id_conflicts",
    "amount_conflicts",
    "date_conflicts",
    "status_conflicts",
    "identity_coverage",
]


def synthetic_data(n=2400):
    """Feature-level simulation, not fabricated documents or real adjudications.

    Latent document completeness and reliability generate reviewer labels; noisy
    extraction creates observable features. Labels do not copy the score rule.
    """
    rng = np.random.default_rng(SEED)
    quality = rng.beta(2, 1.5, n)
    completeness = rng.binomial(5, quality) / 5
    critical = rng.binomial(1, np.clip(quality + 0.1, 0, 1))
    ids = rng.binomial(2, 0.9, n) / 2
    problems = rng.binomial(1, 0.1, (n, 4))
    authenticity = rng.binomial(1, 0.96, n)  # intentionally unobservable limitation
    labels = (
        (completeness >= 0.6)
        & (critical == 1)
        & (ids == 1)
        & (problems.sum(axis=1) == 0)
        & (authenticity == 1)
    ).astype(int)
    # Measurement noise represents OCR/extraction failures and missed conflicts.
    coverage = np.clip(completeness + rng.normal(0, 0.09, n), 0, 1)
    detected = problems * rng.binomial(1, 0.88, (n, 4))
    observed_critical = np.where(rng.random(n) < 0.04, 1 - critical, critical)
    chunks = rng.integers(1, 9, n)
    X = np.column_stack([coverage, observed_critical, chunks, detected, ids])
    return X, labels


def metrics(y, predicted, fp_cost, fn_cost):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "accuracy": accuracy_score(y, predicted),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0,
        "financial_cost": float(fp * fp_cost + fn * fn_cost),
        "cost_per_case": float((fp * fp_cost + fn * fn_cost) / len(y)),
    }


def export_model(model, name, threshold):
    data = {
        "selected_model": name,
        "model_type": name,
        "feature_names": FEATURE_NAMES,
        "threshold": threshold,
        "synthetic_demonstration": True,
        "seed": SEED,
    }
    if name == "logistic_regression":
        data.update(coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
    elif name == "random_forest":
        data["trees"] = []
        for estimator in model.estimators_:
            tree, nodes = estimator.tree_, []
            for i in range(tree.node_count):
                if tree.children_left[i] == -1:
                    value = tree.value[i][0]
                    nodes.append({"value": float(value[1] / value.sum())})
                else:
                    nodes.append(
                        {
                            "feature": int(tree.feature[i]),
                            "threshold": float(tree.threshold[i]),
                            "left": int(tree.children_left[i]),
                            "right": int(tree.children_right[i]),
                        }
                    )
            data["trees"].append(nodes)
    else:
        model.save_model(OUT / "xgboost_model.json")
    (OUT / "model_artifact.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-cost", type=float, default=100.0)
    parser.add_argument("--fn-cost", type=float, default=25.0)
    args = parser.parse_args()
    if not np.isfinite([args.fp_cost, args.fn_cost]).all() or min(args.fp_cost, args.fn_cost) < 0:
        parser.error("Costs must be finite non-negative amounts in the same currency.")
    X, y = synthetic_data()
    train, remaining = train_test_split(
        np.arange(len(y)), test_size=0.4, random_state=SEED, stratify=y
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, random_state=SEED, stratify=y[remaining]
    )
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=5, random_state=SEED, n_jobs=1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=1,
            colsample_bytree=1,
            random_state=SEED,
            n_jobs=1,
            eval_metric="logloss",
        ),
    }
    results, candidates = {}, []
    for name, model in models.items():
        model.fit(X[train], y[train])
        probabilities = model.predict_proba(X[validation])[:, 1]
        trials = []
        for threshold in (0.35, 0.5, 0.65, 0.8):
            measured = metrics(
                y[validation], probabilities >= threshold, args.fp_cost, args.fn_cost
            )
            trials.append({"threshold": threshold, **measured})
        best = max(trials, key=lambda r: (r["f1"], r["precision"], r["recall"]))
        results[name] = {"parameters": model.get_params(), "threshold_trials": trials, "best": best}
        candidates.append((best["f1"], best["precision"], best["recall"], name, best["threshold"]))
    _, _, _, selected, threshold = max(candidates)
    model = models[selected]  # Freeze model and threshold; do not refit or tune on test.
    test_predictions = (
        model.predict_proba(X[test])[:, 1] >= threshold
    )  # Single held-out evaluation.
    report = {
        "dataset": "SYNTHETIC DEMONSTRATION — feature simulation, not real merchant cases",
        "positive_class": "SUFFICIENT_EVIDENCE",
        "seed": SEED,
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "held_out_test": len(test),
        },
        "split_positive_counts": {
            k: int(y[v].sum())
            for k, v in (("train", train), ("validation", validation), ("held_out_test", test))
        },
        "selection_rule": "Maximum validation F1, then precision, then recall, then lexical model name; fixed threshold grid. No test tuning or refit.",
        "features": FEATURE_NAMES,
        "selected_model": selected,
        "selected_threshold": threshold,
        "validation": results,
        "held_out_test": metrics(y[test], test_predictions, args.fp_cost, args.fn_cost),
        "cost_assumptions": {
            "false_positive": args.fp_cost,
            "false_negative": args.fn_cost,
            "currency": "user-defined units",
        },
        "confusion_matrix_order": "Rows actual [INSUFFICIENT, SUFFICIENT]; columns predicted [INSUFFICIENT, SUFFICIENT]",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "limitations": "Simulated completeness/reliability labels with noisy extraction. No real-world precision, recall, fraud determination, or chargeback win-rate claim. Authenticity is unobservable. Repeated runs reproduce this demonstration, not independent test sets.",
    }
    split = {
        int(i): name
        for name, indices in (("train", train), ("validation", validation), ("held_out_test", test))
        for i in indices
    }
    with (OUT / "synthetic_demonstration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["synthetic_case_id", "split", *FEATURE_NAMES, "label", "synthetic"])
        for i in range(len(y)):
            writer.writerow([f"SYNTHETIC-{i:05}", split[i], *X[i], int(y[i]), True])
    report["dataset_sha256"] = hashlib.sha256(
        (OUT / "synthetic_demonstration.csv").read_bytes()
    ).hexdigest()
    export_model(model, selected, threshold)
    # Verify exported inference on validation only, leaving the test untouched.
    from app.services.risk_model import RiskModel

    runtime = RiskModel()
    for idx in validation[:30]:
        actual = runtime.predict(dict(zip(FEATURE_NAMES, X[idx])))["probability_sufficient"]
        assert abs(actual - float(model.predict_proba(X[idx : idx + 1])[0, 1])) < 0.0001
    (OUT / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# RiskRAG evaluation — SYNTHETIC DEMONSTRATION",
        "",
        report["dataset"],
        "",
        f"Seed: {SEED}. Train/validation/test: {len(train)}/{len(validation)}/{len(test)}.",
        report["selection_rule"],
        "",
        f"Selected: **{selected}**, threshold **{threshold}**.",
        "",
        "## Validation comparison",
    ]
    for name, result in results.items():
        b = result["best"]
        lines.append(
            f"- {name}: F1 {b['f1']:.4f}; precision {b['precision']:.4f}; recall {b['recall']:.4f}; threshold {b['threshold']}."
        )
    lines += ["", "## Held-out test (evaluated once after selection)"]
    lines += [f"- {key}: {value}" for key, value in report["held_out_test"].items()]
    lines += [
        "",
        report["confusion_matrix_order"],
        "",
        f"Costs: {report['cost_assumptions']}. FP = insufficient evidence predicted sufficient; FN = sufficient predicted insufficient.",
        "FPR = FP/(FP+TN); FNR = FN/(FN+TP); total cost = FP × FP cost + FN × FN cost.",
        "",
        "## Limits",
        report["limitations"],
        "",
        f"Dataset SHA256: {report['dataset_sha256']}",
        f"Versions: {report['versions']}",
    ]
    (OUT / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "test": report["held_out_test"]}, indent=2))


if __name__ == "__main__":
    main()
