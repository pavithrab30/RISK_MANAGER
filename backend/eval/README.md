# Chargeback evaluation

This directory exposes the single evaluation entry point for RiskRAG:

```bash
python -m eval.run_eval --fp-cost 100 --fn-cost 25
```

The run creates fixed-seed synthetic chargeback cases, compares Logistic Regression, Random Forest, and XGBoost on validation data, selects by validation F1, and evaluates the selected model once on the held-out test split.

The authoritative generated artifacts are:

- `backend/ml/synthetic_demonstration.csv`
- `backend/ml/model_artifact.json`
- `backend/ml/evaluation_report.json`
- `backend/ml/evaluation_report.md`

The dataset is synthetic demonstration data. Its metrics do not establish production performance.
