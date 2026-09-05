# RiskRAG evaluation — SYNTHETIC DEMONSTRATION

SYNTHETIC DEMONSTRATION — feature simulation, not real merchant cases

Seed: 42. Train/validation/test: 1440/480/480.
Maximum validation F1, then precision, then recall, then lexical model name; fixed threshold grid. No test tuning or refit.

Selected: **logistic_regression**, threshold **0.5**.

## Validation comparison
- logistic_regression: F1 0.9068; precision 0.9145; recall 0.8992; threshold 0.5.
- random_forest: F1 0.8945; precision 0.8983; recall 0.8908; threshold 0.5.
- xgboost: F1 0.8992; precision 0.8992; recall 0.8992; threshold 0.65.

## Held-out test (evaluated once after selection)
- precision: 0.8869565217391304
- recall: 0.864406779661017
- f1: 0.8755364806866953
- accuracy: 0.9395833333333333
- confusion_matrix: [[349, 13], [16, 102]]
- false_positive_rate: 0.03591160220994475
- false_negative_rate: 0.13559322033898305
- financial_cost: 1700.0
- cost_per_case: 3.5416666666666665

Rows actual [INSUFFICIENT, SUFFICIENT]; columns predicted [INSUFFICIENT, SUFFICIENT]

Costs: {'false_positive': 100.0, 'false_negative': 25.0, 'currency': 'user-defined units'}. FP = insufficient evidence predicted sufficient; FN = sufficient predicted insufficient.
FPR = FP/(FP+TN); FNR = FN/(FN+TP); total cost = FP × FP cost + FN × FN cost.

## Limits
Simulated completeness/reliability labels with noisy extraction. No real-world precision, recall, fraud determination, or chargeback win-rate claim. Authenticity is unobservable. Repeated runs reproduce this demonstration, not independent test sets.

Dataset SHA256: 90a5ed1a3e12f8c0050d3571cee53c6fbb9b279461aa766c48b6d245371af2a5
Versions: {'python': '3.12.14', 'numpy': '2.2.6', 'sklearn': '1.7.2', 'xgboost': '3.4.1'}
