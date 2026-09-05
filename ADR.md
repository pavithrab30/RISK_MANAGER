# RiskRAG architecture decisions

## Objective

RiskRAG reduces merchant chargeback losses by retrieving case evidence, checking reason-specific requirements, detecting contradictions, predicting evidence sufficiency, and preparing a source-grounded draft for mandatory merchant review.

## Safety boundaries

The system is defense-only. It cannot fabricate or alter evidence, access payment rails, accuse a cardholder, or submit a dispute. Retrieval passages are candidates until a merchant verifies authenticity and relevance. Contradictions force manual review; missing critical evidence forces evidence gathering.

## Evidence ingestion and citations

Docling parses PDF layout, tables, figures, and OCR text. Parent/child chunks preserve section context. Every child chunk retains document ID, page, reading order, block type, and normalized page bounding box, allowing reviewers to open the cited region.

## Retrieval

RiskRAG combines dense semantic retrieval with SQLite FTS5/BM25. Reciprocal Rank Fusion merges both rankings. Compound evidence questions can be decomposed, explicit references can expand the candidate set, and a cross-encoder reranks the merged passages. The risk endpoint always scopes retrieval to documents selected for the case.

## Verification

The supplied chargeback CSV maps network reason codes to evidence requirements. It is reference guidance, never merchant evidence and never an ML label. A retrieved passage must contain the case order ID or transaction ID before it can contribute. Explicit field extraction checks IDs, amounts, dates, delivery status, refund status, and disagreements between records.

Candidate requirement support uses conservative affirmative-sentence token overlap. Negated, missing, pending, hypothetical, and template language is rejected. Conflicted passages cannot satisfy requirements or enter the draft.

## Decision policy

The evidence score combines requirement coverage, all-critical coverage, identifier coverage, and contradiction penalties. Recommendations prioritize contradictions, then missing critical evidence, then score and classifier output. `AUTO_RESPOND` means draft-ready for merchant review, never automatic submission.

## Classifier evaluation

The reference CSV has no labeled cases, so model development uses an explicitly synthetic feature-level dataset with seed 42 and stratified train/validation/held-out splits. Logistic Regression, Random Forest, and XGBoost are fit on train data and compared only on validation metrics. The selected model is evaluated once on the held-out test. Reports include precision, recall, F1, accuracy, confusion matrix, FPR, FNR, and configurable FP/FN cost.

The synthetic benchmark demonstrates methodology only. Production claims require independently labeled merchant cases, locked preprocessing, reason-code and merchant-segment analysis, calibration, drift monitoring, and a new untouched test set.

## Storage and deployment

SQLite stores metadata and keyword indexes; Chroma stores vectors; source PDFs and page images stay in configured local storage. FastAPI provides the chargeback API and React provides the review interface. Docker Compose supports local deployment. A production deployment requires authentication, tenant isolation, encryption, retention controls, and audit logging.
