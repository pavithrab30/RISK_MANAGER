# RiskRAG frontend

React and TypeScript interface for merchant chargeback evidence review.

## Workflow

- Upload and select the PDFs for one chargeback case.
- Enter network reason code, claim, identifiers, amount, and optional status/date fields.
- Review evidence score, risk, recommendation, classifier output, evidence requirements, gaps, contradictions, cited passages, and grounded draft.
- Open a cited PDF page and region for merchant verification.

Every draft requires merchant approval and the interface performs no chargeback submission.

## Run

```powershell
npm ci
Copy-Item .env.example .env
npm run dev
```

Tests and production build:

```powershell
npm test
npm run build
```

`VITE_API_BASE_URL` defaults to `http://localhost:8000`.
