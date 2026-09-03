# DocIntel frontend

React + TypeScript + Vite. See the [repo root README](../README.md) for setup and run
instructions, and [ADR.md](../ADR.md) for the architecture.

Quick reference:

```bash
npm install
cp .env.example .env    # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev              # dev server, http://localhost:5173
npm run build             # type-check + production build
npm run lint
```

## Structure

- `src/api/` — typed client (`client.ts`) and request/response contracts (`types.ts`) mirroring
  `backend/app/api/schemas.py`. Kept in sync by hand rather than via OpenAPI codegen, given the
  one-week scope.
- `src/components/` — `DocumentPanel` (upload + status), `ChatPanel` + `AnswerMessage` (ask /
  answer / citation chips), `PdfViewer` (pdf.js render + normalized-bbox highlight overlay - the
  region-level citation UI).
- `src/hooks/useDocuments.ts` — upload + status-polling for ingestion (a deliberately simple
  stand-in for a push-based status channel; see ADR.md §9).
