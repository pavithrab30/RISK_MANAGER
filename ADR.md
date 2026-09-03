# ADR: Multimodal Document Intelligence — Architecture & Key Trade-offs


## 1. Goal and non-negotiables

Answer questions over a small collection of visually rich PDFs (multi-column, figures, tables,
scans), where (a) an answer may need facts spread across several pages, (b) every part of the
answer cites the exact page/region it came from, and (c) the system says "not found" rather than
guessing when the corpus doesn't support an answer. Must run on a normal laptop or free-tier
cloud — no paid infrastructure.

The brief is explicit that a naive `chunk → embed → cosine top-k → prompt` pipeline is
insufficient. Every non-obvious decision below exists to address one of the three requirements
above, not to add complexity for its own sake.

## 2. System overview

```
PDF upload
   │
   ▼
┌─────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│   Docling    │──▶│  Chunking Service │──▶│  SQLite (metadata +   │
│  (layout+OCR │   │  (parent/child +  │   │  FTS5/BM25) + Chroma  │
│  +table+OCR) │   │  reference edges) │   │  (dense vectors)      │
└─────────────┘   └──────────────────┘   └───────────┬───────────┘
                                                        │
Question ──▶ Query Decomposition ──▶ Hybrid Search (RRF) ──▶ Graph      ▼
             (LLM, gated)            (dense + BM25)         Expansion  Reranker
                                                                 │        │
                                                                 └───┬────┘
                                                                     ▼
                                                          Generation (Groq/Llama)
                                                          + citation-marker parsing
                                                          + groundedness gate
                                                                     │
                                                                     ▼
                                                    Answer + per-sentence citations
                                                    (page + normalized bbox)
```

Layered codebase: `api/` (FastAPI routes, Pydantic contracts) → `services/` (ingestion,
chunking, generation, citation orchestration) → `retrieval/` (hybrid search, fusion, reranking,
decomposition, graph expansion) → `data/` (parsing backends, SQLite + Chroma stores). Routes
contain no business logic — they call one or two service methods and shape the response.

## 3. Ingestion: Docling over raw text extraction or pure vision-embedding retrieval

**Decision:** Parse every PDF with [Docling](https://github.com/docling-project/docling)
(layout model + TableFormer table-structure model + RapidOCR fallback), not
PyMuPDF/pdfplumber text extraction, and not a ColPali-style page-image-embedding pipeline as the
primary retrieval path.

**Why not raw text extraction:** PyMuPDF gives text + coordinates but no layout understanding —
multi-column pages interleave columns in extraction order, and tables come out as unstructured
text runs. That directly breaks region-level citation (a citation box drawn around interleaved
text from two columns is useless) and table-aware answering.

**Why not vision-embedding retrieval (ColPali/ColQwen) as the primary path:** it's a legitimate
answer to "cross-page retrieval" and "visual-document retrieval" and we considered it. Two things
ruled it out as *primary* given the constraints: (1) the vision-embedding models are heavy for a
CPU-only laptop / free-tier target, and (2) it gives you "this region looks relevant" rather than
an actual extracted cell value you can quote — worse for the citation-precision and table-cell
bonus requirements. We do use a page image + Gemini vision as a documented *fallback* for figures
Docling can't structure (see §7).

**What Docling actually buys us in one pass:** reading-order-aware block classification
(heading/text/table/figure/caption), table structure recognition (row/col position per cell), and
OCR auto-engaged for scanned pages — verified end-to-end on both a real arXiv PDF (Docling
correctly extracted "Table 2" as a linearized markdown table with the right row/column data — see
the live query transcript in the walkthrough) and a synthesized image-only PDF (0 extractable
text characters, confirmed OCR path triggered, `Document.is_scanned=True` set correctly).

**Left out, deliberately:** handwriting, non-Latin scripts, heavily skewed/rotated scans beyond
Docling's built-in deskew, and per-cell pixel bounding boxes for table cells (TableFormer exposes
per-cell row/col position and text, but on OCR-scanned documents all cells share the table's
overall bbox rather than individual cell geometry). For structured PDFs where Docling provides
distinct per-cell bboxes, citations highlight the exact cell region. For OCR-scanned tables,
the system computes a proportional row-height slice of the table bbox — precise to one row, not
one cell. See `corpus/README.md` for the concrete coverage this produced.

## 4. Chunking: structure-aware parent/child, not fixed-size windows

**Decision:** Child (retrieval) units are built by merging consecutive `TEXT` blocks up to a
~300-token budget, but a table, figure, or heading boundary *always* forces a flush — tables are
never split across chunks and never merged with surrounding prose. Parent (generation-context)
units are built at the *section* level using detected headings as boundaries, and are allowed to
span multiple pages.

**Why not fixed-size sliding windows** (the default RAG-tutorial move): a token-count window has
no idea where a table row or a sentence ends, so it can and will cut a table's header row off from
its data row, or split a sentence — both directly damage grounding and citation precision. The
extra bookkeeping (respecting Docling's own block boundaries) is worth it.

**Why parent/child specifically:** when a *child* chunk from page N matches the query, the model
is given the surrounding *parent* section text, which may extend onto page N+1 — this is one
concrete mechanism for cross-page answers, independent of retrieval fusion (§5). Verified in the
unit tests (`test_parent_chunk_spans_across_pages_when_section_does`) and observed for real: the
"Methods" section of a test document spanning two pages produced one parent chunk covering both.

**Reference edges:** a regex pass over each chunk's text finds "Figure 3" / "Table 2" / "Section
4.1"-style mentions and records them as edges (`ChunkRef` rows) from the mentioning chunk to a
label to be resolved at query time. This is the input to graph expansion (§5).

## 5. Retrieval: the actual core of the task

Plain top-k similarity on the original query embedding structurally cannot retrieve two facts
that live on different pages if a single query vector doesn't sit "between" them well — RRF fusion
by itself, or reranking by itself, doesn't fix that either; both still start from one embedding of
the original question. Four **stacked**, individually well-established techniques attack this
from different angles, chosen over any single "silver bullet":

1. **Hybrid search (dense + BM25, fused with Reciprocal Rank Fusion).** Dense embeddings
   (`BAAI/bge-small-en-v1.5`, local, CPU, free) catch paraphrase; SQLite FTS5/BM25 catches exact
   terms (model names, table headers, section numbers) dense embeddings often blur. RRF fuses by
   *rank*, not raw score, because BM25 and cosine similarity live on incomparable scales.
2. **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the union of
   candidates. This is the step that actually lets a cross-page answer assemble: it reorders
   candidates pulled from every page and every sub-query together, rather than reasoning about one
   page's top-k in isolation.
3. **Query decomposition.** A compound question ("compare X in the abstract with Y in the results
   table") is split by an LLM call into focused sub-questions, each retrieved independently, then
   merged — this directly targets facts that live on genuinely different pages/sections. Gated by
   a cheap heuristic (`QueryDecomposer.looks_compound`) so simple lookups don't pay for an extra
   LLM round-trip. **Measured live** on a real compound question against the Transformer paper:
   decomposition split it into 2 sub-queries and the final citations spanned pages 1, 2, 8, and 10
   (abstract → intro → results table → conclusion) with 100% citation coverage.
4. **Reference-graph expansion.** A lightweight, scoped alternative to a full document knowledge
   graph: chunks that textually reference "Figure 3"/"Table 2"/"Section 4.1" get that target
   pulled into the candidate pool even if it wouldn't rank on its own. We chose this over building
   an actual entity/relation graph because a real KG (entity extraction, coreference, relation
   typing) is disproportionate engineering for a one-week project and is much harder to evaluate
   cleanly than "does this regex-derived edge resolve to the right chunk" (which the unit tests
   check directly).

**Grounding / refusal:** two independent gates, not one. (a) *Retrieval-confidence gate* — if the
best reranker score is below a floor, the generation LLM is never called; there's nothing worth
answering from (also saves cost/latency). (b) *Post-generation citation-coverage gate* — the
generator is required to tag every factual sentence with a `[Sn]` marker pointing at a numbered
source; if the fraction of sentences with a *resolvable* marker falls below a threshold, the raw
answer is withheld and a typed refusal returned instead. This catches the model answering from
parametric knowledge despite the provided context, rather than trusting the model's own claimed
citations.

## 6. Storage: SQLite + Chroma, not Postgres/Qdrant

Chroma runs embedded (no separate server process) while still persisting to disk with metadata
filtering; SQLite's FTS5 virtual table gives BM25 keyword search "for free" out of the stdlib,
without standing up Elasticsearch/OpenSearch. Together this keeps the system a genuine
one-command run on a laptop, at the cost of not natively clustering/scaling (see §9). Vector and
keyword search are computed separately and fused in the retrieval layer rather than relying on
either store's built-in hybrid support, which also makes the fusion logic itself unit-testable in
isolation (`tests/unit/test_fusion.py`) without any store dependency.

## 7. Models and providers — all free-tier, and why these specifically

| Purpose | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, sentence-transformers) | Free, no rate limits, deterministic across eval runs — a hosted embedding API would make retrieval metrics non-reproducible run-to-run and adds per-chunk cost that scales with corpus size. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | ~80MB, CPU-friendly. `BAAI/bge-reranker-base` is a documented drop-in upgrade (`RERANKER_MODEL` env var) for a beefier machine. |
| Generation + decomposition | Groq (`llama-3.3-70b-versatile`) | Free tier, no card, very low latency (matters since decomposition adds a second LLM round-trip per question) and materially better answer quality than what fits on a CPU-only laptop via Ollama. |
| Eval judge + vision fallback | Gemini (`gemini-flash-lite-latest`) | A *different model family* than the generator specifically to avoid self-preference bias when judging faithfulness. Also used as a live vision fallback: when the top retrieved chunk is a FIGURE block (Docling extracted an image with no structured text), the rendered page PNG is sent directly to `GeminiClient.complete_with_image()` and the answer is returned with a page-level citation. This handles questions like "describe Figure 1" where there is no embeddable text to retrieve. |

**Resource footprint on this machine:** embedding + reranker + Docling's layout/table models
together are a few hundred MB, run entirely on CPU, and the whole app (backend + models) idles
under ~1GB RAM. No GPU required anywhere in the pipeline.

## 8. Evaluation — see `backend/eval/` and the results report

27-question gold set (`backend/eval/gold_set.yaml`) built from four real documents, with expected
`(document, page)` targets verified by hand against extracted page text — not guessed from paper
structure. Categories: single-page (baseline sanity), cross-page (same document, 2+ pages),
table-aware, cross-document (multi-document attribution bonus), OCR/scanned, and three refusal
questions (including one deliberately plausible-sounding trap). Metrics: Hit@k, MRR, page recall
(the metric that specifically rewards covering *all* gold pages, not just one) — computed for
three retrieval configurations side by side (naive dense-only baseline / +hybrid+rerank /
+decomposition+graph-expansion), plus an independent Gemini LLM-judge for faithfulness and
relevance, plus refusal-correctness. One command (`python -m eval.run_eval`) reproduces all of it;
see `backend/eval/report.md` for the actual numbers from the last run, including the failure-case
section.

**Two concrete changes made because of what the numbers/manual testing showed, both verified
before/after with real measurements, not asserted:**

1. **Hybrid+rerank over dense-only, kept as the default despite the extra latency/compute cost** —
   the eval run measured baseline (dense-only cosine) page recall at 0.708 versus 0.875 with
   BM25 fusion + cross-encoder reranking on, on the exact same indexed chunks. That's the number
   that justifies shipping the more expensive retrieval path rather than the simpler baseline the
   brief warns is "not enough."
2. **`RERANK_SCORE_FLOOR` lowered from -3.0 → -6.0 → -12.0, plus a chunking fix that attaches a
   table/figure's caption to the table/figure chunk itself.** Found via manual testing, not the
   gold-set run: a real scanned attendance sheet (single table, no separate caption) asked "list
   the students in the ECE department" and was incorrectly refused — the reranker scored the
   *correct* chunk -4.608, below the original -3.0 floor. Investigating further, the same
   underlying pattern (cross-encoders trained on prose-passage relevance systematically underscore
   raw tabular content) was independently visible in the gold-set run: `table_aware` was the
   single worst-performing category (page recall stuck at 0.600 across all three retrieval
   configs) partly because Docling frequently emits a table's caption ("Table 4: ...") as a
   separate text block next to the table rather than merged into it — the caption
   (natural-language, embeds well) was getting retrieved instead of the table itself (raw cell
   data, embeds poorly), and the reference-graph label lookup couldn't find the table either,
   since the label text lived only in the caption. Fixed both: `services/chunking_service.py` now
   attaches an adjacent caption to its table/figure chunk (verified: the fixed table chunk moved
   from absent-in-top-5 to rank 2, with the caption prepended), and the floor was progressively
   lowered from -3.0 → -6.0 based on the measured -4.608 false-negative score, then further
   to -12.0 after observing OCR-scanned table reranker scores consistently in the -7 to -10.5
   range — still a real gate since the post-generation citation-coverage gate (gate 2) catches
   hallucinations independently of gate 1.

## 9. What breaks at 10,000 documents / 1,000 concurrent users, and what changes

- **Embedded Chroma + single SQLite file** stop being appropriate well before 10k documents —
  SQLite's single-writer model and Chroma's embedded (non-clustered) index both become a
  bottleneck. Change: move to a clustered vector store (Qdrant/pgvector) and Postgres for
  metadata/FTS.
- **Ingestion is synchronous-in-a-background-task today** (`BackgroundTasks`), fine for a handful
  of uploads but not a durable queue — a crash mid-ingest loses the job with no retry. Change:
  a real task queue (Celery/RQ + Redis) so ingestion is retryable and horizontally scalable across
  workers.
- **CPU-only embedding/reranking** throughput is the ingestion bottleneck at scale (each
  document's chunks are embedded serially on one process). Change: batch embedding on a GPU worker
  pool, or use a hosted embedding API once volume justifies the cost.
- **Single FastAPI process** has no rate limiting, connection pooling tuning, or caching layer.
  Change: horizontal scaling behind a load balancer, a cache for repeated queries, and a proper
  connection pool once Postgres replaces SQLite.
- **Cost at scale:** free-tier Groq/Gemini rate limits would be exhausted well before 1,000
  concurrent users; a production deployment needs paid LLM API tiers (or self-hosted inference on
  GPU instances), which becomes the dominant cost line, not storage or compute for
  retrieval/embedding.

## 10. Things we'd defend differently if pushed

- The reference-graph expansion is intentionally shallow (regex-derived edges, not semantic
  relation extraction) — it's a documented, cheap complement to decomposition, not a replacement
  for a real graph, and we can point to exactly why (§5).
- Polling for document ingestion status (frontend) instead of WebSockets/SSE was a scope call, not
  an oversight — noted as the first thing to upgrade in §9's task-queue change.
- Per-cell table bounding boxes aren't available from Docling's TableFormer output on OCR-scanned
  documents (all cells share the same bbox as the whole table). For structured PDFs with distinct
  per-cell geometry, the system uses the actual cell bbox. For OCR tables, it computes a
  proportional row-height slice of the table bbox — precise to one row. We say so rather than
  fake pixel precision we don't have.
