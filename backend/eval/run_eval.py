"""
One-command evaluation harness.

Run from backend/:
    ./.venv/Scripts/python -m eval.run_eval
    ./.venv/Scripts/python -m eval.run_eval --limit 5          # fast smoke run
    ./.venv/Scripts/python -m eval.run_eval --skip-judge        # no GEMINI_API_KEY needed

What it does, end to end:
  1. Ingests corpus/*.pdf into a dedicated eval index (idempotent - re-running
     skips documents already ingested, so repeat runs are fast).
  2. For every gold-set question, runs retrieval under three configurations:
       - baseline:      naive dense-only cosine top-k (the "not enough" pipeline
                         the assignment brief describes)
       - hybrid_rerank:  + BM25 fusion + cross-encoder reranking, decomposition
                         and graph-expansion OFF
       - full:           the complete pipeline (+ query decomposition + reference
                         graph expansion) - what's actually deployed
     and computes Hit@k, MRR, and page recall for each, so the retrieval
     ablation is a real measured comparison, not an assertion.
  3. For the full pipeline only, generates an answer, checks refusal
     correctness, and (unless --skip-judge) scores faithfulness/relevance
     with an independent Gemini judge.
  4. Writes eval/results.json (machine-readable, full per-item detail) and
     eval/report.md (the human-readable results report, including a failure
     case analysis) - both committed to the repo as the evaluation deliverable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))
load_dotenv(_BACKEND_DIR / ".env")

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.data.parsing.docling_parser import DoclingParser  # noqa: E402
from app.data.store.embedding import EmbeddingModel  # noqa: E402
from app.data.store.metadata_store import MetadataStore  # noqa: E402
from app.data.store.vector_store import VectorStore  # noqa: E402
from app.retrieval.graph_expansion import GraphExpander  # noqa: E402
from app.retrieval.hybrid_search import HybridSearcher  # noqa: E402
from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from app.retrieval.query_decomposition import QueryDecomposer  # noqa: E402
from app.retrieval.reranker import Reranker  # noqa: E402
from app.services.chunking_service import ChunkingService  # noqa: E402
from app.services.generation_service import GenerationService  # noqa: E402
from app.services.ingestion_service import IngestionService  # noqa: E402

from eval.baseline_retrieval import baseline_retrieve  # noqa: E402
from eval.metrics import GoldTarget, hit_at_k, page_recall, reciprocal_rank  # noqa: E402
from eval.judge import judge_answer  # noqa: E402
from eval.report import write_report  # noqa: E402

logger = get_logger(__name__)

CORPUS_DIR = _BACKEND_DIR.parent / "corpus"
EVAL_STORAGE = Path(__file__).parent / ".eval_storage"
GOLD_SET_PATH = Path(__file__).parent / "gold_set.yaml"
REPORT_PATH = Path(__file__).parent / "report.md"
RESULTS_JSON_PATH = Path(__file__).parent / "results.json"


def load_gold_set() -> list[dict]:
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_corpus_ingested(metadata_store: MetadataStore, ingestion: IngestionService) -> dict[str, str]:
    """Returns {filename: document_id}. Skips anything already ingested and
    ready, so re-running the harness after the first time is fast."""
    existing = {d.filename: d for d in metadata_store.list_documents()}
    filename_to_id: dict[str, str] = {}
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {CORPUS_DIR} - see corpus/README.md")

    for pdf_path in pdf_files:
        filename = pdf_path.name
        if filename in existing and existing[filename].status.value == "ready":
            filename_to_id[filename] = existing[filename].id
            print(f"  [skip]  {filename} already ingested")
            continue
        print(f"  [parse] {filename} ...")
        t0 = time.time()
        doc = ingestion.register_upload(filename, str(pdf_path.resolve()))
        ingestion.ingest(doc.id)
        refreshed = metadata_store.get_document(doc.id)
        if refreshed.status.value != "ready":
            raise RuntimeError(f"Failed to ingest {filename}: {refreshed.error_message}")
        print(f"  [done]  {filename} ({refreshed.num_pages} pages, {time.time() - t0:.1f}s)")
        filename_to_id[filename] = doc.id
    return filename_to_id


def to_refs(chunks, id_to_filename: dict[str, str]) -> list[tuple[str, int]]:
    return [(id_to_filename[c.chunk.document_id], c.chunk.page_number) for c in chunks]


def parse_targets(expected: list[dict]) -> list[GoldTarget]:
    return [GoldTarget(document=e["document"], pages=e["pages"]) for e in expected]


def main() -> None:
    argp = argparse.ArgumentParser(description="Run the DocIntel evaluation harness")
    argp.add_argument("--top-k", type=int, default=5, help="k for Hit@k and the retrieved-set size")
    argp.add_argument("--skip-judge", action="store_true", help="skip the Gemini LLM-judge pass")
    argp.add_argument("--limit", type=int, default=None, help="only run the first N gold items")
    args = argp.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    EVAL_STORAGE.mkdir(parents=True, exist_ok=True)

    print("=== DocIntel Evaluation Harness ===")
    print(f"Corpus:   {CORPUS_DIR}")
    print(f"Gold set: {GOLD_SET_PATH}")
    print(f"Index:    {EVAL_STORAGE}\n")

    metadata_store = MetadataStore(EVAL_STORAGE / "meta.sqlite3")
    vector_store = VectorStore(EVAL_STORAGE / "chroma")
    embedding_model = EmbeddingModel(settings.embedding_model)
    reranker = Reranker(settings.reranker_model)
    parser = DoclingParser()
    chunker = ChunkingService()
    ingestion = IngestionService(
        metadata_store, vector_store, embedding_model, parser, chunker, EVAL_STORAGE / "pages"
    )

    print("--- Ensuring corpus is ingested ---")
    filename_to_id = ensure_corpus_ingested(metadata_store, ingestion)
    id_to_filename = {v: k for k, v in filename_to_id.items()}

    gold_set = load_gold_set()
    if args.limit:
        gold_set = gold_set[: args.limit]
    print(f"\n--- Running {len(gold_set)} gold questions across 3 retrieval configs ---")

    hybrid_searcher = HybridSearcher(
        vector_store, metadata_store, embedding_model, settings.top_k_dense, settings.top_k_keyword
    )
    graph_expander = GraphExpander(metadata_store)

    groq_client = None
    gemini_client = None
    if settings.groq_api_key:
        from app.llm.groq_client import GroqClient

        groq_client = GroqClient(settings.groq_api_key, settings.groq_model)
    else:
        print("WARNING: GROQ_API_KEY not set - generation/refusal metrics will be skipped.")
    if settings.gemini_api_key:
        from app.llm.gemini_client import GeminiClient

        gemini_client = GeminiClient(settings.gemini_api_key, settings.gemini_model)

    # Groq's free-tier daily quota is easy to exhaust across a 27-question run
    # (decomposition + generation both call it); fall back to Gemini - an
    # independent quota - rather than let the whole generation phase go dark.
    generation_llm = None
    if groq_client is not None:
        from app.llm.fallback_client import FallbackLLMClient

        generation_llm = FallbackLLMClient(groq_client, gemini_client)
    elif gemini_client is not None:
        generation_llm = gemini_client

    pipeline_hybrid_rerank = RetrievalPipeline(
        metadata_store,
        hybrid_searcher,
        reranker,
        QueryDecomposer(None),
        graph_expander,
        query_decomposition_enabled=False,
        graph_expansion_enabled=False,
        top_k_reranked=args.top_k,
    )
    pipeline_full = RetrievalPipeline(
        metadata_store,
        hybrid_searcher,
        reranker,
        QueryDecomposer(generation_llm),
        graph_expander,
        query_decomposition_enabled=bool(generation_llm),
        graph_expansion_enabled=True,
        top_k_reranked=args.top_k,
    )

    generation_service = (
        GenerationService(generation_llm, min_coverage=settings.groundedness_min_coverage)
        if generation_llm
        else None
    )
    judge_client = None
    if not args.skip_judge and gemini_client is not None:
        judge_client = gemini_client
    elif not args.skip_judge:
        print("WARNING: GEMINI_API_KEY not set - faithfulness/relevance judging will be skipped.")

    documents_by_id = {doc_id: metadata_store.get_document(doc_id) for doc_id in id_to_filename}

    results = []
    for i, item in enumerate(gold_set, start=1):
        print(f"  [{i}/{len(gold_set)}] {item['id']}")
        targets = parse_targets(item["expected"])
        doc_ids_filter = None  # gold set always searches the full corpus, matching real usage

        baseline_chunks = baseline_retrieve(
            item["question"], vector_store, metadata_store, embedding_model, top_k=args.top_k
        )
        hybrid_result = pipeline_hybrid_rerank.retrieve(item["question"], doc_ids_filter)
        full_result = pipeline_full.retrieve(item["question"], doc_ids_filter)

        baseline_refs = to_refs(baseline_chunks, id_to_filename)
        hybrid_refs = to_refs(hybrid_result.chunks, id_to_filename)
        full_refs = to_refs(full_result.chunks, id_to_filename)

        record = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "should_refuse": item["should_refuse"],
            "expected": item["expected"],
            "retrieval": {
                "baseline": {
                    "retrieved": baseline_refs,
                    "hit_at_k": hit_at_k(baseline_refs, targets, args.top_k),
                    "mrr": reciprocal_rank(baseline_refs, targets),
                    "page_recall": page_recall(baseline_refs, targets),
                },
                "hybrid_rerank": {
                    "retrieved": hybrid_refs,
                    "hit_at_k": hit_at_k(hybrid_refs, targets, args.top_k),
                    "mrr": reciprocal_rank(hybrid_refs, targets),
                    "page_recall": page_recall(hybrid_refs, targets),
                },
                "full": {
                    "retrieved": full_refs,
                    "hit_at_k": hit_at_k(full_refs, targets, args.top_k),
                    "mrr": reciprocal_rank(full_refs, targets),
                    "page_recall": page_recall(full_refs, targets),
                    "sub_queries": full_result.sub_queries,
                },
            },
        }

        if generation_service is not None:
            # A single rate-limited/quota-exhausted LLM call must not take down the
            # whole run and lose every item scored so far - record the failure on
            # this item and move on. (Real bug hit during development: Groq's free
            # daily token quota was exhausted by combined manual testing + eval
            # runs mid-way through a run, and an uncaught LLMProviderError here
            # crashed the script before results.json/report.md were ever written.)
            try:
                generation = generation_service.generate(
                    item["question"], full_result.chunks, documents_by_id
                )
                refused = generation.refused
                record["generation"] = {
                    "answer": generation.answer_text,
                    "refused": refused,
                    "refusal_correct": refused == item["should_refuse"],
                    "groundedness_coverage": generation.groundedness_coverage,
                    "num_citations": len(generation.citations),
                    "keyword_hits": _keyword_hits(generation.answer_text, item.get("answer_keywords", [])),
                }
                if judge_client is not None and not item["should_refuse"]:
                    try:
                        source_texts = [c.chunk.text for c in full_result.chunks]
                        record["judge"] = judge_answer(
                            judge_client, item["question"], generation.answer_text, source_texts
                        )
                    except Exception as exc:
                        print(f"    [warn] judge call failed for {item['id']}: {exc}")
                        record["judge"] = {"faithfulness": None, "relevance": None, "reasoning": f"judge error: {exc}"}
            except Exception as exc:
                print(f"    [warn] generation failed for {item['id']}: {exc}")
                record["generation_error"] = str(exc)

        # Write after every item, not just at the end, so a later failure never
        # loses more than the one in-flight item.
        results.append(record)
        RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nWrote machine-readable results to {RESULTS_JSON_PATH}")

    write_report(results, REPORT_PATH, top_k=args.top_k)
    print(f"Wrote human-readable report to {REPORT_PATH}")


def _keyword_hits(answer: str, keywords: list[str]) -> dict:
    answer_low = answer.lower()
    hits = {kw: kw.lower() in answer_low for kw in keywords}
    return {"hits": hits, "coverage": (sum(hits.values()) / len(hits)) if hits else None}


if __name__ == "__main__":
    main()
