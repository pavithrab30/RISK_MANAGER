"""
Answer synthesis + groundedness enforcement.

Two separate gates keep the system from guessing when the documents don't
support an answer (a hard requirement in the brief):

  1. Retrieval-confidence gate: if the reranker's top score for the best
     candidate is below RERANK_SCORE_FLOOR, we never even call the
     generation LLM - there's nothing worth answering from.
  2. Post-generation citation-coverage gate: the LLM is instructed to tag
     every factual sentence with a [Sn] marker; if coverage falls below
     GROUNDEDNESS_MIN_COVERAGE the answer is withheld.

  Vision fallback: if ALL top retrieved chunks are FIGURE blocks (Docling
  extracted them as images with no structured text), and a GeminiClient with
  vision support is available, we send the rendered page PNG directly to
  Gemini and return its answer with a page-level citation. This handles
  questions like "describe Figure 1" or "what does the architecture diagram
  show?" where the figure has no embeddable text.

RERANK_SCORE_FLOOR lowered from -3.0 -> -6.0 -> -12.0 progressively based
on measured false negatives on OCR/table content. See ADR §8.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.data.models import BBox, BlockType, Citation, Document, RetrievedChunk
from app.llm.base import LLMClient
from app.services.citation_service import extract_citations, strip_markers

logger = get_logger(__name__)

RERANK_SCORE_FLOOR = -12.0

_SYSTEM_PROMPT = """You are a document question-answering assistant. You are given a user \
question and a set of numbered source passages extracted from one or more PDF documents, each \
labelled with its document name and page number.

Rules:
1. Answer ONLY using the information in the numbered sources below. Do not use outside knowledge.
2. After every sentence that states a fact from the sources, add the source marker(s) it came \
from, like this: "Revenue grew 12% in Q3 [S2]." If a sentence draws on multiple sources, cite all \
of them: "...as shown in both the table and the summary [S1][S4]."
3. If the sources do not contain enough information to answer the question, say so explicitly \
("The provided documents do not contain enough information to answer this.") - do not guess or \
fill gaps from general knowledge.
4. Be concise and directly answer the question first; do not restate the question.
5. When a source is a table, each data row is prefixed with [row N] where N is the row number. \
For every fact you state from a table row, cite BOTH the source and the row like this: \
"KARTHIKEYAN M is in ECE A [S1, row 19]." When listing multiple rows, cite each one individually \
with its own row number: "DHARSHIN M (ECE B) [S1, row 20], GAYATHIRI R (ECE B) [S1, row 21]." \
Always include the row number so the reader can locate the exact cell in the table."""

_VISION_SYSTEM_PROMPT = """You are a document question-answering assistant. You are given a \
question and a page image from a PDF document. Answer the question based ONLY on what you can \
see in the image. Be concise and factual. If the image does not contain enough information to \
answer, say so explicitly ("The provided documents do not contain enough information to answer \
this.")"""


@dataclass
class GenerationResult:
    answer_text: str
    citations: list[Citation]
    groundedness_coverage: float
    refused: bool
    refusal_reason: str | None = None
    raw_answer: str = ""


class GenerationService:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        min_coverage: float = 0.6,
        rerank_score_floor: float = RERANK_SCORE_FLOOR,
        vision_client=None,  # GeminiClient instance, optional
        page_image_dir: Path | None = None,
    ):
        self._llm = llm_client
        self.min_coverage = min_coverage
        self.rerank_score_floor = rerank_score_floor
        self._vision_client = vision_client
        self._page_image_dir = page_image_dir

    def generate(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        documents_by_id: dict[str, Document],
    ) -> GenerationResult:
        if not retrieved:
            return GenerationResult(
                answer_text="The provided documents do not contain enough information to answer this.",
                citations=[],
                groundedness_coverage=0.0,
                refused=True,
                refusal_reason="no_candidates_retrieved",
            )

        top_score = max((c.rerank_score or -999) for c in retrieved)
        if top_score < self.rerank_score_floor:
            logger.info(
                "generation_refused_low_confidence", query=query, top_rerank_score=top_score
            )
            return GenerationResult(
                answer_text="The provided documents do not contain enough information to answer this.",
                citations=[],
                groundedness_coverage=0.0,
                refused=True,
                refusal_reason="low_retrieval_confidence",
            )

        # --- Vision fallback ---
        # Trigger if: (a) top chunk is a FIGURE block, OR (b) query asks about
        # a figure/diagram/chart and a figure chunk exists in the retrieved set
        figure_rcs: list[RetrievedChunk] = []
        top_chunk = retrieved[0].chunk
        if top_chunk.block_type == BlockType.FIGURE:
            # Collect all figure chunks on the same page for bbox union
            figure_rcs = [rc for rc in retrieved
                         if rc.chunk.block_type == BlockType.FIGURE
                         and rc.chunk.page_number == top_chunk.page_number
                         and rc.chunk.document_id == top_chunk.document_id]
        elif self._vision_client is not None and self._is_figure_query(query):
            for rc in retrieved:
                if rc.chunk.block_type == BlockType.FIGURE:
                    figure_rcs = [rc2 for rc2 in retrieved
                                 if rc2.chunk.block_type == BlockType.FIGURE
                                 and rc2.chunk.page_number == rc.chunk.page_number
                                 and rc2.chunk.document_id == rc.chunk.document_id]
                    break

        if figure_rcs and self._vision_client is not None and self._page_image_dir is not None:
            vision_result = self._try_vision_fallback(query, figure_rcs, documents_by_id)
            if vision_result is not None:
                return vision_result

        # --- Normal text generation ---
        prompt = self._build_prompt(query, retrieved, documents_by_id)
        raw_answer = self._llm.complete(_SYSTEM_PROMPT, prompt, temperature=0.0)

        citations, cited_sentences, total_sentences = extract_citations(raw_answer, retrieved)
        for citation in citations:
            doc = documents_by_id.get(citation.document_id)
            citation.document_name = doc.filename if doc else citation.document_id

        coverage = cited_sentences / total_sentences if total_sentences else 0.0
        display_text = strip_markers(raw_answer)

        is_self_reported_refusal = "do not contain enough information" in raw_answer.lower()
        if not is_self_reported_refusal and coverage < self.min_coverage:
            logger.warning(
                "generation_below_groundedness_threshold",
                query=query,
                coverage=coverage,
                min_coverage=self.min_coverage,
            )
            return GenerationResult(
                answer_text=(
                    "The model's answer could not be sufficiently verified against the source "
                    "documents, so it is withheld. Partial answer for reference: " + display_text
                ),
                citations=citations,
                groundedness_coverage=coverage,
                refused=True,
                refusal_reason="low_citation_coverage",
                raw_answer=raw_answer,
            )

        return GenerationResult(
            answer_text=display_text,
            citations=citations,
            groundedness_coverage=coverage,
            refused=is_self_reported_refusal,
            refusal_reason="model_reported_insufficient" if is_self_reported_refusal else None,
            raw_answer=raw_answer,
        )

    @staticmethod
    def _is_figure_query(query: str) -> bool:
        """Returns True if the query is asking about a visual element."""
        q = query.lower()
        return any(kw in q for kw in [
            "figure", "diagram", "image", "chart", "graph", "plot",
            "architecture diagram", "show", "illustrat", "depict", "visual",
        ])

    def _try_vision_fallback(
        self,
        query: str,
        figure_rcs: list[RetrievedChunk],
        documents_by_id: dict[str, Document],
    ) -> GenerationResult | None:
        """Send the page image to Gemini vision. Accepts a list of figure chunks
        on the same page — their bboxes are unioned so split figures (e.g.
        left/right halves of Figure 2) are highlighted together."""
        top_rc = figure_rcs[0]
        chunk = top_rc.chunk
        image_path = (
            self._page_image_dir / chunk.document_id / f"page_{chunk.page_number}.png"
        )
        if not image_path.exists():
            logger.warning(
                "vision_fallback_image_missing",
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                path=str(image_path),
            )
            return None

        try:
            answer_text = self._vision_client.complete_with_image(
                _VISION_SYSTEM_PROMPT,
                f'Question: "{query}"\nAnswer based only on the page image above.',
                str(image_path),
            )
            logger.info(
                "vision_fallback_used",
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                num_figure_chunks=len(figure_rcs),
            )
        except Exception as exc:
            logger.warning("vision_fallback_failed", error=str(exc))
            return None

        is_refusal = "do not contain enough information" in answer_text.lower()

        doc = documents_by_id.get(chunk.document_id)
        doc_name = doc.filename if doc else chunk.document_id

        # Union bboxes of all figure chunks on this page so split figures
        # (e.g. Docling splitting Figure 2 into left/right halves) are
        # highlighted as one combined region
        combined_bbox = BBox.union([rc.chunk.bbox for rc in figure_rcs])

        citation = Citation(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_name=doc_name,
            page_number=chunk.page_number,
            bbox=combined_bbox,
            snippet=f"Figure on page {chunk.page_number}",
            row_number=None,
        )

        return GenerationResult(
            answer_text=answer_text,
            citations=[] if is_refusal else [citation],
            groundedness_coverage=0.0 if is_refusal else 1.0,
            refused=is_refusal,
            refusal_reason="model_reported_insufficient" if is_refusal else None,
            raw_answer=answer_text,
        )

    @staticmethod
    def _build_prompt(
        query: str, retrieved: list[RetrievedChunk], documents_by_id: dict[str, Document]
    ) -> str:
        lines = [f'Question: "{query}"', "", "Sources:"]
        for i, rc in enumerate(retrieved, start=1):
            doc = documents_by_id.get(rc.chunk.document_id)
            doc_name = doc.filename if doc else rc.chunk.document_id
            chunk_text = GenerationService._format_chunk_text(rc.chunk)
            lines.append(
                f"[S{i}] (Document: {doc_name}, Page {rc.chunk.page_number}, "
                f"Section: {rc.chunk.section_path or 'n/a'})\n{chunk_text}\n"
            )
        lines.append("Answer the question using only the sources above, with [Sn] citations.")
        return "\n".join(lines)

    @staticmethod
    def _format_chunk_text(chunk) -> str:
        """For table chunks, prefix each data row with [row N] so the LLM
        can cite specific rows. For other block types, return text as-is."""
        if chunk.block_type != BlockType.TABLE:
            return chunk.text

        lines = chunk.text.splitlines()
        if not lines:
            return chunk.text

        result = []
        data_row_index = 1
        in_header = True
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and all(c in "|-: " for c in stripped):
                in_header = False
                result.append(line)
                continue
            if stripped.startswith("|") and not in_header:
                result.append(f"[row {data_row_index}] {line}")
                data_row_index += 1
                continue
            result.append(line)

        return "\n".join(result)
