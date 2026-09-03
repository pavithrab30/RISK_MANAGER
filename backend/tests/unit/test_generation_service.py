import pytest

from app.data.models import BBox, BlockType, Chunk, Document, DocumentStatus, RetrievedChunk
from app.services.generation_service import GenerationService


class FakeLLMClient:
    name = "fake"

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.response


def _rc(text: str, page: int, score: float = 5.0) -> RetrievedChunk:
    chunk = Chunk.new(
        document_id="doc1", page_number=page, block_type=BlockType.TEXT, text=text, bbox=BBox(0, 0, 1, 1)
    )
    return RetrievedChunk(chunk=chunk, rerank_score=score)


def _doc_map():
    doc = Document.new("report.pdf", "/tmp/report.pdf")
    doc.status = DocumentStatus.READY
    return {"doc1": doc}


def test_refuses_immediately_with_no_retrieved_chunks_and_never_calls_llm():
    llm = FakeLLMClient("should not be used")
    service = GenerationService(llm)
    result = service.generate("What is the revenue?", [], {})
    assert result.refused is True
    assert result.refusal_reason == "no_candidates_retrieved"
    assert llm.calls == 0


def test_refuses_on_low_retrieval_confidence_without_calling_llm():
    llm = FakeLLMClient("should not be used")
    service = GenerationService(llm, rerank_score_floor=-1.0)
    retrieved = [_rc("Irrelevant passage.", page=1, score=-5.0)]
    result = service.generate("What is the revenue?", retrieved, _doc_map())
    assert result.refused is True
    assert result.refusal_reason == "low_retrieval_confidence"
    assert llm.calls == 0


def test_well_cited_answer_is_accepted():
    llm = FakeLLMClient("Revenue grew 12% in Q3 [S1].")
    service = GenerationService(llm, min_coverage=0.6)
    retrieved = [_rc("Revenue grew 12% in Q3 2025.", page=4, score=5.0)]
    result = service.generate("How did revenue change?", retrieved, _doc_map())

    assert result.refused is False
    assert result.groundedness_coverage == 1.0
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 4
    assert result.citations[0].document_name == "report.pdf"
    assert "[S1]" not in result.answer_text  # markers stripped from display text


def test_poorly_cited_answer_is_withheld():
    # Two sentences, only one carries a resolvable citation -> 50% coverage, below the 0.6 floor
    llm = FakeLLMClient("Revenue grew 12% in Q3 [S1]. It will keep growing indefinitely.")
    service = GenerationService(llm, min_coverage=0.6)
    retrieved = [_rc("Revenue grew 12% in Q3 2025.", page=4, score=5.0)]
    result = service.generate("How did revenue change?", retrieved, _doc_map())

    assert result.refused is True
    assert result.refusal_reason == "low_citation_coverage"
    assert "withheld" in result.answer_text.lower()


def test_model_self_reported_refusal_is_respected():
    llm = FakeLLMClient("The provided documents do not contain enough information to answer this.")
    service = GenerationService(llm)
    retrieved = [_rc("Unrelated passage about widgets.", page=1, score=5.0)]
    result = service.generate("What is the CEO's salary?", retrieved, _doc_map())

    assert result.refused is True
    assert result.refusal_reason == "model_reported_insufficient"
