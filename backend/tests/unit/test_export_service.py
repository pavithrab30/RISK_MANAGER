from app.api.schemas import BBoxSchema, CitationSchema, ExportRequest
from app.services.export_service import generate_markdown, generate_pdf


def _export(**overrides) -> ExportRequest:
    defaults = dict(
        question="Was order DEMO-ORDER-1001 delivered?",
        answer="Carrier records show delivery on 2026-08-13 [S1].",
        citations=[
            CitationSchema(
                chunk_id="c1",
                document_id="doc123",
                document_name="A3-shipping-and-delivery.pdf",
                page_number=2,
                bbox=BBoxSchema(x0=0.1, y0=0.1, x1=0.9, y1=0.3),
                snippet="Shipment SYNTHETIC-TRACK-4411 was delivered on 2026-08-13.",
            )
        ],
        groundedness_coverage=1.0,
        refused=False,
        sub_queries=[],
    )
    defaults.update(overrides)
    return ExportRequest(**defaults)


def test_markdown_includes_question_answer_and_citation_link():
    md = generate_markdown(_export(), base_url="http://localhost:8000/")
    assert "Was order DEMO-ORDER-1001 delivered?" in md
    assert "delivery on 2026-08-13" in md
    assert "A3-shipping-and-delivery.pdf, page 2" in md
    assert "http://localhost:8000/api/documents/doc123/file#page=2" in md
    assert "SYNTHETIC-TRACK-4411" in md


def test_markdown_shows_groundedness_percentage():
    md = generate_markdown(_export(groundedness_coverage=0.75), base_url="http://x/")
    assert "75%" in md


def test_markdown_flags_refused_answers():
    md = generate_markdown(_export(refused=True), base_url="http://x/")
    assert "withheld or refused" in md.lower()


def test_markdown_lists_decomposed_subqueries_when_present():
    md = generate_markdown(
        _export(sub_queries=["sub question one", "sub question two"]), base_url="http://x/"
    )
    assert "sub question one" in md
    assert "sub question two" in md


def test_markdown_handles_no_citations():
    md = generate_markdown(_export(citations=[]), base_url="http://x/")
    assert "did not support a grounded answer" in md


def test_pdf_generates_valid_pdf_bytes():
    pdf_bytes = generate_pdf(_export(), base_url="http://localhost:8000/")
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 500  # not an empty/broken document


def test_pdf_generation_does_not_crash_with_special_characters():
    export = _export(
        question="What about <tags> & \"quotes\"?",
        answer="An answer with <html> & special chars, and a newline\nhere.",
    )
    pdf_bytes = generate_pdf(export, base_url="http://x/")
    assert pdf_bytes[:4] == b"%PDF"


def test_pdf_generation_handles_empty_citations():
    pdf_bytes = generate_pdf(_export(citations=[]), base_url="http://x/")
    assert pdf_bytes[:4] == b"%PDF"
