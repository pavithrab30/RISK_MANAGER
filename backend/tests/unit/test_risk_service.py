from datetime import date

import pytest
from pydantic import ValidationError

from app.data.models import BBox, BlockType, Chunk, RetrievedChunk
from app.services.risk_service import ChargebackCase, analyze, field_values, finalize, reasons, supports


def case(**changes):
    return ChargebackCase(
        **(
            {
                "network": "Visa",
                "claim_type": "13.1",
                "description": "Disputed delivery",
                "order_id": "ORDER-1",
                "transaction_id": "TX-1",
                "amount": "50.00",
                "document_ids": ["merchant"],
            }
            | changes
        )
    )


def chunk(text, doc="merchant"):
    return RetrievedChunk(Chunk.new(doc, 2, BlockType.TEXT, text, BBox(0, 0, 1, 1)))


def prediction(label="SUFFICIENT_EVIDENCE"):
    return {"label": label}


def test_reference_schema_not_labels():
    assert len(reasons()) == 64
    assert "key_evidence" in reasons()[0]
    assert "winnability_label" in reasons()[0]
    assert "label" not in reasons()[0]


def test_no_evidence_never_auto_or_invents_facts():
    result = finalize(analyze(case(), []), prediction())
    assert result["evidence_score"] == 0
    assert result["recommendation"] == "GATHER_MORE_EVIDENCE"
    assert "No usable linked evidence" in result["draft_response"]
    assert result["merchant_review_required"]


def test_scope_and_exact_identifier_linkage():
    records = [
        chunk("Order ID: ORDER-10. Signed proof of delivery."),
        chunk("Order ID: ORDER-1. Signed proof of delivery.", "other-merchant"),
        chunk("Signed proof of delivery. Transaction ID: OTHER."),
    ]
    assert not analyze(case(), records)["evidence"]


def test_conflicting_amount_ids_and_dates():
    result = finalize(
        analyze(
            case(transaction_date=date(2026, 1, 10)),
            [
                chunk(
                    "Order ID: ORDER-1; Transaction ID: TX-OTHER; Amount: 65.00; Transaction date: 2026-01-11; Delivery date: 2026-01-01"
                )
            ],
        ),
        prediction(),
    )
    assert {c["kind"] for c in result["contradictions"]} == {"id", "amount", "date"}
    assert result["recommendation"] == "MANUAL_REVIEW"
    assert "TX-OTHER" not in result["draft_response"]


def test_status_and_free_text_claim_conflicts():
    result = analyze(
        case(description="I never received the goods and was not refunded"),
        [chunk("Order ID: ORDER-1; Delivery status: delivered; Refund status: refunded")],
    )
    assert len(result["contradictions"]) == 2


def test_pdf_table_fields_and_comma_amount_are_extracted_without_false_conflict():
    text = (
        "Order ID\nRRC-66309\nTransaction ID\nPAY-903377\n"
        "Transaction amount\nINR 1,899.00.\nDelivery status\nnot delivered"
    )
    assert field_values(text, "amount") == ["1,899.00"]
    assert field_values(text, "delivery_status") == ["not delivered"]
    result = analyze(
        case(
            order_id="RRC-66309",
            transaction_id="PAY-903377",
            amount="1899.00",
            claimed_delivery_status="not_delivered",
        ),
        [chunk(text)],
    )
    assert not [item for item in result["contradictions"] if item["kind"] == "amount"]


def test_returned_to_sender_is_normalized_as_not_delivered():
    result = analyze(
        case(claimed_delivery_status="delivered"),
        [chunk("Order ID: ORDER-1; Delivery status returned to sender")],
    )
    assert any(item["kind"] == "status" for item in result["contradictions"])


def test_source_disagreement():
    result = analyze(
        case(claim_type="10.4"),
        [
            chunk("Order ID: ORDER-1; Delivery status: delivered"),
            chunk("Order ID: ORDER-1; Delivery status: not delivered"),
        ],
    )
    assert any("Sources disagree" in c["message"] for c in result["contradictions"])
    assert all(not item["usable"] for item in result["evidence"])
    assert all(not requirement["matches"] for requirement in result["requirements"])


def test_negated_and_hypothetical_requirements_do_not_count():
    assert not supports("Signed proof of delivery", "Signed proof of delivery is missing")
    assert not supports("Signed proof of delivery", "Signed proof of delivery is required")
    assert supports("Signed proof of delivery", "Signed proof of delivery attached")


def test_missing_amount_is_critical():
    result = analyze(case(), [chunk("Order ID: ORDER-1; Transaction ID: TX-1")])
    assert "transaction_record" in [r["id"] for r in result["critical_missing_evidence"]]


def test_full_candidate_evidence_score_and_ml_gate():
    text = "Order ID: ORDER-1; Transaction ID: TX-1; Amount: 50.00\n" + reasons()[4][
        "key_evidence"
    ].replace("For digital goods:", "Digital goods:")
    analysis = analyze(case(), [chunk(text)])
    assert not analysis["missing_evidence"]
    assert analysis["evidence_score"] == 100
    result = finalize(analysis, prediction())
    assert result["recommendation"] == "AUTO_RESPOND"
    assert result["merchant_review_required"] is True
    assert text.replace("\n", "\n> ") in result["draft_response"]
    assert (
        finalize(analysis, prediction("INSUFFICIENT_EVIDENCE"))["recommendation"] == "MANUAL_REVIEW"
    )


def test_invalid_case_and_unknown_reason():
    for changes in ({"amount": "NaN"}, {"amount": -1}, {"document_ids": []}, {"order_id": ""}):
        with pytest.raises(ValidationError):
            case(**changes)
    with pytest.raises(ValueError):
        analyze(case(claim_type="unknown"), [])
