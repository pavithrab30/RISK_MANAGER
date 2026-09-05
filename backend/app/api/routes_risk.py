from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import Container, get_container
from app.data.models import RetrievedChunk
from app.services.risk_model import RiskModel
from app.services.risk_service import ChargebackCase, analyze, finalize, reason_for, reasons

router = APIRouter(prefix="/api/risk", tags=["chargeback risk"])


@router.get("/reason-codes")
def list_reason_codes():
    return [
        {
            k: row[k]
            for k in (
                "network",
                "code",
                "title",
                "plain_english_meaning",
                "response_deadline",
                "winnability_label",
                "source",
            )
        }
        for row in reasons()
    ]


@router.post("/analyze")
def analyze_case(case: ChargebackCase, container: Annotated[Container, Depends(get_container)]):
    try:
        row = reason_for(case)
    except ValueError as exc:
        raise HTTPException(422, "Unsupported network/reason code") from exc
    query = " ".join(
        [
            row["title"],
            row["plain_english_meaning"],
            row["key_evidence"],
            f"order ID {case.order_id}",
            f"transaction ID {case.transaction_id}",
            f"amount {case.amount}",
        ]
    )
    retrieved = container.retrieval_pipeline.retrieve(query, case.document_ids)
    # Retrieval ranks the strongest passages first. Add remaining chunks from the
    # explicitly selected case documents so a required record on another page is
    # not incorrectly reported missing merely because it fell below top-k.
    case_chunks = list(retrieved.chunks)
    seen = {item.chunk.id for item in case_chunks}
    for document_id in case.document_ids:
        for chunk in container.metadata_store.get_chunks_for_document(document_id):
            if chunk.id not in seen:
                case_chunks.append(RetrievedChunk(chunk=chunk))
                seen.add(chunk.id)
    assessment = analyze(case, case_chunks)
    prediction = RiskModel().predict(assessment["features"])
    return finalize(assessment, prediction)
