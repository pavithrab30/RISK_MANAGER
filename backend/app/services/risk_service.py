"""Conservative, extractive chargeback evidence triage.

CSV guidance is reference material, never merchant evidence. A match means a
candidate passage for review, not an authenticity or network compliance finding.
"""

from __future__ import annotations

import csv
import re
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "chargeback-reason-codes.csv"
FEATURE_NAMES = [
    "coverage",
    "critical_coverage",
    "matched_chunks",
    "id_conflicts",
    "amount_conflicts",
    "date_conflicts",
    "status_conflicts",
    "identity_coverage",
]


class ChargebackCase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    network: str = Field(min_length=1, max_length=80)
    claim_type: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=5000)
    order_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w-]+$")
    transaction_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w-]+$")
    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    transaction_date: date | None = None
    expected_delivery_date: date | None = None
    claimed_delivery_status: Literal["unknown", "delivered", "not_delivered"] = "unknown"
    claimed_refund_status: Literal["unknown", "refunded", "not_refunded"] = "unknown"
    document_ids: list[str] = Field(min_length=1, max_length=100)


@lru_cache(maxsize=1)
def reasons() -> list[dict]:
    with REFERENCE.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def reason_for(case: ChargebackCase) -> dict:
    for row in reasons():
        if row["network"].casefold() == case.network.casefold() and case.claim_type.casefold() in {
            row["code"].casefold(),
            row["title"].casefold(),
        }:
            return row
    raise ValueError("Select a supported network and chargeback reason code.")


def requirements(row: dict) -> list[dict]:
    # Criticality is an explicit conservative application policy, not a CSV label.
    return [
        {"id": f"requirement_{i}", "description": value.strip(), "critical": i == 0, "matches": []}
        for i, value in enumerate(row["key_evidence"].split(";"))
        if value.strip()
    ]


def field_values(text: str, field: str) -> list[str]:
    patterns = {
        "order_id": r"\border\s*(?:id|number|#)\s*[:=#-]?\s*([\w-]+)",
        "transaction_id": r"\btransaction\s*(?:id|number|#)\s*[:=#-]?\s*([\w-]+)",
        "amount": r"\b(?:transaction\s+amount|charged\s+amount|amount|total)\s*(?:[:=]\s*)?(?:[A-Z]{3}\s*|[$£€₹]\s*)?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)(?![\d,])",
        "transaction_date": r"\btransaction\s+date\s*[:=]\s*(\d{4}-\d{2}-\d{2})",
        "delivery_date": r"\bdelivery\s+date\s*[:=]\s*(\d{4}-\d{2}-\d{2})",
        "refund_date": r"\brefund\s+date\s*[:=]\s*(\d{4}-\d{2}-\d{2})",
        "delivery_status": r"\bdelivery\s+status\s*(?:[:=]\s*)?(not delivered|delivered|returned to sender|returned|pending|failed)\b",
        "refund_status": r"\brefund\s+status\s*(?:[:=]\s*)?(not refunded|refunded|pending|failed)\b",
    }
    values = re.findall(patterns[field], text, re.IGNORECASE)
    if field == "delivery_status":
        if re.search(r"\b(?:shows|records?|reports?)\s+deliver(?:y|ed)\b|\bdelivered to\b", text, re.IGNORECASE):
            values.append("delivered")
        if re.search(r"\b(?:package|parcel|shipment)\s+(?:was\s+)?returned\b|\breturned to (?:the )?(?:sender|merchant)\b", text, re.IGNORECASE):
            values.append("returned")
    return list(dict.fromkeys(values))


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "at",
    "from",
    "for",
    "with",
    "showing",
    "proof",
    "your",
    "that",
    "was",
    "is",
    "as",
    "by",
    "any",
    "if",
    "same",
    "it",
    "customer",
    "cardholder",
    "record",
    "records",
}


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2} - STOPWORDS


def supports(requirement: str, text: str) -> bool:
    """Only affirmative local sentences count; keyword mentions alone are uncertain."""
    expected = tokens(requirement)
    for sentence in re.split(r"[\n.;]", text):
        if re.search(
            r"\b(no|not|missing|unavailable|pending|failed|unsigned|unverified|should|must|required|example|template)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        overlap = expected & tokens(sentence)
        if len(overlap) >= min(3, len(expected)) and len(overlap) / max(1, len(expected)) >= 0.55:
            return True
    return False


def analyze(case: ChargebackCase, chunks: list) -> dict:
    row = reason_for(case)
    needed = requirements(row)
    needed.append(
        {
            "id": "transaction_record",
            "description": "Transaction record with matching order ID, transaction ID and amount",
            "critical": True,
            "matches": [],
        }
    )
    evidence, contradictions = [], []
    conflicted_sources: set[str] = set()
    counts = {k: 0 for k in ("id", "amount", "date", "status")}
    identity = set()
    observed: dict[str, dict[str, list[str]]] = {}

    def conflict(kind: str, message: str, source: str):
        conflicted_sources.add(source)
        counts[kind] += 1
        contradictions.append({"kind": kind, "message": message, "chunk_id": source})

    for rc in chunks:
        c = rc.chunk
        if c.document_id not in case.document_ids:
            continue  # Defense in depth around the selected case-document scope.
        values = {
            key: field_values(c.text, key)
            for key in (
                "order_id",
                "transaction_id",
                "amount",
                "transaction_date",
                "delivery_date",
                "refund_date",
                "delivery_status",
                "refund_status",
            )
        }
        linked = any(
            str(getattr(case, key)).casefold() in [v.casefold() for v in values[key]]
            for key in ("order_id", "transaction_id")
        )
        if not linked:
            continue  # Unrelated or unlinked passages cannot support this case.
        before = len(contradictions)
        for key in ("order_id", "transaction_id"):
            for value in values[key]:
                if value.casefold() != getattr(case, key).casefold():
                    conflict("id", f"{key}: case {getattr(case, key)}; source {value}", c.id)
                else:
                    identity.add(key)
        for value in values["amount"]:
            if Decimal(value.replace(",", "")) != case.amount:
                conflict("amount", f"Amount: case {case.amount}; source {value}", c.id)
        for key in ("transaction_date", "delivery_date", "refund_date"):
            for value in values[key]:
                try:
                    parsed = date.fromisoformat(value)
                except ValueError:
                    conflict("date", f"Invalid {key}: {value}", c.id)
                    continue
                if (
                    key == "transaction_date"
                    and case.transaction_date
                    and parsed != case.transaction_date
                ):
                    conflict("date", f"Transaction date differs: {value}", c.id)
                if (
                    key == "delivery_date"
                    and case.expected_delivery_date
                    and parsed > case.expected_delivery_date
                ):
                    conflict("date", f"Delivery after expected date: {value}", c.id)
                if (
                    key != "transaction_date"
                    and case.transaction_date
                    and parsed < case.transaction_date
                ):
                    conflict("date", f"{key} precedes transaction: {value}", c.id)
        delivery_claim = case.claimed_delivery_status
        refund_claim = case.claimed_refund_status
        if delivery_claim == "unknown" and re.search(
            r"\b(never received|not received|not delivered|did not receive|not provided|non-receipt)\b",
            case.description + " " + row["title"],
            re.IGNORECASE,
        ):
            delivery_claim = "not_delivered"
        if refund_claim == "unknown" and re.search(
            r"\b(not refunded|refund not received|credit not processed|never refunded)\b",
            case.description + " " + row["title"],
            re.IGNORECASE,
        ):
            refund_claim = "not_refunded"
        for key, claim in (("delivery_status", delivery_claim), ("refund_status", refund_claim)):
            for value in values[key]:
                normalized_status = value.lower().replace(" ", "_")
                if key == "delivery_status" and normalized_status in {
                    "returned",
                    "returned_to_sender",
                    "failed",
                }:
                    normalized_status = "not_delivered"
                if claim != "unknown" and normalized_status != claim:
                    conflict(
                        "status",
                        f"Claim {key}={claim}; source={value}. Review, not proof of fraud.",
                        c.id,
                    )
        for key in (
            "amount",
            "transaction_date",
            "delivery_date",
            "refund_date",
            "delivery_status",
            "refund_status",
        ):
            bucket = observed.setdefault(key, {})
            for value in values[key]:
                normalized = (
                    str(Decimal(value.replace(",", "")).normalize())
                    if key == "amount"
                    else value.lower()
                )
                if bucket and normalized not in bucket:
                    conflicted_sources.update(
                        source for sources in bucket.values() for source in sources
                    )
                    kind = (
                        "amount"
                        if key == "amount"
                        else "date"
                        if key.endswith("date")
                        else "status"
                    )
                    conflict(
                        kind, f"Sources disagree on {key}: {', '.join(bucket)} / {value}", c.id
                    )
                bucket.setdefault(normalized, []).append(c.id)
        usable = len(contradictions) == before
        evidence.append(
            {
                "chunk_id": c.id,
                "document_id": c.document_id,
                "page_number": c.page_number,
                "bbox": c.bbox.to_dict(),
                "row_number": None,
                "snippet": c.text,
                "usable": usable,
                "extracted_fields": values,
            }
        )
        if usable:
            for req in needed:
                if req["id"] == "transaction_record":
                    if (
                        all(
                            values[key]
                            and all(
                                v.casefold() == getattr(case, key).casefold() for v in values[key]
                            )
                            for key in ("order_id", "transaction_id")
                        )
                        and values["amount"]
                    ):
                        req["matches"].append(c.id)
                elif supports(req["description"], c.text):
                    req["matches"].append(c.id)

    for item in evidence:
        if item["chunk_id"] in conflicted_sources:
            item["usable"] = False
    for req in needed:
        req["matches"] = [source for source in req["matches"] if source not in conflicted_sources]
    missing = [r for r in needed if not r["matches"]]
    critical = [r for r in missing if r["critical"]]
    coverage = (len(needed) - len(missing)) / max(1, len(needed))
    critical_coverage = 0 if critical else 1
    features = [
        coverage,
        critical_coverage,
        min(len(evidence), 20),
        counts["id"],
        counts["amount"],
        counts["date"],
        counts["status"],
        len(identity) / 2,
    ]
    base = round(70 * coverage + 20 * critical_coverage + 10 * len(identity) / 2)
    penalty = min(100, 25 * len(contradictions))
    score = max(0, base - penalty) if evidence else 0
    return {
        "reference": row,
        "requirements": needed,
        "evidence": evidence,
        "missing_evidence": missing,
        "critical_missing_evidence": critical,
        "contradictions": contradictions,
        "features": dict(zip(FEATURE_NAMES, features)),
        "evidence_score": score,
        "explanation": [
            f"Requirement coverage: {len(needed) - len(missing)}/{len(needed)} (70 points maximum).",
            f"Critical requirement: {20 * critical_coverage} points; identifier coverage: {10 * len(identity) / 2:g} points.",
            f"Contradiction penalty: {penalty} points. Candidate matches require merchant verification.",
        ],
    }


def finalize(analysis: dict, prediction: dict) -> dict:
    if analysis["contradictions"]:
        recommendation = "MANUAL_REVIEW"
    elif analysis["critical_missing_evidence"] or not analysis["evidence"]:
        recommendation = "GATHER_MORE_EVIDENCE"
    elif (
        analysis["evidence_score"] >= 85
        and not analysis["missing_evidence"]
        and analysis["features"]["identity_coverage"] == 1
        and prediction.get("label") == "SUFFICIENT_EVIDENCE"
    ):
        recommendation = "AUTO_RESPOND"
    else:
        recommendation = "MANUAL_REVIEW"
    # Extractive drafting deliberately avoids an LLM inventing connective facts.
    draft = [
        "DRAFT — MERCHANT REVIEW REQUIRED — NOT SUBMITTED",
        "Please review the following excerpts from the selected merchant documents.",
        "These excerpts are unverified source quotations, not independent factual findings.",
    ]
    for item in analysis["evidence"]:
        if item["usable"]:
            draft.append(
                f"[{item['chunk_id']}; document {item['document_id']}; page {item['page_number']}]\n> "
                + item["snippet"].replace("\n", "\n> ")
            )
    if len(draft) == 3:
        draft.append(
            "No usable linked evidence was retrieved. Gather documents before preparing a response."
        )
    draft.append(
        "Resolve all missing evidence and contradictions; verify authenticity and relevance before using any excerpt. Merchant approval is always required."
    )
    return {
        **analysis,
        "ml_prediction": prediction,
        "recommendation": recommendation,
        "risk_level": "HIGH"
        if analysis["contradictions"] or analysis["evidence_score"] < 50
        else "MEDIUM"
        if recommendation != "AUTO_RESPOND"
        else "LOW",
        "merchant_review_required": True,
        "draft_response": "\n\n".join(draft),
    }

