from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_container
from app.api.routes_risk import router


def test_risk_route_reuses_pipeline_and_restricts_scope():
    pipeline = Mock()
    pipeline.retrieve.return_value = SimpleNamespace(chunks=[])
    app = FastAPI()
    app.include_router(router)
    metadata_store = Mock()
    metadata_store.get_chunks_for_document.return_value = []
    app.dependency_overrides[get_container] = lambda: SimpleNamespace(
        retrieval_pipeline=pipeline, metadata_store=metadata_store
    )
    client = TestClient(app)
    data = {
        "network": "Visa",
        "claim_type": "13.1",
        "description": "Not received",
        "order_id": "O-1",
        "transaction_id": "T-1",
        "amount": "50.00",
        "document_ids": ["selected"],
    }
    response = client.post("/api/risk/analyze", json=data)
    assert response.status_code == 200
    assert response.json()["recommendation"] == "GATHER_MORE_EVIDENCE"
    assert response.json()["merchant_review_required"]
    assert pipeline.retrieve.call_args.args[1] == ["selected"]
    metadata_store.get_chunks_for_document.assert_called_once_with("selected")
    assert "order ID O-1" in pipeline.retrieve.call_args.args[0]
    pipeline.reset_mock()
    assert (
        client.post("/api/risk/analyze", json=data | {"claim_type": "invalid"}).status_code == 422
    )
    assert client.post("/api/risk/analyze", json=data | {"document_ids": []}).status_code == 422
    pipeline.retrieve.assert_not_called()
    assert len(client.get("/api/risk/reason-codes").json()) == 64
