import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { RiskPanel } from "../components/RiskPanel";
import { analyzeChargeback, listReasonCodes } from "../api/client";
import type { RiskResult } from "../api/types";
vi.mock("../api/client", () => ({ listReasonCodes: vi.fn(), analyzeChargeback: vi.fn() }));
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(listReasonCodes).mockResolvedValue([{network: "Visa", code: "13.1", title: "Not received"}]);
});
it("renders assessment, mandatory-review draft and source navigation", async () => {
  const result: RiskResult = {
    evidence_score: 40, risk_level: "HIGH", recommendation: "GATHER_MORE_EVIDENCE",
    explanation: ["Missing delivery evidence."], requirements: [{ id: "delivery", description: "Signed delivery proof", critical: true, matches: [] }],
    missing_evidence: [], critical_missing_evidence: [], contradictions: [],
    evidence: [{ chunk_id: "c1", document_id: "merchant", page_number: 2, row_number: null, bbox: {x0:0,y0:0,x1:1,y1:1}, snippet: "Original source excerpt", usable: true }],
    ml_prediction: { label: "INSUFFICIENT_EVIDENCE", model: "random_forest", probability_sufficient: .2, threshold: .5, disclaimer: "Synthetic demonstration only" },
    merchant_review_required: true, draft_response: "DRAFT — MERCHANT REVIEW REQUIRED", reference: {network: "Visa", code: "13.1", title: "Not received", source: "Supplied CSV"},
  };
  vi.mocked(analyzeChargeback).mockResolvedValue(result);
  const onOpen = vi.fn();
  render(<RiskPanel documentIds={["merchant"]} onOpenCitation={onOpen} />);
  await screen.findByText("Visa 13.1 — Not received");
  fireEvent.change(screen.getByLabelText("Claim description"), {target: {value: "Claim"}});
  fireEvent.change(screen.getByLabelText("Order ID"), {target: {value: "O-1"}});
  fireEvent.change(screen.getByLabelText("Transaction ID"), {target: {value: "T-1"}});
  fireEvent.change(screen.getByLabelText("Disputed amount"), {target: {value: "50"}});
  fireEvent.click(screen.getByRole("button", {name: "Analyze chargeback evidence"}));
  expect(await screen.findByText("40/100")).toBeInTheDocument();
  expect(screen.getByText("CRITICAL — MISSING")).toBeInTheDocument();
  expect(screen.getByText("DRAFT — MERCHANT REVIEW REQUIRED")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", {name: "Open source · page 2"}));
  expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({document_id: "merchant", page_number: 2}));
});
it("requires selected documents and shows mandatory review", async () => {
  render(<RiskPanel documentIds={[]} onOpenCitation={vi.fn()} />);
  await screen.findByText("Visa 13.1 — Not received");
  expect(screen.getByRole("button", {name: "Analyze chargeback evidence"})).toBeDisabled();
  expect(screen.getByText(/Drafts always require merchant review/)).toBeInTheDocument();
});
it("sends case fields and selected scope, shows useful API errors", async () => {
  vi.mocked(analyzeChargeback).mockRejectedValue(new Error("Unsupported network/reason code"));
  render(<RiskPanel documentIds={["merchant"]} onOpenCitation={vi.fn()} />);
  await screen.findByText("Visa 13.1 — Not received");
  fireEvent.change(screen.getByLabelText("Claim description"), {target: {value: "Not received"}});
  fireEvent.change(screen.getByLabelText("Order ID"), {target: {value: "O-1"}});
  fireEvent.change(screen.getByLabelText("Transaction ID"), {target: {value: "T-1"}});
  fireEvent.change(screen.getByLabelText("Disputed amount"), {target: {value: "50"}});
  fireEvent.click(screen.getByRole("button", {name: "Analyze chargeback evidence"}));
  await waitFor(() => expect(analyzeChargeback).toHaveBeenCalledWith(expect.objectContaining({document_ids: ["merchant"], amount: "50", claim_type: "13.1"})));
  expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported network/reason code");
});
