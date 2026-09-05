import { useEffect, useState } from "react";
import { analyzeChargeback, listReasonCodes } from "../api/client";
import type { Citation, RiskResult, ReasonCode } from "../api/types";

export function RiskPanel({ documentIds, onOpenCitation }: {
  documentIds: string[]; onOpenCitation: (citation: Citation) => void;
}) {
  const [reasons, setReasons] = useState<ReasonCode[]>([]);
  const [reasonIndex, setReasonIndex] = useState(0);
  const [result, setResult] = useState<RiskResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { listReasonCodes().then(setReasons).catch(() => setError("Cannot load reason codes. Check the backend connection.")); }, []);
  useEffect(() => { setResult(null); }, [documentIds.join(",")]);
  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const fields = Object.fromEntries(new FormData(event.currentTarget));
    const reason = reasons[reasonIndex];
    setBusy(true); setError(""); setResult(null);
    try {
      setResult(await analyzeChargeback({ ...fields, network: reason.network, claim_type: reason.code,
        document_ids: documentIds, transaction_date: fields.transaction_date || null,
        expected_delivery_date: fields.expected_delivery_date || null }));
    } catch (e) { setError(e instanceof Error ? e.message : "Analysis failed. Please retry."); }
    finally { setBusy(false); }
  }
  return <main className="risk-panel">
    <div className="risk-heading"><span className="risk-eyebrow">CHARGEBACK EVIDENCE VERIFIER</span><h2>Review evidence. Prepare a grounded response.</h2>
      <p>Upload merchant PDFs, select the documents for this case, then assess evidence readiness.</p></div>
    <p className="risk-notice">Defense only · Drafts always require merchant review · Nothing is submitted automatically.</p>
    <form className="risk-form" onSubmit={submit} onChange={() => setResult(null)}>
      <fieldset disabled={busy}>
        <label className="risk-wide">Claim / reason type<select value={reasonIndex} onChange={e => setReasonIndex(Number(e.target.value))}>
          {reasons.map((r, i) => <option key={`${r.network}-${r.code}`} value={i}>{r.network} {r.code} — {r.title}</option>)}
        </select></label>
        <label className="risk-wide">Claim description<textarea name="description" required maxLength={5000} placeholder="Describe the cardholder’s claim without adding unsupported facts." /></label>
        <label>Order ID<input name="order_id" required pattern="[A-Za-z0-9_-]+" maxLength={100} /></label>
        <label>Transaction ID<input name="transaction_id" required pattern="[A-Za-z0-9_-]+" maxLength={100} /></label>
        <label>Disputed amount<input name="amount" type="number" min="0.01" step="0.01" required /></label>
        <label>Transaction date (optional)<input name="transaction_date" type="date" /></label>
        <label>Expected delivery (optional)<input name="expected_delivery_date" type="date" /></label>
        <label>Claimed delivery status<select name="claimed_delivery_status"><option value="unknown">Unspecified</option><option value="not_delivered">Not delivered</option><option value="delivered">Delivered</option></select></label>
        <label>Claimed refund status<select name="claimed_refund_status"><option value="unknown">Unspecified</option><option value="not_refunded">Not refunded</option><option value="refunded">Refunded</option></select></label>
      </fieldset>
      <p>{documentIds.length} selected document(s). Select only documents belonging to this case.</p>
      <button className="button button--primary" disabled={busy || !reasons.length || !documentIds.length}>{busy ? "Retrieving and analyzing…" : "Analyze chargeback evidence"}</button>
    </form>
    {error && <p role="alert" className="risk-error">{error}</p>}
    {result && <section aria-label="Risk assessment" className="risk-results">
      <div className="risk-metrics"><div><strong>{result.evidence_score}/100</strong><span>Evidence score</span></div><div><strong>{result.risk_level}</strong><span>Evidence risk</span></div><div><strong>{result.recommendation.replaceAll("_", " ")}</strong><span>{result.recommendation === "AUTO_RESPOND" ? "Draft ready for merchant review" : "Suggested next step"}</span></div></div>
      <h3>Why this result</h3>{result.explanation.map(x => <p key={x}>{x}</p>)}
      <h3>ML evidence classification</h3><p><strong>{result.ml_prediction.label}</strong> · Model: {result.ml_prediction.model} · P(sufficient): {(result.ml_prediction.probability_sufficient * 100).toFixed(1)}% · Threshold: {result.ml_prediction.threshold}</p>
      <h3>Evidence requirements & missing documents</h3><ul>{result.requirements.map(r => <li key={r.id}><strong>{r.matches.length ? "Candidate found" : r.critical ? "CRITICAL — MISSING" : "Missing"}</strong>: {r.description}</li>)}</ul>
      <h3>Contradictions requiring review</h3>{!result.contradictions.length ? <p>No contradictions detected by the supported checks. This does not establish authenticity.</p> : <ul>{result.contradictions.map((c, i) => <li key={i} className="risk-error">{c.message} [{c.chunk_id}]</li>)}</ul>}
      <h3>Retrieved merchant evidence</h3>{!result.evidence.length && <p>No linked evidence retrieved. Upload documents with explicit order or transaction identifiers.</p>}
      {result.evidence.map(e => <article className="risk-evidence" key={e.chunk_id}><button className="button" onClick={() => onOpenCitation({ ...e, document_name: e.document_id })}>Open source · page {e.page_number}</button><span>{e.usable ? "Candidate evidence — verify" : "Conflict — excluded from draft"}</span><blockquote>{e.snippet}</blockquote></article>)}
      <h3>Draft response — merchant review required</h3><pre className="risk-draft">{result.draft_response}</pre>
      <details><summary>Reference guidance</summary><p>{result.reference.network} {result.reference.code}: {result.reference.title}</p><p>{result.reference.source}</p><p>CSV guidance is not current network-rule verification. Confirm requirements and deadlines with your processor.</p></details>
    </section>}
  </main>;
}
