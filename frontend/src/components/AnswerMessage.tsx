import { useState } from "react";
import { exportMarkdown, exportPdf, ApiRequestError } from "../api/client";
import type { Citation, QueryResponse } from "../api/types";

interface Props {
  question: string;
  response: QueryResponse;
  onOpenCitation: (citation: Citation) => void;
}

export function AnswerMessage({ question, response, onOpenCitation }: Props) {
  const [showDebug, setShowDebug] = useState(false);
  const [exporting, setExporting] = useState<"markdown" | "pdf" | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const coveragePct = Math.round(response.groundedness_coverage * 100);

  async function handleExport(format: "markdown" | "pdf") {
    setExporting(format);
    setExportError(null);
    try {
      const payload = {
        question,
        answer: response.answer,
        citations: response.citations,
        groundedness_coverage: response.groundedness_coverage,
        refused: response.refused,
        sub_queries: response.sub_queries,
      };
      await (format === "markdown" ? exportMarkdown(payload) : exportPdf(payload));
    } catch (err) {
      setExportError(
        err instanceof ApiRequestError ? err.payload.message : "Export failed. Please try again.",
      );
    } finally {
      setExporting(null);
    }
  }

  return (
    <div className={`answer ${response.refused ? "answer--refused" : ""}`}>
      <p className="answer__text">{response.answer}</p>

      {!response.refused && (
        <div className="answer__groundedness" title="Fraction of answer sentences with a resolvable citation">
          <span className="answer__groundedness-bar">
            <span
              className="answer__groundedness-fill"
              style={{ width: `${coveragePct}%` }}
            />
          </span>
          <span>{coveragePct}% grounded</span>
        </div>
      )}

      {response.citations.length > 0 && (
        <div className="answer__citations">
          <span className="answer__citations-label">Sources:</span>
          {response.citations.map((c, i) => (
            <button
              key={`${c.chunk_id}-${c.row_number ?? i}`}
              className="citation-chip"
              onClick={() => onOpenCitation(c)}
              title={c.snippet}
            >
              [{i + 1}] {c.document_name} · p.{c.page_number}{c.row_number != null ? ` · row ${c.row_number}` : ""}
            </button>
          ))}
        </div>
      )}

      {response.sub_queries.length > 1 && (
        <p className="answer__subqueries">
          Question was decomposed into: {response.sub_queries.map((q) => `"${q}"`).join("  ·  ")}
        </p>
      )}

      {response.retrieval_debug.length > 0 && (
        <div className="answer__debug">
          <button className="answer__debug-toggle" onClick={() => setShowDebug((v) => !v)}>
            {showDebug ? "Hide" : "Show"} retrieval details ({response.retrieval_debug.length})
          </button>
          {showDebug && (
            <table className="debug-table">
              <thead>
                <tr>
                  <th>Page</th>
                  <th>Dense</th>
                  <th>Keyword</th>
                  <th>Fused</th>
                  <th>Rerank</th>
                  <th>Via</th>
                </tr>
              </thead>
              <tbody>
                {response.retrieval_debug.map((d) => (
                  <tr key={d.chunk_id}>
                    <td>{d.page_number}</td>
                    <td>{d.dense_score.toFixed(3)}</td>
                    <td>{d.keyword_score.toFixed(3)}</td>
                    <td>{d.fused_score.toFixed(3)}</td>
                    <td>{d.rerank_score?.toFixed(3) ?? "—"}</td>
                    <td>{d.via_graph_expansion ? "graph" : "search"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="answer__export">
        <button
          className="button answer__export-btn"
          disabled={exporting !== null}
          onClick={() => handleExport("markdown")}
        >
          {exporting === "markdown" ? "Exporting…" : "⤓ Export Markdown"}
        </button>
        <button
          className="button answer__export-btn"
          disabled={exporting !== null}
          onClick={() => handleExport("pdf")}
        >
          {exporting === "pdf" ? "Exporting…" : "⤓ Export PDF"}
        </button>
        {exportError && <span className="answer__export-error">{exportError}</span>}
      </div>
    </div>
  );
}
