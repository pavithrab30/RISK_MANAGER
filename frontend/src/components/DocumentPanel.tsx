import { useRef } from "react";
import type { DocumentSummary } from "../api/types";

interface Props {
  documents: DocumentSummary[];
  uploading: boolean;
  onUpload: (file: File) => void;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onViewDocument: (id: string, page: number) => void;
}

const STATUS_LABEL: Record<DocumentSummary["status"], string> = {
  pending: "Queued",
  parsing: "Parsing…",
  ready: "Ready",
  failed: "Failed",
};

export function DocumentPanel({
  documents,
  uploading,
  onUpload,
  selectedIds,
  onToggleSelect,
  onViewDocument,
}: Props) {
  const fileInput = useRef<HTMLInputElement>(null);

  return (
    <div className="document-panel">
      <div className="document-panel__header">
        <h2>Documents</h2>
        <button
          className="button button--primary"
          disabled={uploading}
          onClick={() => fileInput.current?.click()}
        >
          {uploading ? "Uploading…" : "+ Upload PDF"}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
      </div>

      {documents.length === 0 && (
        <p className="document-panel__empty">No documents yet. Upload a PDF to get started.</p>
      )}

      <ul className="document-list">
        {documents.map((doc) => (
          <li key={doc.id} className={`document-item document-item--${doc.status}`}>
            <label className="document-item__select">
              <input
                type="checkbox"
                checked={selectedIds.has(doc.id)}
                disabled={doc.status !== "ready"}
                onChange={() => onToggleSelect(doc.id)}
              />
            </label>
            <div className="document-item__info" onClick={() => doc.status === "ready" && onViewDocument(doc.id, 1)}>
              <div className="document-item__name" title={doc.filename}>
                {doc.filename}
              </div>
              <div className="document-item__meta">
                <span className={`status-badge status-badge--${doc.status}`}>
                  {STATUS_LABEL[doc.status]}
                </span>
                {doc.status === "ready" && (
                  <span>
                    {doc.num_pages} page{doc.num_pages === 1 ? "" : "s"}
                    {doc.is_scanned ? " · OCR" : ""}
                  </span>
                )}
                {doc.status === "failed" && doc.error_message && (
                  <span className="document-item__error" title={doc.error_message}>
                    {doc.error_message.slice(0, 60)}
                  </span>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
      {documents.some((d) => d.status === "ready") && (
        <p className="document-panel__hint">
          {selectedIds.size === 0
            ? "Searching all documents. Check boxes to restrict a question to specific documents."
            : `Restricting search to ${selectedIds.size} selected document${selectedIds.size === 1 ? "" : "s"}.`}
        </p>
      )}
    </div>
  );
}
