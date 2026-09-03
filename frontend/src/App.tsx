import { useState } from "react";
import { useDocuments } from "./hooks/useDocuments";
import { DocumentPanel } from "./components/DocumentPanel";
import { ChatPanel } from "./components/ChatPanel";
import { PdfViewer } from "./components/PdfViewer";
import type { Citation } from "./api/types";

interface ViewerState {
  documentId: string;
  page: number;
  bbox: Citation["bbox"] | null;
  totalPages: number;
}

export default function App() {
  const { documents, upload, uploading } = useDocuments();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [viewer, setViewer] = useState<ViewerState | null>(null);

  const readyDocuments = documents.filter((d) => d.status === "ready");

  function toggleSelect(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function openCitation(citation: Citation) {
    const doc = documents.find((d) => d.id === citation.document_id);
    setViewer({
      documentId: citation.document_id,
      page: citation.page_number,
      bbox: citation.bbox,
      totalPages: doc?.num_pages ?? 0,
    });
  }

  function viewDocument(documentId: string, page: number) {
    const doc = documents.find((d) => d.id === documentId);
    setViewer({ documentId, page, bbox: null, totalPages: doc?.num_pages ?? 0 });
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1>⬡ DocIntel</h1>
        <span className="app__tagline">Multimodal document intelligence with region-level citations</span>
      </header>
      <div className={`app__body ${viewer ? "app__body--with-viewer" : ""}`}>
        <DocumentPanel
          documents={documents}
          uploading={uploading}
          onUpload={upload}
          selectedIds={selectedIds}
          onToggleSelect={toggleSelect}
          onViewDocument={viewDocument}
        />
        <ChatPanel
          documentIds={[...selectedIds]}
          onOpenCitation={openCitation}
          disabled={readyDocuments.length === 0}
        />
        {viewer && (
          <div className="viewer-pane">
            <div className="viewer-pane__header">
              <span className="viewer-pane__filename">
                {documents.find((d) => d.id === viewer.documentId)?.filename}
              </span>
              <button className="button" onClick={() => setViewer(null)}>
                ✕
              </button>
            </div>
            <PdfViewer
              documentId={viewer.documentId}
              pageNumber={viewer.page}
              highlightBbox={viewer.bbox}
              totalPages={viewer.totalPages}
              onPageChange={(page) => setViewer((v) => (v ? { ...v, page, bbox: null } : v))}
            />
          </div>
        )}
      </div>
    </div>
  );
}
