import { useEffect, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import type { BBox } from "../api/types";
import { documentFileUrl } from "../api/client";

// Use the CDN worker that matches the installed pdfjs-dist version.
// This works in both dev (Vite) and Docker (nginx) without any build config.
const PDFJS_VERSION = pdfjsLib.version;
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${PDFJS_VERSION}/pdf.worker.min.mjs`;

interface Props {
  documentId: string;
  pageNumber: number;
  highlightBbox?: BBox | null;
  totalPages?: number;
  onPageChange?: (page: number) => void;
}

export function PdfViewer({ documentId, pageNumber, highlightBbox, totalPages, onPageChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Blob URL avoids CORS issues with the PDF.js web worker — the worker
  // fetches from a blob:// URL which is same-origin by definition.
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  const highlightKey = highlightBbox
    ? `${highlightBbox.x0}-${highlightBbox.y0}-${highlightBbox.x1}-${highlightBbox.y1}`
    : null;

  // Step 1: fetch the PDF bytes via the main thread (which has CORS headers)
  // and create a blob URL. Re-runs only when documentId changes.
  useEffect(() => {
    let cancelled = false;
    setBlobUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setError(null);

    fetch(documentFileUrl(documentId))
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.blob();
      })
      .then((blob) => {
        if (!cancelled) setBlobUrl(URL.createObjectURL(blob));
      })
      .catch((err) => {
        if (!cancelled) setError(err.message ?? "Failed to fetch PDF");
      });

    return () => {
      cancelled = true;
    };
  }, [documentId]);

  // Step 2: render the page whenever blobUrl or pageNumber changes
  useEffect(() => {
    if (!blobUrl) return;
    let cancelled = false;
    setLoading(true);

    async function render() {
      try {
        const loadingTask = pdfjsLib.getDocument({ url: blobUrl! });
        const pdf = await loadingTask.promise;
        if (cancelled) return;
        const page = await pdf.getPage(pageNumber);

        const containerWidth = wrapRef.current
          ? wrapRef.current.clientWidth - 16
          : 600;
        const baseViewport = page.getViewport({ scale: 1 });
        const scale = Math.min(containerWidth / baseViewport.width, 2.0);
        const viewport = page.getViewport({ scale });

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: ctx, viewport, canvas }).promise;
        if (cancelled) return;
        setCanvasSize({ width: viewport.width, height: viewport.height });
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to render page");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    render();
    return () => { cancelled = true; };
  }, [blobUrl, pageNumber]);

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [blobUrl]);

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer__toolbar">
        <button
          disabled={pageNumber <= 1}
          onClick={() => onPageChange?.(pageNumber - 1)}
          aria-label="Previous page"
        >
          ‹
        </button>
        <span>
          Page {pageNumber}
          {totalPages ? ` / ${totalPages}` : ""}
        </span>
        <button
          disabled={!!totalPages && pageNumber >= totalPages}
          onClick={() => onPageChange?.(pageNumber + 1)}
          aria-label="Next page"
        >
          ›
        </button>
      </div>
      <div className="pdf-viewer__canvas-wrap" ref={wrapRef}>
        {(loading || !blobUrl) && !error && (
          <div className="pdf-viewer__status">Loading…</div>
        )}
        {error && (
          <div className="pdf-viewer__status pdf-viewer__status--error">{error}</div>
        )}
        <div
          className="pdf-viewer__canvas-container"
          style={{
            width: canvasSize.width ? canvasSize.width : undefined,
            height: canvasSize.height ? canvasSize.height : undefined,
          }}
        >
          <canvas ref={canvasRef} />
          {highlightBbox && canvasSize.width > 0 && (
            <div
              key={highlightKey}
              className="pdf-viewer__highlight"
              style={{
                left: `${highlightBbox.x0 * 100}%`,
                top: `${highlightBbox.y0 * 100}%`,
                width: `${(highlightBbox.x1 - highlightBbox.x0) * 100}%`,
                height: `${(highlightBbox.y1 - highlightBbox.y0) * 100}%`,
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
