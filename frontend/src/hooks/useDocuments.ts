import { useCallback, useEffect, useRef, useState } from "react";
import { getDocument, listDocuments, uploadDocument } from "../api/client";
import type { DocumentSummary } from "../api/types";

const POLL_INTERVAL_MS = 2500;

/** Owns the document list + upload + status polling. Polling is a
 * deliberately simple stand-in for a push-based status channel
 * (WebSocket/SSE) - fine at this scale, called out as the first thing to
 * change in ADR.md "Cost & Scale". */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollTimers = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  const refresh = useCallback(async () => {
    const docs = await listDocuments();
    setDocuments(docs);
  }, []);

  useEffect(() => {
    refresh().catch((err) => setError(err instanceof Error ? err.message : String(err)));
    const timers = pollTimers.current;
    return () => {
      timers.forEach((timer) => clearInterval(timer));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollDocument = useCallback((id: string) => {
    if (pollTimers.current.has(id)) return;
    const timer = setInterval(async () => {
      try {
        const doc = await getDocument(id);
        setDocuments((prev) => prev.map((d) => (d.id === id ? doc : d)));
        if (doc.status === "ready" || doc.status === "failed") {
          clearInterval(timer);
          pollTimers.current.delete(id);
        }
      } catch {
        clearInterval(timer);
        pollTimers.current.delete(id);
      }
    }, POLL_INTERVAL_MS);
    pollTimers.current.set(id, timer);
  }, []);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        const doc = await uploadDocument(file);
        setDocuments((prev) => [doc, ...prev]);
        pollDocument(doc.id);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [pollDocument],
  );

  return { documents, upload, uploading, error, refresh };
}
