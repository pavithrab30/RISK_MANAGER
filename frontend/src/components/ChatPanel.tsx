import { useState } from "react";
import { askQuestion, ApiRequestError } from "../api/client";
import type { Citation, QueryResponse } from "../api/types";
import { AnswerMessage } from "./AnswerMessage";

interface ChatMessage {
  id: string;
  question: string;
  response?: QueryResponse;
  error?: string;
  loading: boolean;
}

interface Props {
  documentIds: string[];
  onOpenCitation: (citation: Citation) => void;
  disabled: boolean;
}

export function ChatPanel({ documentIds, onOpenCitation, disabled }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question) return;
    setInput("");

    const id = crypto.randomUUID();
    setMessages((prev) => [...prev, { id, question, loading: true }]);

    try {
      const response = await askQuestion(question, documentIds.length ? documentIds : undefined);
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, response, loading: false } : m)));
    } catch (err) {
      const message =
        err instanceof ApiRequestError
          ? `${err.payload.message} (trace: ${err.payload.trace_id})`
          : err instanceof Error
            ? err.message
            : "Something went wrong.";
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, error: message, loading: false } : m)));
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-panel__messages">
        {messages.length === 0 && (
          <div className="chat-panel__empty">
            <p>Ask a question about your uploaded documents.</p>
            <p className="chat-panel__empty-hint">
              Try a question that needs facts from more than one page or document - that's what
              this system is built for.
            </p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className="chat-turn">
            <div className="chat-turn__question">{m.question}</div>
            {m.loading && <div className="chat-turn__loading">Retrieving and generating…</div>}
            {m.error && <div className="chat-turn__error">{m.error}</div>}
            {m.response && (
              <AnswerMessage question={m.question} response={m.response} onOpenCitation={onOpenCitation} />
            )}
          </div>
        ))}
      </div>
      <form className="chat-panel__input" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={disabled ? "Upload a document first…" : "Ask a question…"}
          value={input}
          disabled={disabled}
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit" className="button button--primary" disabled={disabled || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
