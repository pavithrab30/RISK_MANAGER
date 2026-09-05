/**
 * Tests for AnswerMessage component — the primary output surface.
 * Covers: answer text rendering, groundedness bar, citation chips
 * (including row numbers), sub-query display, refused state, and
 * export button presence.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AnswerMessage } from '../components/AnswerMessage';
import type { QueryResponse, Citation } from '../api/types';

// Mock export functions so tests don't make real HTTP calls
vi.mock('../api/client', () => ({
  exportMarkdown: vi.fn().mockResolvedValue(undefined),
  exportPdf: vi.fn().mockResolvedValue(undefined),
  ApiRequestError: class ApiRequestError extends Error {
    status: number; payload: unknown;
    constructor(status: number, payload: unknown) {
      super('error'); this.status = status; this.payload = payload;
    }
  },
}));

const makeCitation = (overrides: Partial<Citation> = {}): Citation => ({
  chunk_id: 'chunk_1',
  document_id: 'doc_1',
  document_name: 'delivery-evidence.pdf',
  page_number: 3,
  bbox: { x0: 0.1, y0: 0.2, x1: 0.9, y1: 0.4 },
  snippet: 'Carrier tracking records delivery on 2026-08-13.',
  row_number: null,
  ...overrides,
});

const makeResponse = (overrides: Partial<QueryResponse> = {}): QueryResponse => ({
  trace_id: 'trace-abc',
  answer: 'The evidence records delivery on 2026-08-13.',
  citations: [],
  refused: false,
  refusal_reason: null,
  groundedness_coverage: 1.0,
  sub_queries: [],
  retrieval_debug: [],
  ...overrides,
});

describe('AnswerMessage', () => {
  it('renders the answer text', () => {
    render(
      <AnswerMessage
        question="Was the order delivered?"
        response={makeResponse({ answer: 'The evidence records delivery on 2026-08-13.' })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByText('The evidence records delivery on 2026-08-13.')).toBeInTheDocument();
  });

  it('shows groundedness percentage for non-refused answers', () => {
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ groundedness_coverage: 0.85 })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByText('85% grounded')).toBeInTheDocument();
  });

  it('hides groundedness bar when answer is refused', () => {
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ refused: true, groundedness_coverage: 0 })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.queryByText(/grounded/)).not.toBeInTheDocument();
  });

  it('renders citation chips with document name and page', () => {
    const citation = makeCitation({ document_name: 'delivery-evidence.pdf', page_number: 5, row_number: null });
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ citations: [citation] })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /delivery-evidence\.pdf.*p\.5/i })).toBeInTheDocument();
  });

  it('renders row number in citation chip when present', () => {
    const citation = makeCitation({ document_name: 'sheet.pdf', page_number: 1, row_number: 7 });
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ citations: [citation] })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /row 7/i })).toBeInTheDocument();
  });

  it('calls onOpenCitation with the correct citation when chip is clicked', async () => {
    const user = userEvent.setup();
    const citation = makeCitation({ chunk_id: 'chunk_42' });
    const onOpenCitation = vi.fn();
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ citations: [citation] })}
        onOpenCitation={onOpenCitation}
      />
    );
    await user.click(screen.getByRole('button', { name: /delivery-evidence\.pdf/i }));
    expect(onOpenCitation).toHaveBeenCalledWith(citation);
  });

  it('renders multiple citation chips for multiple citations', () => {
    const citations = [
      makeCitation({ chunk_id: 'c1', document_name: 'doc1.pdf', page_number: 1 }),
      makeCitation({ chunk_id: 'c2', document_name: 'doc2.pdf', page_number: 2 }),
    ];
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ citations })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /doc1\.pdf/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /doc2\.pdf/ })).toBeInTheDocument();
  });

  it('shows sub-query decomposition when multiple sub-queries exist', () => {
    render(
      <AnswerMessage
        question="compound question"
        response={makeResponse({ sub_queries: ['sub q 1', 'sub q 2'] })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByText(/sub q 1/)).toBeInTheDocument();
    expect(screen.getByText(/sub q 2/)).toBeInTheDocument();
  });

  it('does not show sub-query section for single sub-query', () => {
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse({ sub_queries: ['only query'] })}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.queryByText(/decomposed into/i)).not.toBeInTheDocument();
  });

  it('renders export buttons', () => {
    render(
      <AnswerMessage
        question="Q"
        response={makeResponse()}
        onOpenCitation={vi.fn()}
      />
    );
    expect(screen.getByRole('button', { name: /export markdown/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /export pdf/i })).toBeInTheDocument();
  });
});
