/**
 * Tests for API client utility functions.
 * These test the URL construction and error handling logic without
 * making real network calls — the fetch itself is mocked.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { documentFileUrl, pageImageUrl, ApiRequestError } from '../api/client';

// --- URL helpers ---

describe('documentFileUrl', () => {
  it('constructs the correct file URL for a document id', () => {
    const url = documentFileUrl('doc_abc123');
    expect(url).toBe('http://localhost:8000/api/documents/doc_abc123/file');
  });
});

describe('pageImageUrl', () => {
  it('constructs the correct page image URL', () => {
    const url = pageImageUrl('doc_abc123', 3);
    expect(url).toBe('http://localhost:8000/api/documents/doc_abc123/pages/3/image');
  });
});

// --- ApiRequestError ---

describe('ApiRequestError', () => {
  it('carries status and payload', () => {
    const payload = { code: 'not_found', message: 'Doc not found', details: {}, trace_id: 'abc' };
    const err = new ApiRequestError(404, payload);
    expect(err.status).toBe(404);
    expect(err.payload).toBe(payload);
    expect(err.message).toBe('Doc not found');
  });

  it('is an instance of Error', () => {
    const payload = { code: 'err', message: 'oops', details: {}, trace_id: '-' };
    const err = new ApiRequestError(500, payload);
    expect(err).toBeInstanceOf(Error);
  });
});

// --- uploadDocument error handling ---

describe('uploadDocument', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('throws ApiRequestError on non-ok response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        code: 'unsupported_file_type',
        message: 'Only PDF is supported.',
        details: {},
        trace_id: 'xyz',
      }),
    } as Response);

    const { uploadDocument } = await import('../api/client');
    const file = new File(['dummy'], 'test.docx', { type: 'application/octet-stream' });
    await expect(uploadDocument(file)).rejects.toBeInstanceOf(ApiRequestError);
  });
});
