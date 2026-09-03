import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock pdfjs-dist worker — not needed in jsdom test environment
// PdfViewer is not tested directly; this prevents the "Failed to fetch worker" warning
vi.mock('pdfjs-dist/build/pdf.worker.mjs?url', () => ({
  default: '',
}));
