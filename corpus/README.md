# Sample corpus

Four documents, chosen to exercise every kind of "messy" content the assignment calls out:
multi-column layout, figures, tables, and a scanned/OCR-only page.

| File | Source | Why it's here |
|---|---|---|
| `attention_is_all_you_need.pdf` | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) (open access) | Classic 2-column paper: architecture figures, multiple results tables (BLEU scores), section cross-references ("see Table 2") - good for reference-graph expansion and cross-page questions. |
| `bert.pdf` | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) (open access) | Multi-column paper with GLUE benchmark result tables spread across several pages - good for cross-page table aggregation questions. |
| `nist_digital_identity_guidelines.pdf` | [NIST SP 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html) (US government work, public domain), trimmed to its first 30 pages for ingestion speed | Real-world technical report with genuine data tables (authenticator assurance levels) and a single-column layout - contrasts with the arXiv papers' 2-column layout. |
| `scanned_attention_excerpt.pdf` | Synthesized from the first 3 pages of `attention_is_all_you_need.pdf` | **Deliberately built as an image-only PDF** (pages rasterized at 200 DPI, re-embedded with zero text layer - verified 0 extractable characters) to force Docling's OCR path. We did not source a naturally-scanned example in the time available; this is documented explicitly rather than silently passed off as a real scan. It exercises the same OCR pipeline a genuine scan would. |

## What's deliberately left out of this corpus (and the system's OCR coverage generally)

- Handwriting.
- Non-Latin scripts.
- Heavily skewed/rotated scans beyond Docling's built-in deskue - our synthesized scan is
  perfectly axis-aligned, which is easier than a real phone-camera scan.
- Extremely dense multi-table financial filings (e.g. 10-Ks) - the NIST report has real tables
  but nothing approaching SEC-filing density.

See the main [README](../README.md) and [ADR](../ADR.md) for how these gaps map to product
decisions.
