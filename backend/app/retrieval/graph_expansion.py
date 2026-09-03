"""
Reference-graph expansion.

A lightweight, scoped alternative to a full document knowledge graph: during
chunking (services/chunking_service.py) we already extracted edges like
"this chunk's text mentions Figure 3" as ChunkRef rows. At query time, for
every chunk that made it into the candidate pool, we look up its outgoing
references and resolve the label ("Figure 3") to the chunk(s) that actually
*are* that figure/table/section - then add those into the pool, tagged
via_graph_expansion=True so they're visible in citations/debugging even if
they wouldn't have ranked on their own merits.

Why this over a general entity/knowledge graph: building and maintaining a
real KG (entity extraction, coreference, relation typing) is disproportionate
engineering for a one-week project and is hard to evaluate cleanly. Reference
edges are cheap to extract (regex), cheap to resolve (a LIKE query), and
target exactly the failure mode the brief calls out - "as shown in Table 3"
sitting on a different page than the sentence that discusses it.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.data.models import RetrievedChunk
from app.data.store.metadata_store import MetadataStore

logger = get_logger(__name__)

_MAX_EXPANSIONS = 6


class GraphExpander:
    def __init__(self, metadata_store: MetadataStore):
        self._store = metadata_store

    def expand(
        self, document_id: str, candidates: dict[str, RetrievedChunk]
    ) -> dict[str, RetrievedChunk]:
        chunk_ids = list(candidates.keys())
        refs = self._store.get_refs_for_chunks(chunk_ids)
        if not refs:
            return candidates

        added = 0
        for ref in refs:
            if added >= _MAX_EXPANSIONS:
                break
            target_chunks = self._store.find_chunks_by_label(document_id, ref.target_label)
            for target in target_chunks:
                if target.id in candidates or target.id == ref.source_chunk_id:
                    continue
                candidates[target.id] = RetrievedChunk(
                    chunk=target,
                    via_graph_expansion=True,
                )
                added += 1
                logger.debug(
                    "graph_expansion_added",
                    source_chunk_id=ref.source_chunk_id,
                    target_chunk_id=target.id,
                    label=ref.target_label,
                )
                if added >= _MAX_EXPANSIONS:
                    break
        return candidates
