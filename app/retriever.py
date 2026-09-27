"""Hybrid retrieval: dense (Qdrant) + keyword (BM25), merged with Reciprocal Rank Fusion."""
from .models import Hit


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Merge ranked lists. Each chunk gets sum(1 / (k + rank)) across lists.

    RRF only uses rank positions, so we don't need to normalise BM25 scores against cosine scores.
    """
    fused: dict[int, Hit] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit.chunk.id, Hit(hit.chunk, 0.0))
            entry.score += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)


class HybridRetriever:
    def __init__(self, embedder, store, bm25):
        self.embedder, self.store, self.bm25 = embedder, store, bm25

    def dense(self, query: str, k: int) -> list[Hit]:
        return self.store.search(self.embedder.embed_query(query), k)

    def keyword(self, query: str, k: int) -> list[Hit]:
        return self.bm25.search(query, k)
