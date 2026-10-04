"""Hybrid search index: dense vectors (meaning) + BM25 (exact words).

Dense search has two interchangeable backends:
  - NumpyDense  : in-memory cosine similarity. Default. For a few thousand chunks it is faster than a
                  network call to a vector DB, needs no server, and builds in milliseconds - ideal for
                  scale-to-zero hosting and for each visitor's private sandbox.
  - QdrantDense : a Qdrant server (QDRANT_URL). Use it when the corpus outgrows one machine's memory or
                  must be shared by many replicas. docker-compose runs the app this way.
"""
import numpy as np

from .bm25 import BM25Index
from .models import Chunk, Hit


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


class NumpyDense:
    name = "numpy"

    def __init__(self, vectors: np.ndarray):
        self.vectors = _normalise(vectors) if len(vectors) else np.zeros((0, 1), dtype=np.float32)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.vectors):
            return []
        sims = self.vectors @ _normalise(query_vector)
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k] if len(sims) > k else np.arange(len(sims))
        return sorted(((int(i), float(sims[i])) for i in top), key=lambda x: -x[1])


class QdrantDense:
    name = "qdrant"

    def __init__(self, url: str, collection: str, vectors: np.ndarray):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client, self.collection = QdrantClient(url=url, timeout=10), collection
        # the sample index is small, so (re)loading it at startup keeps Qdrant in sync with the image
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))
        points = [PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)]
        for i in range(0, len(points), 256):
            self.client.upsert(collection, points=points[i : i + 256])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        res = self.client.query_points(self.collection, query=query_vector.tolist(), limit=k)
        return [(int(p.id), float(p.score)) for p in res.points]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder, dense=None):
        self.chunks, self.embedder = chunks, embedder
        self.dense_store = dense or NumpyDense(vectors)
        self.bm25 = BM25Index(chunks)

    @classmethod
    def from_parts(cls, parts: list[tuple[list[Chunk], np.ndarray]], embedder) -> "HybridIndex":
        """Merge per-document (chunks, vectors) into one index; chunk ids are renumbered to stay unique."""
        chunks, vectors = [], []
        for doc_chunks, doc_vectors in parts:
            for c, v in zip(doc_chunks, doc_vectors):
                chunks.append(Chunk(len(chunks), c.text, c.source, c.title))
                vectors.append(v)
        return cls(chunks, np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32), embedder)

    @property
    def backend(self) -> str:
        return self.dense_store.name

    def dense(self, query: str, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        return [Hit(self.chunks[i], s) for i, s in self.dense_store.search(self.embedder.embed_query(query), k)]

    def keyword(self, query: str, k: int) -> list[Hit]:
        return self.bm25.search(query, k)


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Merge ranked lists: each chunk scores sum(1 / (k + rank)). Uses ranks only, so BM25 and cosine
    scores never need to be normalised against each other."""
    fused: dict[int, Hit] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            fused.setdefault(hit.chunk.id, Hit(hit.chunk, 0.0)).score += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)
