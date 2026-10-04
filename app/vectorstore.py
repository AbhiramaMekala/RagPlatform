"""Dense retrieval store backed by Qdrant.

Local mode (default) stores vectors in ./data/qdrant with no server.
Server mode (Docker) connects to QDRANT_URL.
"""
import threading
from contextlib import nullcontext

import numpy as np

from .models import Chunk, Hit


class QdrantStore:
    def __init__(self, url: str, path: str, collection: str):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url) if url else QdrantClient(path=path)
        self.collection = collection
        # embedded (local) mode is not built for concurrent access, so serialise searches there
        self.lock = threading.Lock() if not url else nullcontext()

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    def reset(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def add(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch):
            points = [
                PointStruct(id=c.id, vector=v.tolist(), payload={"text": c.text, "source": c.source, "title": c.title})
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch])
            ]
            self.client.upsert(self.collection, points=points)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        with self.lock:
            result = self.client.query_points(self.collection, query=vector.tolist(), limit=k, with_payload=True)
        return [Hit(_to_chunk(p), float(p.score)) for p in result.points]

    def all_chunks(self) -> list[Chunk]:
        """Read every chunk back (used to build the BM25 index at startup)."""
        chunks, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True)
            chunks.extend(_to_chunk(p) for p in points)
            if offset is None:
                return sorted(chunks, key=lambda c: c.id)


class MemoryStore:
    """Tiny in-memory vector store (cosine similarity with numpy).

    Used for each visitor's private document set in the demo: it's built in milliseconds,
    needs no server, and disappears when the visitor's session ends.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        if len(chunks):
            v = np.asarray(vectors, dtype=np.float32)
            self.vectors = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        else:
            self.vectors = np.zeros((0, 1), dtype=np.float32)

    def count(self) -> int:
        return len(self.chunks)

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if not self.chunks:
            return []
        q = np.asarray(vector, dtype=np.float32)
        sims = self.vectors @ (q / (np.linalg.norm(q) + 1e-9))
        return [Hit(self.chunks[i], float(sims[i])) for i in np.argsort(-sims)[:k]]

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)


def _to_chunk(point) -> Chunk:
    p = point.payload
    return Chunk(id=int(point.id), text=p["text"], source=p["source"], title=p["title"])
