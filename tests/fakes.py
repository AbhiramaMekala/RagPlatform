"""Tiny stand-ins for the ML models so tests run in seconds with no downloads."""
import re

import numpy as np

from app.models import Chunk, Hit


class FakeEmbedder:
    """Hashed bag-of-words vectors: similar wording -> similar vectors."""

    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9_]+", text.lower()):
            v[hash(w) % self.dim] += 1
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeStore:
    def __init__(self, chunks, embedder):
        self.chunks, self.vectors = chunks, embedder.embed_documents([c.text for c in chunks])

    def search(self, vector, k):
        sims = self.vectors @ vector
        return [Hit(self.chunks[i], float(sims[i])) for i in np.argsort(-sims)[:k]]


class FakeReranker:
    """Score = number of shared words (minus 5 so unrelated queries score below the threshold)."""

    def rerank(self, query, hits, top_k):
        q = set(re.findall(r"[a-z]+", query.lower()))
        scored = [Hit(h.chunk, len(q & set(re.findall(r"[a-z]+", h.chunk.text.lower()))) - 5.0) for h in hits]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]


CORPUS = [
    Chunk(0, "Constants are usually defined on a module level and written in all capital letters with underscores separating words.", "pep-0008.rst", "PEP 8: Style Guide"),
    Chunk(1, "Function names should be lowercase, with words separated by underscores as necessary to improve readability.", "pep-0008.rst", "PEP 8: Style Guide"),
    Chunk(2, "The assignment expression operator := assigns values to variables as part of an expression. It is nicknamed the walrus operator.", "pep-0572.rst", "PEP 572: Assignment Expressions"),
    Chunk(3, "F-strings provide a concise way to embed expressions inside string literals using a minimal syntax prefixed with f.", "pep-0498.rst", "PEP 498: Literal String Interpolation"),
]
