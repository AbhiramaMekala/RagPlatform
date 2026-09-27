"""Keyword retrieval with BM25 (Okapi). Small, dependency-free implementation so it's easy to read.

Dense search is good at meaning ("can I take a holiday without asking?"); BM25 is good at exact terms
("INC-2291", "Tollbooth", "Kestrel"). Hybrid search uses both.
"""
import math
import re
from collections import Counter

from .models import Chunk, Hit

STOPWORDS = set(
    "a an the and or but if of to in on for with at by from as is are was were be been it its this that "
    "these those do does did how what why when where which who can should would will i you we they".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(c.text)) for c in chunks]
        self.doc_len = [sum(tf.values()) for tf in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(len(chunks), 1)
        df = Counter(term for tf in self.doc_tokens for term in tf)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int) -> list[Hit]:
        terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self.doc_tokens):
            s = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [Hit(self.chunks[i], s) for s, i in scores[:k]]
