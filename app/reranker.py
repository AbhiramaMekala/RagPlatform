"""Cross-encoder reranking: reads (question, chunk) together and scores true relevance.

Retrievers are fast but approximate; the reranker is slower but much more precise,
so it only runs on the ~20 fused candidates.
"""
from .models import Hit


class CrossEncoderReranker:
    def __init__(self, model_name: str, threads: int = 0):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name, **({"threads": threads} if threads else {}))
        list(self.model.rerank("warmup", ["warmup"]))  # load the ONNX session now, not on the first visitor's request

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = list(self.model.rerank(query, [h.chunk.text for h in hits]))
        ranked = sorted((Hit(h.chunk, float(s)) for h, s in zip(hits, scores)), key=lambda h: h.score, reverse=True)
        return ranked[:top_k]
