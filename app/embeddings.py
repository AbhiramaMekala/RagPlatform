"""Text -> vectors with fastembed (ONNX on CPU: no GPU, no API key, no per-call cost)."""
import numpy as np


class Embedder:
    def __init__(self, model_name: str, threads: int = 0):
        from fastembed import TextEmbedding  # imported lazily so tests don't need it

        self.model = TextEmbedding(model_name, **({"threads": threads} if threads else {}))
        self.dim = len(self.embed_query("warmup"))  # also warms up the ONNX session

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.array(list(self.model.embed(texts, batch_size=64)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the instruction prefix that bge models expect for queries
        return np.array(next(iter(self.model.query_embed(text))), dtype=np.float32)
