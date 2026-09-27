"""Text -> vectors. Uses fastembed (ONNX) so it runs on a laptop CPU with no API key."""
import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily so tests don't need it

        self.model = TextEmbedding(model_name)
        self.dim = len(self.embed_query("warmup"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.embed(texts, batch_size=64)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed adds the instruction prefix that bge models expect for queries
        return np.array(next(iter(self.model.query_embed(text))), dtype=np.float32)
