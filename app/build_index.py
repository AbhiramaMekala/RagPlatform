"""Download the models and build the sample index (run at Docker build time).

    python -m app.build_index

Writes data/index/index.json + vectors.npy. At startup the app loads these instead of embedding
the documents again, so a cold start on Cloud Run only has to load the two ONNX models.
"""
import time

from .config import settings
from .embeddings import Embedder
from .reranker import CrossEncoderReranker
from .workspace import SampleLibrary


def main():
    t0 = time.perf_counter()
    embedder = Embedder(settings.embed_model)
    CrossEncoderReranker(settings.rerank_model)  # downloads the reranker into the image too
    lib = SampleLibrary.load(settings, embedder)
    print(f"[build_index] {len(lib.docs)} documents, {len(lib.index.chunks)} chunks -> {settings.index_dir} "
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
