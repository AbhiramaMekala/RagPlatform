"""Builds everything the API needs, once per process."""
import logging
import time
from dataclasses import dataclass

from .llm import LLMRouter
from .pipeline import RAGPipeline
from .tracing import TraceStore
from .workspace import Sessions

log = logging.getLogger("rag.services")


@dataclass
class Services:
    settings: object
    pipeline: RAGPipeline
    sessions: Sessions
    router: LLMRouter


def build_services(settings) -> Services:
    from .embeddings import Embedder
    from .reranker import CrossEncoderReranker
    from .workspace import SampleLibrary

    t0 = time.perf_counter()
    embedder = Embedder(settings.embed_model, settings.onnx_threads)
    reranker = CrossEncoderReranker(settings.rerank_model, settings.onnx_threads)
    library = SampleLibrary.load(settings, embedder)  # loads the prebuilt index (no embedding at startup)
    traces = TraceStore(settings.traces_file or None, langfuse=settings.langfuse_enabled)
    router = LLMRouter(settings, embedder)
    log.info("Ready in %.1fs. LLM chain: visitor OpenAI key -> %d Gemini key(s)%s -> extractive fallback",
             time.perf_counter() - t0, len(router.gemini), " -> server OpenAI key" if settings.openai_api_key else "")
    return Services(settings, RAGPipeline(settings, embedder, reranker, traces), Sessions(library, embedder, settings), router)
