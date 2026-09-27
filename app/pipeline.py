"""The RAG pipeline — ties every module together. Read this file first.

question
  -> [guard.input]      block injection / unsafe, redact PII
  -> [retrieve.dense]   Qdrant vector search      \
  -> [retrieve.bm25]    keyword search             } hybrid retrieval
  -> [fusion.rrf]       merge both ranked lists   /
  -> [rerank]           cross-encoder picks best top_k
  -> [guard.relevance]  nothing relevant? say "I don't know"
  -> [generate]         LLM answer with [n] citations
  -> [guard.grounding]  hallucination check
  -> answer + sources + trace
"""
import logging

from . import guardrails
from .llm import ExtractiveGenerator
from .retriever import reciprocal_rank_fusion
from .tracing import Trace

log = logging.getLogger("rag.pipeline")
IDK = "I don't know based on the provided documents."


class RAGPipeline:
    def __init__(self, settings, embedder, retriever, reranker, generator, traces):
        self.s = settings
        self.embedder, self.retriever, self.reranker, self.generator = embedder, retriever, reranker, generator
        self.traces = traces

    def ask(self, question: str) -> dict:
        t = Trace(question)

        # 1. input guardrail
        with t.span("guard.input") as sp:
            check = guardrails.check_input(question, self.s.max_query_chars)
            sp.update(allowed=check.allowed, redactions=check.redactions)
        if not check.allowed:
            return self._done(t, status="blocked", answer=check.reason, sources=[], guard={"input": check.reason})
        q = check.query

        # 2. hybrid retrieval
        with t.span("retrieve.dense") as sp:
            dense = self.retriever.dense(q, self.s.candidates)
            sp["hits"] = len(dense)
        with t.span("retrieve.bm25") as sp:
            keyword = self.retriever.keyword(q, self.s.candidates)
            sp["hits"] = len(keyword)
        with t.span("fusion.rrf") as sp:
            fused = reciprocal_rank_fusion([dense, keyword])[: self.s.candidates]
            sp["candidates"] = len(fused)

        # 3. rerank
        with t.span("rerank") as sp:
            top = self.reranker.rerank(q, fused, self.s.top_k)
            best = top[0].score if top else None
            sp["best_score"] = round(best, 3) if best is not None else None

        sources = self._sources(top, dense, keyword)

        # 4. relevance guardrail
        with t.span("guard.relevance") as sp:
            relevant = guardrails.check_relevance(best, self.s.min_rerank_score)
            sp["relevant"] = relevant
        if not relevant:
            return self._done(t, status="no_relevant_context", answer=IDK, sources=sources,
                              guard={"input": "passed", "relevance": "no relevant documents found"})

        # 5. generation (falls back to extractive if the LLM call fails)
        llm_error = None
        with t.span("generate", model=getattr(self.generator, "model", "")) as sp:
            try:
                gen = self.generator.generate(q, top)
            except Exception as exc:
                log.exception("LLM call failed, using extractive fallback")
                llm_error = str(exc)[:200]
                gen = ExtractiveGenerator(self.embedder).generate(q, top)
            sp.update(model=gen.model, input_tokens=gen.input_tokens, output_tokens=gen.output_tokens, cost_usd=gen.cost_usd)

        # 6. output guardrail (hallucination check)
        with t.span("guard.grounding") as sp:
            if gen.text.strip().startswith("I don't know"):
                g = guardrails.GroundingCheck(1.0, False, [])
            else:
                g = guardrails.check_grounding(gen.text, [h.chunk.text for h in top], self.embedder,
                                               self.s.grounding_similarity, self.s.min_groundedness)
            sp.update(groundedness=g.score, hallucination=g.hallucination)

        return self._done(
            t, status="answered", answer=gen.text, sources=sources, model=gen.model,
            input_tokens=gen.input_tokens, output_tokens=gen.output_tokens, cost_usd=gen.cost_usd,
            groundedness=g.score, hallucination=g.hallucination, llm_error=llm_error,
            guard={"input": "passed" + (f" (redacted {', '.join(check.redactions)})" if check.redactions else ""),
                   "relevance": "passed", "grounding": {"score": g.score, "flagged": g.hallucination, "unsupported": g.unsupported}},
        )

    def _sources(self, top, dense, keyword):
        dense_ids = {h.chunk.id for h in dense}
        kw_ids = {h.chunk.id for h in keyword}
        return [
            {"n": i, "title": h.chunk.title, "source": h.chunk.source, "text": h.chunk.text,
             "rerank_score": round(h.score, 3),
             "found_by": [name for name, ids in (("dense", dense_ids), ("bm25", kw_ids)) if h.chunk.id in ids]}
            for i, h in enumerate(top, start=1)
        ]

    def _done(self, t: Trace, **data) -> dict:
        trace = t.finish(**{k: v for k, v in data.items() if k != "sources"})
        self.traces.record(trace)
        return {"answer": data["answer"], "status": data["status"], "sources": data.get("sources", []),
                "guardrails": data.get("guard", {}), "trace": trace}


def build_pipeline(settings) -> RAGPipeline:
    """Create the real components (downloads models on first run) and index docs if needed."""
    from .bm25 import BM25Index
    from .embeddings import Embedder
    from .ingest import ingest
    from .llm import OpenAIGenerator
    from .reranker import CrossEncoderReranker
    from .retriever import HybridRetriever
    from .tracing import TraceStore
    from .vectorstore import QdrantStore

    embedder = Embedder(settings.embed_model)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_path, settings.collection)
    if store.count() == 0:
        ingest(settings, embedder, store)
    bm25 = BM25Index(store.all_chunks())
    reranker = CrossEncoderReranker(settings.rerank_model)
    if settings.openai_api_key:
        generator = OpenAIGenerator(settings.llm_model, settings.openai_api_key)
    else:
        log.warning("OPENAI_API_KEY not set - using extractive fallback (no LLM)")
        generator = ExtractiveGenerator(embedder)
    traces = TraceStore(settings.traces_file, langfuse=settings.langfuse_enabled)
    return RAGPipeline(settings, embedder, HybridRetriever(embedder, store, bm25), reranker, generator, traces)
