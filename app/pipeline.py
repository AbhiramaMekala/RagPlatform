"""The RAG pipeline - read this file first.

question
  -> [guard.input]      block prompt injection / unsafe requests, redact personal data
  -> [retrieve.dense]   vector search (meaning)       \
  -> [retrieve.bm25]    keyword search (exact terms)   } hybrid retrieval
  -> [fusion.rrf]       merge both ranked lists       /
  -> [rerank]           cross-encoder keeps the best top_k
  -> [guard.relevance]  nothing relevant? answer "I don't know" instead of guessing
  -> [generate]         LLM answer citing [n] (fallback chain: visitor key -> Gemini -> extractive)
  -> [guard.grounding]  hallucination check: is every sentence supported by the sources?
  -> answer + sources + trace
"""
import logging

from . import guardrails
from .index import reciprocal_rank_fusion
from .llm import ExtractiveGenerator, describe_error, redact_secrets
from .tracing import Trace

log = logging.getLogger("rag.pipeline")
IDK = "I don't know based on the provided documents."


class RAGPipeline:
    def __init__(self, settings, embedder, reranker, traces):
        self.s, self.embedder, self.reranker, self.traces = settings, embedder, reranker, traces
        self.fallback = ExtractiveGenerator(embedder)

    def ask(self, question: str, index, generator, trace_meta: dict | None = None, persist: bool = True) -> dict:
        """index: what to search (shared samples or a visitor's private set).
        generator: who writes the answer for this request (see llm.LLMRouter).
        persist=False keeps the trace in memory only (questions about private uploads)."""
        t = Trace(question)
        t.data.update(trace_meta or {})
        t.persist = persist

        # 1. input guardrail
        with t.span("guard.input") as sp:
            check = guardrails.check_input(question, self.s.max_query_chars)
            sp.update(allowed=check.allowed, redactions=check.redactions)
        if not check.allowed:
            return self._done(t, status="blocked", answer=check.reason, sources=[], guard={"input": check.reason})
        q = check.query

        # 2. hybrid retrieval
        with t.span("retrieve.dense", backend=index.backend) as sp:
            dense = index.dense(q, self.s.candidates)
            sp["hits"] = len(dense)
        with t.span("retrieve.bm25") as sp:
            keyword = index.keyword(q, self.s.candidates)
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
                              guard={"input": _input_note(check), "relevance": "no relevant documents found"})

        # 5. generation
        llm_error = None
        with t.span("generate") as sp:
            try:
                gen = generator.generate(q, top)
            except Exception as exc:  # the chain already falls back; this is a last line of defence
                llm_error = describe_error(exc)
                log.warning("LLM chain failed, using extractive fallback: %s", llm_error)
                gen = self.fallback.generate(q, top)
            sp.update(model=gen.model, provider=gen.provider, input_tokens=gen.input_tokens, output_tokens=gen.output_tokens)

        # 6. output guardrail
        with t.span("guard.grounding") as sp:
            if gen.text.strip().startswith("I don't know"):
                g = guardrails.GroundingCheck(1.0, False, [])
            else:
                g = guardrails.check_grounding(gen.text, [h.chunk.text for h in top], self.embedder,
                                               self.s.grounding_similarity, self.s.min_groundedness)
            sp.update(groundedness=g.score, hallucination=g.hallucination)

        return self._done(
            t, status="answered", answer=gen.text, sources=sources, model=gen.model, provider=gen.provider,
            llm_notes=[redact_secrets(n) for n in gen.notes], llm_error=llm_error,
            input_tokens=gen.input_tokens, output_tokens=gen.output_tokens, cost_usd=gen.cost_usd,
            groundedness=g.score, hallucination=g.hallucination,
            guard={"input": _input_note(check), "relevance": "passed",
                   "grounding": {"score": g.score, "flagged": g.hallucination, "unsupported": g.unsupported}},
        )

    @staticmethod
    def _sources(top, dense, keyword):
        dense_ids, kw_ids = {h.chunk.id for h in dense}, {h.chunk.id for h in keyword}
        return [{"n": i, "title": h.chunk.title, "source": h.chunk.source, "text": h.chunk.text, "rerank_score": round(h.score, 3),
                 "found_by": [name for name, ids in (("dense", dense_ids), ("bm25", kw_ids)) if h.chunk.id in ids]}
                for i, h in enumerate(top, start=1)]

    def _done(self, t: Trace, **data) -> dict:
        trace = t.finish(**{k: v for k, v in data.items() if k not in ("sources", "guard")})
        self.traces.record(trace, persist=t.persist)
        return {"answer": data["answer"], "status": data["status"], "sources": data.get("sources", []),
                "guardrails": data.get("guard", {}), "trace": trace}


def _input_note(check) -> str:
    return "passed" + (f" (redacted {', '.join(check.redactions)})" if check.redactions else "")
