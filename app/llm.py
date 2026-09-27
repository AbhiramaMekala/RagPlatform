"""Answer generation.

- With OPENAI_API_KEY: LangChain ChatOpenAI writes a cited answer from the retrieved chunks.
- Without a key: an extractive fallback picks the most relevant sentences, so the demo still works offline.
"""
from dataclasses import dataclass

import numpy as np

from .guardrails import split_sentences
from .models import Hit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages.
Rules:
- Cite passages inline like [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Be concise (at most 5 sentences). Do not use outside knowledge."""

# USD per 1M tokens (input, output). Update if OpenAI pricing changes.
PRICING = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o": (2.50, 10.00)}


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))


class OpenAIGenerator:
    def __init__(self, model: str, api_key: str):
        from langchain_openai import ChatOpenAI

        self.model = model
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0, timeout=30)

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        messages = [("system", SYSTEM_PROMPT), ("human", f"Context:\n{build_context(hits)}\n\nQuestion: {question}")]
        msg = self.llm.invoke(messages)
        usage = msg.usage_metadata or {}
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        pin, pout = PRICING.get(self.model, (0.0, 0.0))
        cost = (tin * pin + tout * pout) / 1_000_000
        return Generation(msg.content, self.model, tin, tout, round(cost, 6))


class ExtractiveGenerator:
    """No-LLM fallback: returns the 3 context sentences most similar to the question, with citations."""

    model = "extractive-fallback"

    def __init__(self, embedder):
        self.embedder = embedder

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        seen, candidates = set(), []
        for i, h in enumerate(hits, start=1):
            for s in split_sentences(h.chunk.text):
                if s not in seen:  # overlapping chunks repeat sentences
                    seen.add(s)
                    candidates.append((i, s))
        if not candidates:
            return Generation("I don't know based on the provided documents.", self.model)
        q = self.embedder.embed_query(question)
        m = self.embedder.embed_documents([s for _, s in candidates])
        sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
        best = sorted(np.argsort(-sims)[:3])  # keep original reading order
        return Generation(" ".join(f"{candidates[j][1]} [{candidates[j][0]}]" for j in best), self.model)
