"""Answer generation, with a fallback chain so the public demo keeps working for free.

For each question the generators are tried in this order:
  1. the visitor's own OpenAI key (sent per request, never stored or logged)
  2. a pool of free Gemini API keys (GEMINI_API_KEYS) - rotated round-robin; a key that fails
     (quota, rate limit, invalid) is benched for a while and the next key is tried
  3. the server's OpenAI key, if OPENAI_API_KEY is set
  4. an extractive fallback (no LLM): picks the most relevant sentences, so the demo never breaks
"""
import logging
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .guardrails import split_sentences
from .models import Hit

log = logging.getLogger("rag.llm")

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages.
Rules:
- Cite passages inline like [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Be concise (at most 5 sentences). Do not use outside knowledge."""

# USD per 1M tokens (input, output). Update if pricing changes. Gemini free tier = 0.
PRICING = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o": (2.50, 10.00)}
OPENAI_KEY_FORMAT = re.compile(r"^sk-[A-Za-z0-9_\-]{20,300}$")
POOL = object()  # placeholder step in a chain: "try the Gemini key pool here"


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    provider: str = ""  # e.g. "your OpenAI key", "Gemini (free key 2 of 4)", "extractive fallback"
    notes: list[str] = field(default_factory=list)  # why earlier options were skipped


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title})\n{h.chunk.text}" for i, h in enumerate(hits, start=1))


def redact_secrets(text: str) -> str:
    """API errors sometimes echo part of the key. Never let that reach traces, logs or the browser."""
    text = re.sub(r"sk-[A-Za-z0-9_\-*]{4,}", "sk-…", str(text))
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "AIza…", text)
    return re.sub(r"(?i)(api[_-]?key[\"'=: ]+)[^\s\"',}]+", r"\1…", text)


def describe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return "invalid or unauthorised key"
    if status == 429:
        return "rate limit or quota exceeded"
    if status == 400 and "key" in str(exc).lower():
        return "invalid key"
    if status:
        return f"HTTP {status}"
    msg = redact_secrets(str(exc))
    return f"{type(exc).__name__}: {msg[:120]}" if msg else type(exc).__name__


class ChatGenerator:
    """Any OpenAI-compatible chat API via LangChain (OpenAI itself, or Gemini's OpenAI-compatible endpoint)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, timeout: float = 25.0, max_retries: int = 0):
        from langchain_openai import ChatOpenAI

        self.model = model
        kwargs = {"base_url": base_url} if base_url else {}
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0, timeout=timeout, max_retries=max_retries, **kwargs)

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        messages = [("system", SYSTEM_PROMPT), ("human", f"Context:\n{build_context(hits)}\n\nQuestion: {question}")]
        msg = self.llm.invoke(messages)
        usage = msg.usage_metadata or {}
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        pin, pout = PRICING.get(self.model, (0.0, 0.0))
        text = msg.content if isinstance(msg.content, str) else " ".join(str(p) for p in msg.content)
        return Generation(text.strip(), self.model, tin, tout, round((tin * pin + tout * pout) / 1_000_000, 6))


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
            return Generation("I don't know based on the provided documents.", self.model, provider="extractive fallback")
        q = self.embedder.embed_query(question)
        m = self.embedder.embed_documents([s for _, s in candidates])
        sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
        best = sorted(np.argsort(-sims)[:3])  # keep original reading order
        text = " ".join(f"{candidates[j][1]} [{candidates[j][0]}]" for j in best)
        return Generation(text, self.model, provider="extractive fallback")


class KeyPool:
    """Round-robin over several API keys; a key that fails is benched for `cooldown` seconds."""

    def __init__(self, keys: list[str], make_generator, clock=time.monotonic):
        self.keys = list(keys)
        self.make_generator = make_generator  # key -> generator
        self.clock = clock
        self._next = 0
        self._bench: dict[int, float] = {}
        self._cache: dict[int, object] = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.keys)

    def order(self) -> list[int]:
        """Key indexes to try for one request: available keys first (starting at the round-robin pointer)."""
        with self._lock:
            n = len(self.keys)
            if not n:
                return []
            start, self._next = self._next, (self._next + 1) % n
            now = self.clock()
            ring = [(start + i) % n for i in range(n)]
            return [i for i in ring if self._bench.get(i, 0) <= now]

    def generator(self, i: int):
        with self._lock:
            if i not in self._cache:
                self._cache[i] = self.make_generator(self.keys[i])
            return self._cache[i]

    def bench(self, i: int, exc: Exception):
        status = getattr(exc, "status_code", None)
        seconds = 3600 if status in (400, 401, 403) else 60  # bad key: 1 hour; quota/rate/network: 1 minute
        with self._lock:
            self._bench[i] = self.clock() + seconds


class GeneratorChain:
    """Tries each (label, generator) in order; always ends with the extractive fallback.

    A step's generator may also be a zero-argument factory (built lazily, so a failure while
    creating the client is handled like any other failure) or POOL (the Gemini key pool)."""

    def __init__(self, steps: list, fallback: ExtractiveGenerator, pool: KeyPool | None = None, notes: list[str] | None = None):
        self.steps, self.fallback, self.pool = steps, fallback, pool
        self.initial_notes = list(notes or [])
        self.model = "auto"

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        notes = list(self.initial_notes)
        for label, gen, pool_index in self._candidates(notes):
            try:
                if callable(gen) and not hasattr(gen, "generate"):
                    gen = gen()
                out = gen.generate(question, hits)
                out.provider, out.notes = label, notes
                return out
            except Exception as exc:
                reason = describe_error(exc)
                log.warning("LLM step '%s' failed: %s", label, reason)
                notes.append(f"{label} failed ({reason})")
                if pool_index is not None:
                    self.pool.bench(pool_index, exc)
        out = self.fallback.generate(question, hits)
        out.notes = notes
        return out

    def _candidates(self, notes: list[str]):
        for label, gen in self.steps:
            if gen is POOL:
                if not self.pool or not len(self.pool):
                    continue
                order = self.pool.order()
                if not order:
                    notes.append("all Gemini keys are cooling down after recent errors")
                for i in order:
                    yield f"{label} (key {i + 1} of {len(self.pool)})", (lambda i=i: self.pool.generator(i)), i
            else:
                yield label, gen, None


class LLMRouter:
    """Builds the per-request generator chain."""

    def __init__(self, settings, embedder):
        self.s = settings
        self.fallback = ExtractiveGenerator(embedder)
        self.gemini = KeyPool(
            settings.gemini_api_keys,
            lambda key: ChatGenerator(settings.gemini_model, key, settings.gemini_base_url, settings.llm_timeout),
        )
        self._server_openai = None

    def _server_generator(self):
        if self._server_openai is None:
            self._server_openai = ChatGenerator(self.s.llm_model, self.s.openai_api_key, timeout=self.s.llm_timeout, max_retries=1)
        return self._server_openai

    def info(self) -> dict:
        return {"user_key_model": self.s.llm_model, "gemini_model": self.s.gemini_model if len(self.gemini) else None,
                "gemini_keys": len(self.gemini), "server_openai": bool(self.s.openai_api_key)}

    def for_request(self, user_openai_key: str | None = None) -> GeneratorChain:
        steps, notes = [], []
        key = (user_openai_key or "").strip()
        if key:
            if OPENAI_KEY_FORMAT.match(key):
                steps.append(("your OpenAI key", lambda: ChatGenerator(self.s.llm_model, key, timeout=self.s.llm_timeout)))
            else:
                notes.append("the OpenAI key you entered doesn't look valid (should start with sk-), so it was skipped")
        steps.append((self.s.gemini_model, POOL))
        if self.s.openai_api_key:
            steps.append(("server OpenAI key", self._server_generator))
        return GeneratorChain(steps, self.fallback, self.gemini, notes)
