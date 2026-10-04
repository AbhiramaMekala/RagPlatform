"""Answer generation. The visitor chooses who writes the answer:

  anthropic / openai : the visitor's own API key and chosen model. The key is sent with each request,
                       used for that request only, and never stored or logged.
  gemini             : the site's built-in pool of free Gemini keys (GEMINI_API_KEYS). Used ONLY when
                       the visitor picks it. Keys rotate round-robin; a key that fails (quota, rate limit,
                       invalid) is benched for a while and the next one is tried.
  none               : no language model.

Whatever is chosen, if it can't produce an answer (no key, bad key, quota, outage) the extractive
fallback answers by quoting the most relevant sentences - retrieval, citations and guardrails still work.
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

PROVIDERS = {"anthropic": "Anthropic", "openai": "OpenAI", "gemini": "Gemini", "none": "No model"}
KEY_FORMATS = {
    "anthropic": (re.compile(r"^sk-ant-[A-Za-z0-9_\-]{20,300}$"), "sk-ant-"),
    "openai": (re.compile(r"^sk-(?!ant-)[A-Za-z0-9_\-]{20,300}$"), "sk-"),
}
MODEL_NAME = re.compile(r"^[A-Za-z0-9._:\-]{1,80}$")

# USD per 1M tokens (input, output), matched by model-name prefix. Check the providers' pricing pages
# before relying on these; unknown models show no cost. The Gemini demo keys are on the free tier.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00), "gpt-4o": (2.50, 10.00), "claude-haiku-4-5": (1.00, 5.00),
}
POOL = object()  # placeholder step in a chain: "try the Gemini key pool here"


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None = price unknown for this model
    provider: str = ""  # e.g. "your Anthropic key", "gemini-2.5-flash (key 2 of 4)", "extractive fallback"
    notes: list[str] = field(default_factory=list)  # why the chosen model didn't answer, if it didn't


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
    if status == 404:
        return "model not found for this key"
    if status == 429:
        return "rate limit or quota exceeded"
    if status == 400 and "key" in str(exc).lower():
        return "invalid key"
    if status:
        return f"HTTP {status}"
    msg = redact_secrets(str(exc))
    return f"{type(exc).__name__}: {msg[:120]}" if msg else type(exc).__name__


def price(model: str, tokens_in: int, tokens_out: int) -> float | None:
    for prefix, (pin, pout) in PRICING.items():
        if model.startswith(prefix):
            return round((tokens_in * pin + tokens_out * pout) / 1_000_000, 6)
    return None


class LangChainGenerator:
    """Shared by every chat model: same prompt, same citation format, token usage and cost."""

    model: str
    llm: object

    def generate(self, question: str, hits: list[Hit]) -> Generation:
        messages = [("system", SYSTEM_PROMPT), ("human", f"Context:\n{build_context(hits)}\n\nQuestion: {question}")]
        msg = self.llm.invoke(messages)
        usage = msg.usage_metadata or {}
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        content = msg.content
        if not isinstance(content, str):  # list of content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return Generation(content.strip(), self.model, tin, tout, price(self.model, tin, tout))


class OpenAICompatibleGenerator(LangChainGenerator):
    """OpenAI, or Gemini through Google's OpenAI-compatible endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None, timeout: float = 25.0):
        from langchain_openai import ChatOpenAI

        self.model = model
        kwargs = {"base_url": base_url} if base_url else {}
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0, timeout=timeout, max_retries=0, **kwargs)


class AnthropicGenerator(LangChainGenerator):
    def __init__(self, model: str, api_key: str, timeout: float = 25.0):
        from langchain_anthropic import ChatAnthropic

        self.model = model
        self.llm = ChatAnthropic(model=model, api_key=api_key, temperature=0, max_tokens=600, timeout=timeout, max_retries=0)


class ExtractiveGenerator:
    """No-LLM answer: the 3 context sentences most similar to the question, with citations."""

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
    """Round-robin over several API keys; a key that fails is benched for a while."""

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
        """Key indexes to try for one request: available keys, starting at the round-robin pointer."""
        with self._lock:
            n = len(self.keys)
            if not n:
                return []
            start, self._next = self._next, (self._next + 1) % n
            now = self.clock()
            return [i for i in ((start + k) % n for k in range(n)) if self._bench.get(i, 0) <= now]

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
    """Tries each (label, generator) step in order; always ends with the extractive fallback.

    A step's generator may be a zero-argument factory (built lazily, so a failure while creating the
    client is handled like any other failure) or POOL (the Gemini key pool)."""

    def __init__(self, steps: list, fallback: ExtractiveGenerator, pool: KeyPool | None = None, notes: list[str] | None = None):
        self.steps, self.fallback, self.pool = steps, fallback, pool
        self.initial_notes = list(notes or [])

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
                order = self.pool.order() if self.pool else []
                if not order:
                    notes.append("all Gemini demo keys are cooling down after recent errors")
                for i in order:
                    yield f"{label} (demo key {i + 1} of {len(self.pool)})", (lambda i=i: self.pool.generator(i)), i
            else:
                yield label, gen, None


class LLMRouter:
    """Builds the generator chain for one request from the visitor's choice."""

    def __init__(self, settings, embedder):
        self.s = settings
        self.fallback = ExtractiveGenerator(embedder)
        self.gemini = KeyPool(
            settings.gemini_api_keys,
            lambda key: OpenAICompatibleGenerator(settings.gemini_model, key, settings.gemini_base_url, settings.llm_timeout),
        )

    def info(self) -> dict:
        """What the UI offers. Never includes keys."""
        return {
            "default_provider": "anthropic",
            "anthropic": {"models": self.s.anthropic_models, "key_prefix": "sk-ant-"},
            "openai": {"models": self.s.openai_models, "key_prefix": "sk-"},
            "gemini": {"model": self.s.gemini_model, "available": len(self.gemini) > 0, "keys": len(self.gemini)},
        }

    def resolve(self, provider: str | None, model: str | None) -> tuple[str, str]:
        """Normalise the visitor's choice to a known provider and a safe model name."""
        provider = provider if provider in PROVIDERS else "none"
        if provider == "gemini":
            return provider, self.s.gemini_model  # the site's keys: the site picks the model
        if provider in ("anthropic", "openai"):
            models = self.s.anthropic_models if provider == "anthropic" else self.s.openai_models
            return provider, model if model and MODEL_NAME.match(model) else models[0]
        return "none", ExtractiveGenerator.model

    def for_request(self, provider: str | None, model: str | None = None, key: str | None = None) -> GeneratorChain:
        provider, model = self.resolve(provider, model)
        name, key = PROVIDERS[provider], (key or "").strip()
        steps, notes = [], []
        if provider in KEY_FORMATS:
            pattern, prefix = KEY_FORMATS[provider]
            if not key:
                notes.append(f"no {name} API key was entered, so the answer was quoted from the documents instead")
            elif not pattern.match(key):
                notes.append(f"that doesn't look like an {name} key (they start with {prefix}), so it wasn't used")
            elif provider == "anthropic":
                steps.append((f"{model} (your key)", lambda: AnthropicGenerator(model, key, self.s.llm_timeout)))
            else:
                steps.append((f"{model} (your key)", lambda: OpenAICompatibleGenerator(model, key, timeout=self.s.llm_timeout)))
        elif provider == "gemini":
            if len(self.gemini):
                steps.append((model, POOL))
            else:
                notes.append("the Gemini demo keys aren't set up on this server")
        return GeneratorChain(steps, self.fallback, self.gemini, notes)
