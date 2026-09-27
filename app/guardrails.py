"""Guardrails layer.

1. Input guard  - runs BEFORE retrieval: blocks prompt-injection / harmful queries, redacts personal data.
2. Relevance guard - runs AFTER reranking: refuses if nothing relevant was found (prevents made-up answers).
3. Output guard - runs AFTER generation: checks every answer sentence is supported by the retrieved
   text (groundedness score). Low score = likely hallucination, flagged to the user.
"""
import re
from dataclasses import dataclass, field

import numpy as np

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (your|the|all) (instructions|rules|guidelines)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"\byou are now\b",
    r"\bjailbreak\b",
    r"\bDAN mode\b",
    r"pretend (you have|there are) no (rules|restrictions)",
]
HARMFUL_PATTERNS = [
    r"\b(make|build|create)\b.{0,30}\b(bomb|explosive|weapon|nerve agent)",
    r"\b(write|create|build)\b.{0,30}\b(malware|ransomware|keylogger|virus)\b",
    r"\bsteal\b.{0,30}\b(password|credential|credit card)s?",
    r"\b(kill|hurt) (myself|someone|people)\b",
]
PII_PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "PHONE": r"\+?\d{1,2}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
}


@dataclass
class InputCheck:
    allowed: bool
    query: str  # possibly redacted
    reason: str = ""
    redactions: list[str] = field(default_factory=list)


def check_input(query: str, max_chars: int = 500) -> InputCheck:
    q = (query or "").strip()
    if not q:
        return InputCheck(False, q, "Empty question.")
    if len(q) > max_chars:
        return InputCheck(False, q, f"Question too long (max {max_chars} characters).")
    for p in INJECTION_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return InputCheck(False, q, "Blocked: looks like a prompt-injection attempt.")
    for p in HARMFUL_PATTERNS:
        if re.search(p, q, re.IGNORECASE):
            return InputCheck(False, q, "Blocked: unsafe request.")
    redactions = []
    for label, p in PII_PATTERNS.items():
        if re.search(p, q):
            q = re.sub(p, f"[{label}]", q)
            redactions.append(label)
    return InputCheck(True, q, redactions=redactions)


def check_relevance(best_score: float | None, min_score: float) -> bool:
    return best_score is not None and best_score >= min_score


# ---------- output guard: groundedness ----------

def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'`(\[])", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 3}


@dataclass
class GroundingCheck:
    score: float  # fraction of answer sentences supported by the context (0-1)
    hallucination: bool
    unsupported: list[str]


def check_grounding(answer: str, contexts: list[str], embedder, sim_threshold: float, min_groundedness: float) -> GroundingCheck:
    """A sentence is 'supported' if it is semantically close to some context sentence
    (embedding similarity) OR most of its content words appear in the context (lexical overlap)."""
    answer = re.sub(r"\[\d+\]", "", answer)  # ignore citation markers
    sentences = split_sentences(answer)
    ctx_sentences = [s for c in contexts for s in split_sentences(c)]
    if not sentences or not ctx_sentences:
        return GroundingCheck(1.0, False, [])

    a = _normalise(embedder.embed_documents(sentences))
    c = _normalise(embedder.embed_documents(ctx_sentences))
    best_sim = (a @ c.T).max(axis=1)
    ctx_words = _content_words(" ".join(contexts))

    unsupported = []
    for sent, sim in zip(sentences, best_sim):
        words = _content_words(sent)
        overlap = len(words & ctx_words) / len(words) if words else 1.0
        if sim < sim_threshold and overlap < 0.8:
            unsupported.append(sent)
    score = 1 - len(unsupported) / len(sentences)
    return GroundingCheck(round(score, 3), score < min_groundedness, unsupported)


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
