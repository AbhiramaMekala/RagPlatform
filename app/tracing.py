"""Observability: every request produces one trace (each pipeline step timed, tokens, cost).

Traces are:
  - kept in memory for /api/traces, /api/metrics and the demo UI,
  - logged to stdout as one JSON line (Cloud Run ships stdout to Cloud Logging, so they're searchable there),
  - optionally appended to TRACES_FILE and exported to Langfuse (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).
Traces about a visitor's private uploads are kept in memory only.
"""
import json
import logging
import statistics
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("rag.trace")


class Trace:
    def __init__(self, question: str):
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        self.started = time.perf_counter()
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.spans: list[dict] = []
        self.data: dict = {}
        self.persist = True

    @contextmanager
    def span(self, name: str, **meta):
        start = time.perf_counter()
        record = {"name": name, "start_ms": round((start - self.started) * 1000, 1), **meta}
        try:
            yield record  # the step can add metadata: record["hits"] = 20
        finally:
            record["ms"] = round((time.perf_counter() - start) * 1000, 1)
            self.spans.append(record)

    def finish(self, **data) -> dict:
        self.data.update(data)
        return {"id": self.id, "timestamp": self.timestamp, "question": self.question,
                "total_ms": round((time.perf_counter() - self.started) * 1000, 1), "spans": self.spans, **self.data}


class TraceStore:
    def __init__(self, path: str | Path | None = None, keep: int = 500, langfuse: bool = False, log_json: bool = True):
        self.recent: deque[dict] = deque(maxlen=keep)
        self.path = Path(path) if path else None
        self.langfuse, self.log_json = langfuse, log_json
        self.cache_hits = 0
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: dict, persist: bool = True) -> None:
        with self.lock:
            self.recent.append(trace)
            if persist and self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(trace) + "\n")
        if not persist:
            return
        if self.log_json:
            log.info(json.dumps({"message": "rag trace", "severity": "INFO", **_summary(trace)}))
        if self.langfuse:
            export_to_langfuse(trace)

    def metrics(self) -> dict:
        with self.lock:
            traces = list(self.recent)
        if not traces:
            return {"requests": 0, "cache_hits": self.cache_hits}
        lat = sorted(t["total_ms"] for t in traces)
        answered = [t for t in traces if t.get("status") == "answered"]
        return {
            "requests": len(traces),
            "cache_hits": self.cache_hits,
            "blocked": sum(t.get("status") == "blocked" for t in traces),
            "no_answer": sum(t.get("status") == "no_relevant_context" for t in traces),
            "hallucination_flags": sum(bool(t.get("hallucination")) for t in answered),
            "avg_groundedness": round(statistics.mean(t.get("groundedness", 1) for t in answered), 3) if answered else None,
            "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95), "max": lat[-1]},
            "tokens_total": sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in traces),
            "cost_usd_total": round(sum(t.get("cost_usd") or 0 for t in traces), 6),  # None = unpriced model
        }


def _summary(trace: dict) -> dict:
    """What goes to the logs: everything except retrieved text."""
    keep = ("id", "question", "status", "total_ms", "model", "provider", "groundedness", "hallucination",
            "input_tokens", "output_tokens", "cost_usd", "llm_notes")
    return {**{k: trace[k] for k in keep if k in trace}, "spans": {s["name"]: s["ms"] for s in trace.get("spans", [])}}


def _pct(sorted_values: list[float], p: int) -> float:
    idx = min(len(sorted_values) - 1, max(0, round(p / 100 * len(sorted_values) + 0.5) - 1))
    return sorted_values[idx]


_langfuse_warned = False


def export_to_langfuse(trace: dict) -> None:
    """Send one trace to Langfuse (SDK v3). Never breaks the request if Langfuse is down."""
    global _langfuse_warned
    try:
        from langfuse import get_client

        lf = get_client()
        with lf.start_as_current_span(name="rag-query", input={"question": trace["question"]}) as root:
            for s in trace["spans"]:
                meta = {k: v for k, v in s.items() if k != "name"}
                if s["name"] == "generate":
                    root.start_generation(
                        name="generate", model=trace.get("model"), output=trace.get("answer"),
                        usage_details={"input": trace.get("input_tokens", 0), "output": trace.get("output_tokens", 0)},
                        cost_details={"total": trace.get("cost_usd", 0.0)}, metadata=meta,
                    ).end()
                else:
                    root.start_span(name=s["name"], metadata=meta).end()
            root.update(output={"answer": trace.get("answer"), "status": trace.get("status")})
            root.update_trace(name="rag-query", tags=[trace.get("status", "unknown")], metadata={"total_ms": trace["total_ms"]})
    except Exception as exc:  # observability must never take the API down
        if not _langfuse_warned:
            log.warning("Langfuse export failed (continuing without it): %s", exc)
            _langfuse_warned = True
