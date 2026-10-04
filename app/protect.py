"""Small guards for a public deployment: per-visitor rate limits and an answer cache.

Both are in-memory, which is right for a single Cloud Run instance (max instances = 1).
"""
import copy
import re
import threading
import time
from collections import OrderedDict, deque


class RateLimiter:
    """At most `limit` events per `window` seconds per key (visitor IP)."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._events: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self.clock()
        with self._lock:
            q = self._events.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self._events) > 5000:  # forget idle visitors
                for k in [k for k, v in self._events.items() if not v or now - v[-1] > self.window]:
                    del self._events[k]
            return True


class AnswerCache:
    """Most visitors click the same sample questions. Answering those from memory makes the demo
    instant and saves the free LLM quota. Only used for the shared sample documents and only for
    answers worth repeating (an LLM answer, or a deterministic guardrail result)."""

    def __init__(self, max_items: int, ttl_seconds: float, clock=time.monotonic):
        self.max_items, self.ttl, self.clock = max_items, ttl_seconds, clock
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(question: str) -> str:
        return re.sub(r"[^a-z0-9$@.%-]+", " ", question.lower()).strip()

    def get(self, question: str) -> dict | None:
        k = self.key(question)
        with self._lock:
            item = self._items.get(k)
            if not item or self.clock() - item[0] > self.ttl:
                self._items.pop(k, None)
                return None
            self._items.move_to_end(k)
            return copy.deepcopy(item[1])

    def put(self, question: str, result: dict) -> None:
        if self.max_items <= 0 or not worth_caching(result):
            return
        with self._lock:
            self._items[self.key(question)] = (self.clock(), copy.deepcopy(result))
            self._items.move_to_end(self.key(question))
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)


def worth_caching(result: dict) -> bool:
    if result["status"] in ("blocked", "no_relevant_context"):
        return True
    t = result.get("trace", {})
    # don't pin a fallback answer: next time a working LLM may answer better
    return result["status"] == "answered" and t.get("model") != "extractive-fallback" and not t.get("llm_notes")
