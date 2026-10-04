from app.protect import AnswerCache, RateLimiter


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_rate_limiter_window():
    c = Clock()
    rl = RateLimiter(2, 60, clock=c)
    assert rl.allow("ip") and rl.allow("ip") and not rl.allow("ip")
    assert rl.allow("other-ip")
    c.t = 61
    assert rl.allow("ip")


def answered(model="gemini-2.5-flash", notes=()):
    return {"status": "answered", "answer": "A [1]", "trace": {"model": model, "llm_notes": list(notes)}}


def test_cache_normalises_and_expires():
    c = Clock()
    cache = AnswerCache(10, 100, clock=c)
    cache.put("What is Kestrel?", answered())
    assert cache.get("  what is KESTREL ? ")["answer"] == "A [1]"
    c.t = 101
    assert cache.get("What is Kestrel?") is None


def test_cache_skips_failed_models_separates_models_and_evicts():
    cache = AnswerCache(2, 100)
    cache.put("b", answered(notes=["key 1 failed"]))
    assert cache.get("b") is None
    cache.put("a", answered(model="extractive-fallback"), "none:extractive-fallback")  # chosen "no model": deterministic
    assert cache.get("a", "none:extractive-fallback") is not None and cache.get("a", "gemini:gemini-2.5-flash") is None
    for q in "cde":
        cache.put(q, {"status": "blocked", "answer": "no", "trace": {}})
    assert cache.get("c") is None and cache.get("e") is not None


def test_cached_copy_is_independent():
    cache = AnswerCache(10, 100)
    cache.put("q", answered())
    cache.get("q")["answer"] = "changed"
    assert cache.get("q")["answer"] == "A [1]"
