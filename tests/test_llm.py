"""LLM fallback chain: visitor key -> Gemini key pool -> extractive. No network calls."""
from app.config import Settings
from app.llm import POOL, ExtractiveGenerator, Generation, GeneratorChain, KeyPool, LLMRouter, redact_secrets
from app.models import Chunk, Hit
from tests.fakes import FakeEmbedder

HITS = [Hit(Chunk(0, "The on-call stipend is 400 dollars per week. Handoff happens on Tuesdays.", "oncall.md", "On-call"), 1.0)]


class Boom(Exception):
    def __init__(self, status, msg="error"):
        super().__init__(msg)
        self.status_code = status


class Fake:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.calls = name, fail, 0

    def generate(self, q, hits):
        self.calls += 1
        if self.fail:
            raise self.fail
        return Generation(f"answer from {self.name} [1]", self.name)


def test_first_working_step_answers():
    user, gem = Fake("user", Boom(401, "Incorrect API key provided: sk-abcd1234efgh5678")), Fake("gemini")
    pool = KeyPool(["k1"], lambda k: gem)
    out = GeneratorChain([("your OpenAI key", user), ("Gemini", POOL)], ExtractiveGenerator(FakeEmbedder()), pool).generate("q", HITS)
    assert out.text == "answer from gemini [1]"
    assert out.provider == "Gemini (key 1 of 1)"
    assert out.notes == ["your OpenAI key failed (invalid or unauthorised key)"]


def test_pool_rotates_and_benches_failing_keys():
    t = [0.0]
    gens = {"k1": Fake("k1", Boom(429)), "k2": Fake("k2"), "k3": Fake("k3")}
    pool = KeyPool(list(gens), lambda k: gens[k], clock=lambda: t[0])
    chain = GeneratorChain([("Gemini", POOL)], ExtractiveGenerator(FakeEmbedder()), pool)

    assert chain.generate("q", HITS).model == "k2"  # k1 hit its quota -> benched, k2 answers
    assert chain.generate("q", HITS).model == "k2"  # round-robin starts at k2 now
    assert chain.generate("q", HITS).model == "k3"
    assert gens["k1"].calls == 1  # benched key was skipped
    t[0] += 61  # cooldown over
    gens["k1"].fail = None
    assert {chain.generate("q", HITS).model for _ in range(3)} == {"k1", "k2", "k3"}


def test_everything_fails_falls_back_to_extractive():
    pool = KeyPool(["k1", "k2"], lambda k: Fake(k, Boom(429)))
    out = GeneratorChain([("Gemini", POOL)], ExtractiveGenerator(FakeEmbedder()), pool).generate("on-call stipend", HITS)
    assert out.model == "extractive-fallback" and "[1]" in out.text
    assert len(out.notes) == 2


def test_router_without_any_keys_is_extractive():
    out = LLMRouter(Settings(gemini_api_keys=[], openai_api_key=""), FakeEmbedder()).for_request(None).generate("stipend", HITS)
    assert out.model == "extractive-fallback" and out.notes == []


def test_router_skips_malformed_visitor_key():
    chain = LLMRouter(Settings(gemini_api_keys=[], openai_api_key=""), FakeEmbedder()).for_request("hello")
    out = chain.generate("stipend", HITS)
    assert out.model == "extractive-fallback" and "doesn't look valid" in out.notes[0]


def test_keys_never_leak_into_messages():
    msg = redact_secrets("Incorrect API key provided: sk-proj-abcdEFGH1234. Also AIzaSyA1234567890abcdef and api_key=secret123")
    assert "abcdEFGH1234" not in msg and "1234567890abcdef" not in msg and "secret123" not in msg


def test_client_creation_failure_falls_through():
    def broken():
        raise ImportError("langchain_openai not installed")

    out = GeneratorChain([("your OpenAI key", broken)], ExtractiveGenerator(FakeEmbedder())).generate("stipend", HITS)
    assert out.model == "extractive-fallback" and "your OpenAI key failed" in out.notes[0]


def test_valid_looking_visitor_key_is_tried_first():
    router = LLMRouter(Settings(gemini_api_keys=[], openai_api_key=""), FakeEmbedder())
    chain = router.for_request("sk-" + "a" * 40)
    assert chain.steps[0][0] == "your OpenAI key" and callable(chain.steps[0][1])
