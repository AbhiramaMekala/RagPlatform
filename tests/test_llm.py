"""Model choice and fallbacks. No network calls: generators are fakes."""
from app.config import Settings
from app.llm import POOL, ExtractiveGenerator, Generation, GeneratorChain, KeyPool, LLMRouter, price, redact_secrets
from app.models import Chunk, Hit
from tests.fakes import FakeEmbedder

HITS = [Hit(Chunk(0, "The on-call stipend is 400 dollars per week. Handoff happens on Tuesdays.", "oncall.md", "On-call"), 1.0)]
ANT_KEY, OAI_KEY = "sk-ant-" + "a" * 40, "sk-proj-" + "b" * 40


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


def router(keys=()):
    return LLMRouter(Settings(gemini_api_keys=list(keys)), FakeEmbedder())


def test_default_is_no_model_and_quotes_the_documents():
    out = router(["g1"]).for_request(None).generate("on-call stipend", HITS)
    assert out.model == "extractive-fallback" and out.notes == []


def test_visitor_key_is_required_and_checked():
    r = router()
    missing = r.for_request("anthropic", None, "").generate("stipend", HITS)
    assert missing.model == "extractive-fallback" and "no Anthropic API key" in missing.notes[0]
    wrong = r.for_request("anthropic", None, OAI_KEY).generate("stipend", HITS)  # an OpenAI key in the Anthropic box
    assert wrong.model == "extractive-fallback" and "sk-ant-" in wrong.notes[0]
    assert "OpenAI key" in r.for_request("openai", None, ANT_KEY).generate("stipend", HITS).notes[0]


def test_valid_keys_build_the_right_step_without_gemini_fallback():
    r = router(["g1", "g2"])
    ant = r.for_request("anthropic", "claude-haiku-4-5-20251001", ANT_KEY)
    assert [label for label, _ in ant.steps] == ["claude-haiku-4-5-20251001 (your key)"]  # Gemini is never added silently
    oai = r.for_request("openai", None, OAI_KEY)
    assert oai.steps[0][0] == "gpt-4o-mini (your key)"  # first model in the list is the default
    assert r.resolve("openai", "bad model; drop table") == ("openai", "gpt-4o-mini")
    assert r.resolve("gemini", "anything") == ("gemini", "gemini-2.5-flash")  # the site picks the Gemini model
    assert r.resolve("mystery", None) == ("none", "extractive-fallback")


def test_gemini_only_when_chosen_and_configured():
    assert router([]).for_request("gemini").generate("stipend", HITS).notes == ["the Gemini demo keys aren't set up on this server"]
    chain = router(["g1"]).for_request("gemini")
    assert chain.steps[0][1] is POOL


def test_failing_visitor_model_falls_back_to_quotes():
    chain = GeneratorChain([("claude (your key)", Fake("c", Boom(401, "invalid x-api-key sk-ant-abcd1234efgh")))], ExtractiveGenerator(FakeEmbedder()))
    out = chain.generate("on-call stipend", HITS)
    assert out.model == "extractive-fallback" and out.notes == ["claude (your key) failed (invalid or unauthorised key)"]


def test_gemini_pool_rotates_and_benches_failing_keys():
    t = [0.0]
    gens = {"k1": Fake("k1", Boom(429)), "k2": Fake("k2"), "k3": Fake("k3")}
    pool = KeyPool(list(gens), lambda k: gens[k], clock=lambda: t[0])
    chain = GeneratorChain([("gemini", POOL)], ExtractiveGenerator(FakeEmbedder()), pool)
    assert chain.generate("q", HITS).model == "k2"  # k1 is over quota -> benched, k2 answers
    assert chain.generate("q", HITS).model == "k2"
    assert chain.generate("q", HITS).model == "k3"
    assert gens["k1"].calls == 1
    t[0] += 61
    gens["k1"].fail = None
    assert {chain.generate("q", HITS).model for _ in range(3)} == {"k1", "k2", "k3"}


def test_all_gemini_keys_failing_quotes_the_documents():
    pool = KeyPool(["k1", "k2"], lambda k: Fake(k, Boom(429)))
    out = GeneratorChain([("gemini", POOL)], ExtractiveGenerator(FakeEmbedder()), pool).generate("on-call stipend", HITS)
    assert out.model == "extractive-fallback" and "[1]" in out.text and len(out.notes) == 2


def test_client_creation_failure_falls_through():
    def broken():
        raise ImportError("langchain_anthropic not installed")

    out = GeneratorChain([("claude (your key)", broken)], ExtractiveGenerator(FakeEmbedder())).generate("stipend", HITS)
    assert out.model == "extractive-fallback" and "failed" in out.notes[0]


def test_info_never_contains_keys():
    info = router(["AIzaSECRETSECRETSECRET"]).info()
    assert "AIza" not in str(info) and info["gemini"]["available"] and info["anthropic"]["models"]


def test_keys_never_leak_and_pricing():
    msg = redact_secrets("Incorrect API key provided: sk-ant-api03-abcdEFGH1234. Also AIzaSyA1234567890abcdef and api_key=secret123")
    assert "abcdEFGH1234" not in msg and "1234567890abcdef" not in msg and "secret123" not in msg
    assert price("gpt-4o-mini", 1_000_000, 0) == 0.15 and price("some-new-model", 10, 10) is None
