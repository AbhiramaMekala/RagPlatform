from app.bm25 import BM25Index, tokenize
from app.models import Chunk, Hit
from app.retriever import reciprocal_rank_fusion
from tests.fakes import CORPUS


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Walrus operator?") == ["walrus", "operator"]


def test_bm25_finds_exact_keyword():
    hits = BM25Index(CORPUS).search("walrus operator", k=2)
    assert hits[0].chunk.id == 2
    assert hits[0].score > 0


def test_bm25_returns_nothing_for_unknown_terms():
    assert BM25Index(CORPUS).search("kubernetes helm", k=3) == []


def test_rrf_rewards_chunks_found_by_both_retrievers():
    a, b, c = (Chunk(i, f"t{i}", "s", "t") for i in range(3))
    dense = [Hit(a, 0.9), Hit(b, 0.8)]
    keyword = [Hit(b, 12.0), Hit(c, 3.0)]
    fused = reciprocal_rank_fusion([dense, keyword])
    assert fused[0].chunk.id == b.id  # b appears in both lists
    assert {h.chunk.id for h in fused} == {0, 1, 2}
