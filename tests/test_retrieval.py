import numpy as np

from app.bm25 import BM25Index, tokenize
from app.index import HybridIndex, NumpyDense, reciprocal_rank_fusion
from app.models import Chunk, Hit
from tests.fakes import CORPUS, FakeEmbedder, corpus_index


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Walrus operator?") == ["walrus", "operator"]


def test_bm25_exact_keyword():
    hits = BM25Index(CORPUS).search("walrus operator", k=2)
    assert hits[0].chunk.id == 2 and hits[0].score > 0
    assert BM25Index(CORPUS).search("kubernetes helm", k=3) == []


def test_rrf_rewards_agreement():
    a, b, c = (Hit(Chunk(i, "t", "s", "T"), 0) for i in range(3))
    fused = reciprocal_rank_fusion([[a, b], [b, c]])
    assert fused[0].chunk.id == 1  # b is in both lists


def test_numpy_dense_ranks_by_cosine():
    store = NumpyDense(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    assert [i for i, _ in store.search(np.array([1, 0.1]), 2)] == [0, 2]
    assert NumpyDense(np.zeros((0, 2))).search(np.array([1, 0]), 3) == []


def test_hybrid_index_dense_and_keyword():
    idx = corpus_index()
    assert idx.dense("walrus operator assignment", 1)[0].chunk.id == 2
    assert idx.keyword("capital letters", 1)[0].chunk.id == 0
    assert idx.backend == "numpy"


def test_from_parts_renumbers_ids():
    emb = FakeEmbedder()
    part = (CORPUS[:2], emb.embed_documents([c.text for c in CORPUS[:2]]))
    idx = HybridIndex.from_parts([part, part], emb)
    assert [c.id for c in idx.chunks] == [0, 1, 2, 3]
    assert HybridIndex.from_parts([], emb).dense("anything", 3) == []
