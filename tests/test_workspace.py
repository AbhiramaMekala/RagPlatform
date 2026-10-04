"""Sample library (prebuilt index) and the visitor sandbox, with fake models."""
import pytest

pytest.importorskip("langchain_text_splitters")

from app.config import Settings
from app.workspace import SampleLibrary, Sessions
from tests.fakes import FakeEmbedder

SID = "visitor-1234abcd"
UPLOAD = (
    b"Zephyr Robotics onboarding guide.\n\nEvery new engineer at Zephyr Robotics receives a titanium badge "
    b"on day one. The badge opens the Hangar 9 workshop, which is open from 7 AM to 9 PM.\n\n"
    b"The onboarding buddy for the firmware team is Marisol Okonkwo."
)


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


class CountingEmbedder(FakeEmbedder):
    calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


@pytest.fixture
def settings(tmp_path):
    return Settings(index_dir=tmp_path / "index", max_uploads=2, session_ttl_minutes=30, max_sessions=3)


@pytest.fixture
def sessions(settings):
    emb = FakeEmbedder()
    s = Sessions(SampleLibrary.load(settings, emb), emb, settings, clock=Clock())
    return s


def sources(index, query):
    return {h.chunk.source for h in index.dense(query, 50)} | {h.chunk.source for h in index.keyword(query, 50)}


def test_prebuilt_index_is_reused_and_invalidated(settings):
    emb = CountingEmbedder()
    first = SampleLibrary.load(settings, emb)
    built = emb.calls
    assert built == len(first.docs) and (settings.index_dir / "vectors.npy").exists()
    second = SampleLibrary.load(settings, emb)
    assert emb.calls == built  # loaded from disk: no embedding at startup
    assert [c.text for c in second.index.chunks] == [c.text for c in first.index.chunks]
    settings.chunk_size = 500  # different settings -> stale index -> rebuilt
    SampleLibrary.load(settings, emb)
    assert emb.calls == 2 * built


def test_new_visitor_shares_the_sample_index(sessions):
    st = sessions.state(SID)
    assert st["custom"] is False and len(st["documents"]) >= 5
    index, private = sessions.index(SID)
    assert private is False and index is sessions.library.index
    assert len(sessions.document(SID, st["documents"][0]["id"]).doc.text) > 200


def test_remove_only_affects_that_visitor(sessions):
    first = sessions.state(None)["documents"][0]["id"]
    st = sessions.remove(SID, first)
    assert st["custom"] is True and first not in [d["id"] for d in st["documents"]]
    index, private = sessions.index(SID)
    assert private and first not in sources(index, "Quillfeather policy incident roadmap overview")
    assert first in [d["id"] for d in sessions.state("someone-else-123")["documents"]]


def test_upload_is_searchable_and_viewable(sessions):
    st = sessions.add(SID, "zephyr-onboarding.txt", UPLOAD)
    up = [d for d in st["documents"] if d["origin"] == "uploaded"]
    assert len(up) == 1 and up[0]["chunks"] >= 1 and st["limits"]["uploads_left"] == 1
    index, _ = sessions.index(SID)
    assert index.keyword("titanium badge Hangar", 3)[0].chunk.source == up[0]["id"]
    assert "Marisol" in sessions.document(SID, up[0]["id"]).doc.text


def test_reset_expiry_and_eviction(sessions):
    sessions.add(SID, "a.txt", UPLOAD)
    assert sessions.reset(SID)["custom"] is False
    sessions.add(SID, "a.txt", UPLOAD)
    sessions.clock.t += 31 * 60
    assert sessions.state(SID)["custom"] is False  # idle too long
    for i in range(4):
        sessions.clock.t += 1
        sessions.remove(f"visitor-{i:08d}", sessions.state(None)["documents"][0]["id"])
    assert sessions.state("visitor-00000000")["custom"] is False and len(sessions) == 3


def test_upload_limits_and_bad_ids(sessions):
    with pytest.raises(ValueError, match="Unsupported"):
        sessions.add(SID, "virus.exe", b"MZ" * 100)
    sessions.add(SID, "a.txt", UPLOAD)
    sessions.add(SID, "b.txt", UPLOAD)
    with pytest.raises(ValueError, match="limit"):
        sessions.add(SID, "c.txt", UPLOAD)
    with pytest.raises(ValueError):
        sessions.add("../../etc", "a.txt", UPLOAD)
    assert sessions.state("bad id!")["custom"] is False
