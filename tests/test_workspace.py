"""Visitor sandbox: view, remove, upload, reset, expiry - with fake models (no downloads)."""
import pytest

pytest.importorskip("langchain_text_splitters")

from app.config import Settings  # noqa: E402
from app.workspace import SampleLibrary, SessionManager  # noqa: E402
from tests.fakes import FakeEmbedder  # noqa: E402

SID = "visitor-1234abcd"
UPLOAD = (
    b"Zephyr Robotics onboarding guide.\n\nEvery new engineer at Zephyr Robotics receives a titanium badge "
    b"on day one. The badge opens the Hangar 9 workshop, which is open from 7 AM to 9 PM.\n\n"
    b"The onboarding buddy for the firmware team is Marisol Okonkwo."
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def mgr():
    s = Settings(max_uploads=2, session_ttl_minutes=30, max_sessions=3)
    emb = FakeEmbedder()
    clock = Clock()
    m = SessionManager(SampleLibrary.load(s, emb), emb, s, clock=clock)
    m.clock_ref = clock
    return m


def sources(m, sid, query):
    r = m.retriever(sid)
    return {h.chunk.source for h in r.dense(query, 50)} | {h.chunk.source for h in r.keyword(query, 50)}


def test_new_visitor_sees_original_samples(mgr):
    st = mgr.state(SID)
    assert st["custom"] is False
    assert len(st["documents"]) >= 5 and all(d["origin"] == "sample" for d in st["documents"])
    assert mgr.retriever(SID) is None  # uses the shared sample index
    doc = mgr.document(SID, st["documents"][0]["id"])
    assert len(doc.text) > 200


def test_remove_only_affects_that_visitor(mgr):
    first = mgr.state(None)["documents"][0]["id"]
    st = mgr.remove(SID, first)
    assert st["custom"] is True and first not in [d["id"] for d in st["documents"]]
    assert first not in sources(mgr, SID, "Quillfeather policy incident roadmap")
    # someone else still sees everything
    assert first in [d["id"] for d in mgr.state("other-visitor-999")["documents"]]


def test_upload_is_searchable(mgr):
    st = mgr.add(SID, "zephyr-onboarding.txt", UPLOAD)
    up = [d for d in st["documents"] if d["origin"] == "uploaded"]
    assert len(up) == 1 and up[0]["name"] == "zephyr-onboarding.txt" and up[0]["chunks"] >= 1
    hits = mgr.retriever(SID).keyword("titanium badge Hangar", 3)
    assert hits and hits[0].chunk.source == up[0]["id"]
    assert "Marisol" in mgr.document(SID, up[0]["id"]).text


def test_remove_all_samples_then_upload(mgr):
    for d in mgr.state(None)["documents"]:
        mgr.remove(SID, d["id"])
    assert mgr.state(SID)["documents"] == []
    mgr.add(SID, "notes.md", UPLOAD)
    assert sources(mgr, SID, "Quillfeather badge") == {mgr.state(SID)["documents"][0]["id"]}


def test_reset_restores_samples(mgr):
    mgr.add(SID, "a.txt", UPLOAD)
    st = mgr.reset(SID)
    assert st["custom"] is False and all(d["origin"] == "sample" for d in st["documents"])
    assert mgr.retriever(SID) is None


def test_idle_session_expires(mgr):
    mgr.add(SID, "a.txt", UPLOAD)
    mgr.clock_ref.t += 31 * 60
    assert mgr.state(SID)["custom"] is False


def test_oldest_session_dropped_beyond_limit(mgr):
    for i in range(4):
        mgr.clock_ref.t += 1
        mgr.remove(f"visitor-{i:08d}", mgr.state(None)["documents"][0]["id"])
    assert mgr.state("visitor-00000000")["custom"] is False  # evicted
    assert mgr.state("visitor-00000003")["custom"] is True


def test_upload_limits(mgr):
    with pytest.raises(ValueError, match="Unsupported"):
        mgr.add(SID, "virus.exe", b"MZ" * 100)
    with pytest.raises(ValueError, match="readable text"):
        mgr.add(SID, "tiny.txt", b"hi")
    mgr.add(SID, "a.txt", UPLOAD)
    mgr.add(SID, "b.txt", UPLOAD)
    with pytest.raises(ValueError, match="limit"):
        mgr.add(SID, "c.txt", UPLOAD)


def test_invalid_session_id_rejected(mgr):
    with pytest.raises(ValueError):
        mgr.add("../../etc", "a.txt", UPLOAD)
    assert mgr.state("bad id!")["custom"] is False
