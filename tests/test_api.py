"""API tests: swap the real (model-loading) pipeline for the fake one."""
import base64

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langchain_text_splitters")
from fastapi.testclient import TestClient  # noqa: E402

import app.api as api  # noqa: E402
from tests.test_pipeline import make_pipeline  # noqa: E402

SID = "test-session-0001"
UPLOAD = (
    "Zephyr Robotics handbook.\n\nEvery new engineer at Zephyr Robotics receives a titanium badge on day one. "
    "The titanium badge opens the Hangar 9 workshop, which is open from 7 AM to 9 PM every weekday."
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "build_pipeline", lambda settings: make_pipeline())
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_and_metrics(client):
    r = client.post("/ask", json={"question": "What is the walrus operator?"})
    assert r.status_code == 200 and r.json()["status"] == "answered" and r.json()["workspace"] == "samples"
    assert client.get("/metrics").json()["requests"] == 1
    assert len(client.get("/traces").json()) == 1


def test_demo_page_served(client):
    assert "RAG Platform" in client.get("/").text


def test_documents_are_viewable(client):
    docs = client.get("/documents").json()["documents"]
    assert len(docs) >= 5
    d = client.get(f"/documents/{docs[0]['id']}").json()
    assert d["title"] and len(d["text"]) > 100
    assert client.get("/documents/nope.md").status_code == 404


def test_upload_ask_and_reset(client):
    up = client.post(f"/sessions/{SID}/documents",
                     json={"filename": "zephyr.txt", "content_base64": base64.b64encode(UPLOAD.encode()).decode()})
    assert up.status_code == 200 and up.json()["custom"] is True
    new_id = [d for d in up.json()["documents"] if d["origin"] == "uploaded"][0]["id"]

    r = client.post("/ask", json={"question": "What does the titanium badge open at Zephyr?", "session_id": SID}).json()
    assert r["workspace"] == "custom" and r["status"] == "answered"
    assert new_id in [s["source"] for s in r["sources"]]

    # private-document traces are hidden from everyone except that visitor
    assert all(t.get("session") != SID for t in client.get("/traces").json())
    assert any(t.get("session") == SID for t in client.get(f"/traces?session={SID}").json())

    assert client.delete(f"/sessions/{SID}/documents/{new_id}").status_code == 200
    reset = client.post(f"/sessions/{SID}/reset").json()
    assert reset["custom"] is False
    assert client.get("/documents", params={"session": SID}).json()["custom"] is False


def test_bad_uploads(client):
    bad = client.post(f"/sessions/{SID}/documents", json={"filename": "x.exe", "content_base64": base64.b64encode(b"MZ" * 50).decode()})
    assert bad.status_code == 400 and "Unsupported" in bad.json()["detail"]
    assert client.post(f"/sessions/{SID}/documents", json={"filename": "x.txt", "content_base64": "%%%"}).status_code == 400
    assert client.post("/sessions/bad id/reset").status_code in (400, 404)


def test_visitor_key_header_accepted(client):
    r = client.post("/ask", json={"question": "What is the walrus operator?"}, headers={"X-OpenAI-Key": "not-a-key"})
    assert r.status_code == 200
    assert "doesn't look valid" in " ".join(r.json()["trace"].get("llm_notes", []))
