"""API test: swaps the real (model-loading) pipeline for the fake one."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import app.api as api  # noqa: E402
from tests.test_pipeline import make_pipeline  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "build_pipeline", lambda settings: make_pipeline())
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_and_metrics(client):
    r = client.post("/ask", json={"question": "What is the walrus operator?"})
    assert r.status_code == 200 and r.json()["status"] == "answered"
    assert client.get("/metrics").json()["requests"] == 1
    assert len(client.get("/traces").json()) == 1


def test_demo_page_served(client):
    assert "RAG Platform" in client.get("/").text
