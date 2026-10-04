"""FastAPI service.  Run:  uvicorn app.api:app --reload   then open http://localhost:8000"""
import base64
import binascii
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .llm import LLMRouter
    from .workspace import SampleLibrary, SessionManager

    pipeline = build_pipeline(settings)  # loads models + indexes documents on first run
    library = SampleLibrary.load(pipeline.s, pipeline.embedder)  # the docs every visitor starts with
    state["pipeline"] = pipeline
    state["sessions"] = SessionManager(library, pipeline.embedder, pipeline.s)
    state["router"] = getattr(pipeline, "router", None) or LLMRouter(pipeline.s, pipeline.embedder)
    yield
    if settings.langfuse_enabled:
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:
            pass


app = FastAPI(title="Production RAG Platform", version="1.1", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., examples=["What is Project Kestrel and when does it launch?"])
    session_id: str | None = Field(None, description="Visitor session; omit to use the sample documents")


class UploadRequest(BaseModel):
    filename: str = Field(..., max_length=200)
    content_base64: str = Field(..., description="The file's bytes, base64-encoded")


def _ready():
    if "pipeline" not in state:
        raise HTTPException(503, "Pipeline is still loading")
    return state["pipeline"], state["sessions"]


def _bad_request(exc: Exception):
    raise HTTPException(400, str(exc)) from exc


@app.get("/")
def demo_page():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in state else "starting"}


@app.get("/config")
def config():
    """What the demo UI needs to explain which model will answer."""
    _ready()
    return state["router"].info()


@app.post("/ask")
async def ask(req: AskRequest, x_openai_key: str | None = Header(None, alias="X-OpenAI-Key")):
    pipeline, sessions = _ready()
    retriever = sessions.retriever(req.session_id)  # None = shared sample index
    custom = retriever is not None
    generator = state["router"].for_request(x_openai_key)  # the visitor's key is used for this request only
    # the pipeline is CPU-bound, so run it in a worker thread to keep the server responsive
    result = await run_in_threadpool(
        pipeline.ask, req.question, retriever, generator,
        {"session": req.session_id} if custom else None,  # private-document traces are only visible to that visitor
        not custom,  # ...and are never written to disk or exported
    )
    result["workspace"] = "custom" if custom else "samples"
    return result


# ---------- documents (the visitor sandbox) ----------

@app.get("/documents")
def list_documents(session: str | None = None):
    _, sessions = _ready()
    return sessions.state(session)


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, session: str | None = None):
    _, sessions = _ready()
    try:
        d = sessions.document(session, doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found (it may have been removed, or your session reset).") from None
    return {**d.summary(), "text": d.text}


@app.post("/sessions/{session_id}/documents")
async def upload_document(session_id: str, req: UploadRequest):
    _, sessions = _ready()
    try:
        data = base64.b64decode(req.content_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "File content must be base64-encoded.") from None
    try:
        return await run_in_threadpool(sessions.add, session_id, req.filename, data)
    except ValueError as exc:
        _bad_request(exc)


@app.delete("/sessions/{session_id}/documents/{doc_id}")
def remove_document(session_id: str, doc_id: str):
    _, sessions = _ready()
    try:
        return sessions.remove(session_id, doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found.") from None
    except ValueError as exc:
        _bad_request(exc)


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str):
    _, sessions = _ready()
    try:
        return sessions.reset(session_id)
    except ValueError as exc:
        _bad_request(exc)


# ---------- observability ----------

@app.get("/metrics")
def metrics():
    return state["pipeline"].traces.metrics()


@app.get("/traces")
def traces(limit: int = 20, session: str | None = None):
    """Recent traces. Without `session`: only questions about the shared sample documents.
    With `session`: that visitor's own questions too (nobody else can see them)."""
    recent = list(state["pipeline"].traces.recent)
    visible = [t for t in recent if not t.get("session") or (session and t.get("session") == session)]
    return visible[-limit:][::-1]
