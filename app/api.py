"""HTTP API + demo UI.   Run locally:  uvicorn app.api:app --reload   ->  http://localhost:8000

All endpoints live under /api. A visitor's browser tab creates a random session id; until the visitor
changes the documents, the session id simply maps to the shared sample index.
"""
import base64
import binascii
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings
from .protect import AnswerCache, RateLimiter
from .services import build_services

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = build_services(settings)
    s = svc.settings
    state.update(svc=svc, cache=AnswerCache(s.cache_size, s.cache_minutes * 60),
                 ask_limit=RateLimiter(s.asks_per_minute, 60), upload_limit=RateLimiter(s.uploads_per_hour, 3600))
    yield
    if s.langfuse_enabled:
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:  # best effort on shutdown
            pass


app = FastAPI(title="Production RAG Platform", version="2.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=2000, examples=["What is Project Kestrel and when does it launch?"])
    session_id: str | None = Field(None, max_length=64, description="Visitor session; omit to use the sample documents")


class UploadRequest(BaseModel):
    filename: str = Field(..., max_length=200)
    content_base64: str = Field(..., description="The file's bytes, base64-encoded")


def svc():
    if "svc" not in state:
        raise HTTPException(503, "Starting up, try again in a few seconds.")
    return state["svc"]


def client_ip(request: Request) -> str:
    # Cloud Run puts the visitor's address first in X-Forwarded-For
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def limit(name: str, request: Request, message: str):
    if not state[name].allow(client_ip(request)):
        raise HTTPException(429, message)


# ---------- UI + health ----------

@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    return {"status": "ok" if "svc" in state else "starting"}


# ---------- ask ----------

@app.post("/api/ask")
async def ask(req: AskRequest, request: Request, x_openai_key: str | None = Header(None, alias="X-OpenAI-Key")):
    s = svc()
    limit("ask_limit", request, "Too many questions in a minute. Please wait a moment and try again.")
    index, private = s.sessions.index(req.session_id)
    cacheable = not private and not x_openai_key
    if cacheable and (hit := state["cache"].get(req.question)):
        s.pipeline.traces.cache_hits += 1
        return {**hit, "cached": True, "workspace": "samples"}

    generator = s.router.for_request(x_openai_key)  # the visitor's key is used for this request only, never stored
    result = await run_in_threadpool(  # CPU-bound (embeddings, reranking): keep the event loop free
        s.pipeline.ask, req.question, index, generator,
        {"session": req.session_id} if private else None,  # private traces are visible only to that session
        not private,  # ...and are never logged, written or exported
    )
    if cacheable:
        state["cache"].put(req.question, result)
    return {**result, "cached": False, "workspace": "custom" if private else "samples"}


# ---------- documents (the visitor sandbox) ----------

@app.get("/api/state")
def get_state(session: str | None = None):
    """Documents this visitor searches, sandbox limits and the available LLMs - everything the UI needs."""
    s = svc()
    return {**s.sessions.state(session), "llm": s.router.info(), "dense_backend": s.sessions.library.index.backend}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str, session: str | None = None):
    try:
        d = svc().sessions.document(session, doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found (it may have been removed, or your session reset).") from None
    return {**d.summary(), "text": d.doc.text}


@app.post("/api/sessions/{session_id}/documents")
async def upload_document(session_id: str, req: UploadRequest, request: Request):
    s = svc()
    limit("upload_limit", request, "Upload limit reached for now. Please try again later.")
    if len(req.content_base64) > s.settings.max_upload_bytes * 4 // 3 + 8:
        raise HTTPException(413, f"File too large (max {s.settings.max_upload_bytes / 1e6:.1f} MB).")
    try:
        data = base64.b64decode(req.content_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "File content must be base64-encoded.") from None
    try:
        return await run_in_threadpool(s.sessions.add, session_id, req.filename, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/sessions/{session_id}/documents/{doc_id}")
def remove_document(session_id: str, doc_id: str):
    try:
        return svc().sessions.remove(session_id, doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found.") from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str):
    try:
        return svc().sessions.reset(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------- observability ----------

@app.get("/api/metrics")
def metrics():
    s = svc()
    return {**s.pipeline.traces.metrics(), "active_sessions": len(s.sessions)}


@app.get("/api/traces")
def traces(limit: int = 20, session: str | None = None):
    """Recent traces. Private-upload traces are only returned to their own session."""
    recent = list(svc().pipeline.traces.recent)
    visible = [t for t in recent if not t.get("session") or (session and t.get("session") == session)]
    return visible[-max(1, min(limit, 100)):][::-1]
