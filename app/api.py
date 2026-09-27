"""FastAPI service.  Run:  uvicorn app.api:app --reload   then open http://localhost:8000"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
    state["pipeline"] = build_pipeline(settings)  # loads models + indexes documents on first run
    yield
    if settings.langfuse_enabled:
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:
            pass


app = FastAPI(title="Production RAG Platform", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., examples=["What is Project Kestrel and when does it launch?"])


@app.get("/")
def demo_page():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok" if "pipeline" in state else "starting"}


@app.post("/ask")
async def ask(req: AskRequest):
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Pipeline is still loading")
    # the pipeline is CPU-bound, so run it in a worker thread to keep the server responsive
    return await run_in_threadpool(pipeline.ask, req.question)


@app.get("/metrics")
def metrics():
    return state["pipeline"].traces.metrics()


@app.get("/traces")
def traces(limit: int = 20):
    return list(state["pipeline"].traces.recent)[-limit:][::-1]
