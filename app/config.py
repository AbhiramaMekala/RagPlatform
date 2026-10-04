"""All settings in one place. Every value can be overridden with an environment variable (or a .env file)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # load .env for local development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return type(default)(value)


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]


@dataclass
class Settings:
    # --- data ---
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCS_DIR", str(ROOT / "data" / "docs"))))
    # The sample index (chunks + vectors) is built once - in the Docker image at build time - and loaded at startup.
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "data" / "index"))))
    chunk_size: int = _env("CHUNK_SIZE", 800)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 120)

    # --- models (ONNX via fastembed, CPU only) ---
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    onnx_threads: int = _env("ONNX_THREADS", 0)  # 0 = let ONNX Runtime decide

    # --- vector store ---
    # Empty = in-memory numpy index (the serverless demo: nothing to run, starts in milliseconds).
    # Set to a Qdrant server URL (docker-compose does) to serve the shared sample index from Qdrant.
    qdrant_url: str = _env("QDRANT_URL", "")
    collection: str = _env("COLLECTION", "quillfeather")

    # --- retrieval ---
    candidates: int = _env("CANDIDATES", 20)  # per retriever, before fusion
    top_k: int = _env("TOP_K", 4)  # chunks given to the LLM after reranking
    min_rerank_score: float = _env("MIN_RERANK_SCORE", -3.0)  # below this = "not in the documents"

    # --- guardrails ---
    max_query_chars: int = _env("MAX_QUERY_CHARS", 500)
    grounding_similarity: float = _env("GROUNDING_SIMILARITY", 0.78)
    min_groundedness: float = _env("MIN_GROUNDEDNESS", 0.7)

    # --- LLMs: the visitor chooses (see llm.py). Anthropic / OpenAI use the visitor's own key; Gemini uses
    # the site's free keys and runs only when picked; with nothing usable the answer is quoted (no LLM). ---
    anthropic_models: list = field(default_factory=lambda: _list("ANTHROPIC_MODELS")
                                   or ["claude-haiku-4-5-20251001", "claude-sonnet-5-5", "claude-opus-5-5"])
    openai_models: list = field(default_factory=lambda: _list("OPENAI_MODELS") or ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"])
    gemini_api_keys: list = field(default_factory=lambda: _list("GEMINI_API_KEYS"))
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_base_url: str = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm_timeout: float = _env("LLM_TIMEOUT", 25.0)

    # --- visitor sandbox ---
    session_ttl_minutes: int = _env("SESSION_TTL_MINUTES", 30)
    max_sessions: int = _env("MAX_SESSIONS", 50)
    max_uploads: int = _env("MAX_UPLOADS", 5)
    max_upload_bytes: int = _env("MAX_UPLOAD_BYTES", 2_000_000)
    max_doc_chars: int = _env("MAX_DOC_CHARS", 60_000)

    # --- protecting a public deployment ---
    asks_per_minute: int = _env("ASKS_PER_MINUTE", 12)  # per visitor IP
    uploads_per_hour: int = _env("UPLOADS_PER_HOUR", 20)
    cache_minutes: int = _env("CACHE_MINUTES", 360)  # identical questions about the samples are answered from cache
    cache_size: int = _env("CACHE_SIZE", 300)

    # --- observability ---
    # Traces are logged to stdout as JSON (Cloud Run sends them to Cloud Logging). A file is optional:
    # on Cloud Run the filesystem lives in RAM, so an ever-growing file would leak memory.
    traces_file: str = _env("TRACES_FILE", "")
    langfuse_enabled: bool = field(default_factory=lambda: bool(os.getenv("LANGFUSE_PUBLIC_KEY")))


settings = Settings()
