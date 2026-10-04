"""All settings in one place. Every value can be overridden with an environment variable (or .env file)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # load .env if python-dotenv is installed
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


@dataclass
class Settings:
    # --- data ---
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCS_DIR", str(ROOT / "data" / "docs"))))
    traces_file: Path = field(default_factory=lambda: Path(_env("TRACES_FILE", str(ROOT / "data" / "traces.jsonl"))))
    chunk_size: int = _env("CHUNK_SIZE", 800)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 120)

    # --- vector store (Qdrant) ---
    # Empty QDRANT_URL = embedded local mode (no server needed). Docker sets it to http://qdrant:6333
    qdrant_url: str = _env("QDRANT_URL", "")
    qdrant_path: str = _env("QDRANT_PATH", str(ROOT / "data" / "qdrant"))
    collection: str = _env("COLLECTION", "quillfeather")

    # --- models (run locally via fastembed / ONNX, no GPU needed) ---
    embed_model: str = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    llm_model: str = _env("LLM_MODEL", "gpt-4o-mini")
    # Optional server-side OpenAI key. Leave empty on a public demo so you never pay for visitors' questions.
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    # Free backup LLM: comma-separated Gemini API keys. Requests rotate across them and a failing key is
    # skipped for a while, so one exhausted free-tier key doesn't break the demo.
    gemini_api_keys: list = field(default_factory=lambda: [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()])
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_base_url: str = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm_timeout: float = _env("LLM_TIMEOUT", 25.0)

    # --- visitor sandbox (per-visitor document sets that reset to the samples) ---
    session_ttl_minutes: int = _env("SESSION_TTL_MINUTES", 30)  # idle sessions reset after this
    max_sessions: int = _env("MAX_SESSIONS", 50)  # oldest idle session is dropped beyond this
    max_uploads: int = _env("MAX_UPLOADS", 5)  # uploaded files per session
    max_upload_bytes: int = _env("MAX_UPLOAD_BYTES", 2_000_000)  # per file
    max_doc_chars: int = _env("MAX_DOC_CHARS", 60_000)  # extracted text per file (longer files are truncated)

    # --- retrieval ---
    candidates: int = _env("CANDIDATES", 20)  # how many each retriever returns before fusion
    top_k: int = _env("TOP_K", 4)  # chunks passed to the LLM after reranking
    min_rerank_score: float = _env("MIN_RERANK_SCORE", -3.0)  # below this = "not in the documents"

    # --- guardrails ---
    max_query_chars: int = _env("MAX_QUERY_CHARS", 500)
    grounding_similarity: float = _env("GROUNDING_SIMILARITY", 0.78)  # sentence counts as supported above this
    min_groundedness: float = _env("MIN_GROUNDEDNESS", 0.7)  # answer flagged as hallucination below this

    # --- observability ---
    langfuse_enabled: bool = field(default_factory=lambda: bool(os.getenv("LANGFUSE_PUBLIC_KEY")))


settings = Settings()
