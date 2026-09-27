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
    openai_api_key: str = _env("OPENAI_API_KEY", "")

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
