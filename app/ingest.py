"""Ingestion: load documents -> clean -> chunk (LangChain splitter) -> embed -> store in Qdrant.

Run manually:  python -m app.ingest
(The API also runs this automatically on first start if the collection is empty.)
"""
import re
from pathlib import Path

from .models import Chunk

SUPPORTED = {".rst", ".md", ".txt"}


def load_documents(docs_dir: Path) -> list[tuple[str, str, str]]:
    """Returns (file_name, title, cleaned_text) for every supported file."""
    docs = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            docs.append((path.name, extract_title(raw, path.stem), clean_text(raw)))
    return docs


FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def extract_title(raw: str, fallback: str) -> str:
    """Title from Markdown front matter (`title: Time off`) or a `Title:` header line; else the file name."""
    m = re.search(r"^title:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
    if not m:
        return fallback.replace("-", " ").title()
    return m.group(1).strip().strip("\"'").replace("``", "")


def clean_text(raw: str) -> str:
    """Strip Markdown / reStructuredText markup so chunks read like plain prose."""
    text = FRONT_MATTER.sub("", raw)  # drop the --- title/sidebar --- block
    text = re.sub(r"^import .*$", "", text, flags=re.MULTILINE)  # MDX imports
    text = re.sub(r"<[^>\n]+/?>", "", text)  # HTML / JSX components like <CompensationCalculator />
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [link text](url) -> link text
    text = re.sub(r":[\w-]+:`([^`<]*?)(?:\s*<[^>]*>)?`", r"\1", text)  # rst :role:`x` -> x
    text = re.sub(r"`{1,2}([^`\n]+)`{1,2}_?", r"\1", text)  # `code` -> code
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*([=\-~^#*+])\1{3,}\s*", line):  # rst heading underlines / horizontal rules
            continue
        if re.match(r"\s*\.\. (\w[\w-]*::|_|\[)", line):  # rst directives
            continue
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def chunk_documents(docs, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for source, title, text in docs:
        for piece in splitter.split_text(text):
            if len(piece) > 80:  # drop tiny fragments
                chunks.append(Chunk(id=len(chunks), text=piece, source=source, title=title))
    return chunks


def ingest(settings, embedder, store) -> int:
    docs = load_documents(settings.docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    print(f"[ingest] {len(docs)} documents -> {len(chunks)} chunks. Embedding (first run takes ~1 min)...")
    vectors = embedder.embed_documents([f"{c.title}\n{c.text}" for c in chunks])
    store.reset(embedder.dim)
    store.add(chunks, vectors)
    print(f"[ingest] stored {len(chunks)} chunks in Qdrant collection '{settings.collection}'")
    return len(chunks)


if __name__ == "__main__":
    from .config import settings
    from .embeddings import Embedder
    from .vectorstore import QdrantStore

    ingest(settings, Embedder(settings.embed_model), QdrantStore(settings.qdrant_url, settings.qdrant_path, settings.collection))
