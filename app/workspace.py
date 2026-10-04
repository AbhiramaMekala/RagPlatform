"""Visitor sandbox for the public demo.

Everyone starts with the same sample documents (the Quillfeather Labs handbook). A visitor can open
and read them, remove some, and upload their own files. Those changes live in a private, in-memory
*session* that only that visitor's browser knows the id of:

  - reloading the page starts a new session -> back to the original samples
  - "Reset" deletes the session
  - idle sessions expire after SESSION_TTL_MINUTES, and the oldest is dropped beyond MAX_SESSIONS

The shared sample index (Qdrant) is never modified. A changed session gets its own small
in-memory index (MemoryStore + BM25) built from pre-computed chunk vectors, so removing a
document is instant and only uploaded files need embedding.
"""
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25Index
from .documents import extract_text, safe_filename, suffix
from .ingest import SUPPORTED, chunk_documents, clean_text, extract_title
from .models import Chunk
from .retriever import HybridRetriever
from .vectorstore import MemoryStore

SESSION_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


@dataclass
class Document:
    id: str  # also used as Chunk.source, so answer citations can link back to the document
    name: str  # file name shown to the visitor
    title: str
    text: str  # full text shown in the document viewer
    origin: str  # "sample" or "uploaded"
    truncated: bool = False
    chunks: list[Chunk] = field(default_factory=list, repr=False)
    vectors: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "title": self.title, "origin": self.origin,
                "chars": len(self.text), "chunks": len(self.chunks), "truncated": self.truncated}


def index_document(doc_id: str, title: str, text: str, embedder, settings) -> tuple[list[Chunk], np.ndarray]:
    """Clean -> chunk -> embed, exactly like the main ingestion pipeline."""
    chunks = chunk_documents([(doc_id, title, clean_text(text))], settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return [], np.zeros((0, 1), dtype=np.float32)
    return chunks, np.asarray(embedder.embed_documents([f"{c.title}\n{c.text}" for c in chunks]), dtype=np.float32)


class SampleLibrary:
    """The original documents every visitor starts with (pre-chunked and pre-embedded once at startup)."""

    def __init__(self, docs: list[Document]):
        self.docs = docs

    @classmethod
    def load(cls, settings, embedder) -> "SampleLibrary":
        docs = []
        for path in sorted(settings.docs_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED:
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
            title = extract_title(raw, path.stem)
            chunks, vectors = index_document(path.name, title, raw, embedder, settings)
            docs.append(Document(path.name, path.name, title, raw, "sample", chunks=chunks, vectors=vectors))
        return cls(docs)


def build_retriever(docs: list[Document], embedder) -> HybridRetriever:
    chunks, vectors = [], []
    for d in docs:
        for c, v in zip(d.chunks, d.vectors if d.vectors is not None else []):
            chunks.append(Chunk(len(chunks), c.text, c.source, c.title))  # ids must be unique per index
            vectors.append(v)
    store = MemoryStore(chunks, np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32))
    return HybridRetriever(embedder, store, BM25Index(chunks))


@dataclass
class Session:
    docs: "OrderedDict[str, Document]"
    retriever: HybridRetriever
    last_used: float
    uploads: int = 0


class SessionManager:
    def __init__(self, library: SampleLibrary, embedder, settings, clock=time.monotonic):
        self.library, self.embedder, self.s, self.clock = library, embedder, settings, clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    # ---------- lookups ----------
    def documents(self, sid: str | None) -> list[Document]:
        s = self._get(sid)
        return list(s.docs.values()) if s else list(self.library.docs)

    def document(self, sid: str | None, doc_id: str) -> Document:
        for d in self.documents(sid):
            if d.id == doc_id:
                return d
        raise KeyError(doc_id)

    def retriever(self, sid: str | None):
        """The visitor's private retriever, or None = use the shared sample index."""
        s = self._get(sid)
        return s.retriever if s else None

    def state(self, sid: str | None) -> dict:
        s = self._get(sid)
        docs = list(s.docs.values()) if s else self.library.docs
        return {
            "custom": s is not None,
            "documents": [d.summary() for d in docs],
            "limits": {"max_uploads": self.s.max_uploads, "max_upload_mb": round(self.s.max_upload_bytes / 1e6, 1),
                       "max_doc_chars": self.s.max_doc_chars, "ttl_minutes": self.s.session_ttl_minutes,
                       "uploads_left": self.s.max_uploads - (s.uploads if s else 0)},
        }

    # ---------- changes (each one copies the samples into a private session first) ----------
    def remove(self, sid: str, doc_id: str) -> dict:
        with self._lock:
            s = self._get_or_create(sid)
            if doc_id not in s.docs:
                raise KeyError(doc_id)
            del s.docs[doc_id]
            s.retriever = build_retriever(list(s.docs.values()), self.embedder)
        return self.state(sid)

    def add(self, sid: str, filename: str, data: bytes) -> dict:
        self._check_id(sid)
        name = safe_filename(filename)
        if len(data) > self.s.max_upload_bytes:
            raise ValueError(f"File too large (max {self.s.max_upload_bytes / 1e6:.1f} MB).")
        current = self._get(sid)
        if current and current.uploads >= self.s.max_uploads:
            raise ValueError(f"Upload limit reached ({self.s.max_uploads} files per session). Reset to start over.")
        text = extract_text(name, data)
        truncated = len(text) > self.s.max_doc_chars
        text = text[: self.s.max_doc_chars]
        title = name[: -len(suffix(name))] if suffix(name) else name

        # embedding is the slow part, so do it outside the lock
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
        doc_id = f"upload-{uuid.uuid4().hex[:8]}-{slug}"
        chunks, vectors = index_document(doc_id, title, text, self.embedder, self.s)
        if not chunks:
            raise ValueError("That file has too little text to index.")

        with self._lock:
            s = self._get_or_create(sid)
            if s.uploads >= self.s.max_uploads:
                raise ValueError(f"Upload limit reached ({self.s.max_uploads} files per session). Reset to start over.")
            s.uploads += 1
            s.docs[doc_id] = Document(doc_id, name, title, text, "uploaded", truncated, chunks, vectors)
            s.retriever = build_retriever(list(s.docs.values()), self.embedder)
        return self.state(sid)

    def reset(self, sid: str) -> dict:
        self._check_id(sid)
        with self._lock:
            self._sessions.pop(sid, None)
        return self.state(None)

    # ---------- internals ----------
    def _check_id(self, sid):
        if not sid or not SESSION_ID.match(sid):
            raise ValueError("Invalid session id.")

    def _expire(self, now: float):
        ttl = self.s.session_ttl_minutes * 60
        for key in [k for k, v in self._sessions.items() if now - v.last_used > ttl]:
            del self._sessions[key]

    def _get(self, sid: str | None) -> Session | None:
        if not sid or not SESSION_ID.match(sid):
            return None
        with self._lock:
            now = self.clock()
            self._expire(now)
            s = self._sessions.get(sid)
            if s:
                s.last_used = now
                self._sessions.move_to_end(sid)
            return s

    def _get_or_create(self, sid: str) -> Session:
        """Caller must hold the lock."""
        self._check_id(sid)
        now = self.clock()
        self._expire(now)
        s = self._sessions.get(sid)
        if s is None:
            while len(self._sessions) >= self.s.max_sessions:
                self._sessions.popitem(last=False)  # drop the least recently used
            docs = OrderedDict((d.id, d) for d in self.library.docs)
            s = Session(docs, build_retriever(list(docs.values()), self.embedder), now)
            self._sessions[sid] = s
        s.last_used = now
        self._sessions.move_to_end(sid)
        return s
