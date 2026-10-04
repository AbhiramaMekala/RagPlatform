"""Documents and who sees which ones.

SampleLibrary - the original documents. Chunked and embedded ONCE (at Docker build time) and saved to
                data/index/, so a cold start just loads two small files instead of running the embedder.
Sessions      - the visitor sandbox. A visitor who removes or uploads a file gets a private copy of the
                library (copy-on-write) with its own small in-memory index. Only that browser tab knows
                the session id, so: reload = new session = original samples again. Idle sessions expire,
                and the least recently used is dropped beyond MAX_SESSIONS, so memory stays bounded.
"""
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .index import HybridIndex, QdrantDense
from .models import Chunk, Document
from .text import TEXT_VERSION, chunk_text, extract_upload_text, load_samples, safe_filename, suffix

log = logging.getLogger("rag.workspace")
SESSION_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


@dataclass
class IndexedDoc:
    doc: Document
    chunks: list[Chunk]
    vectors: np.ndarray

    def summary(self) -> dict:
        d = self.doc
        return {"id": d.id, "name": d.name, "title": d.title, "origin": d.origin,
                "chars": len(d.text), "chunks": len(self.chunks), "truncated": d.truncated}


def embed_doc(doc: Document, embedder, settings) -> IndexedDoc:
    chunks = chunk_text(doc, settings.chunk_size, settings.chunk_overlap)
    return IndexedDoc(doc, chunks, embedder.embed_documents([f"{c.title}\n{c.text}" for c in chunks]))


# ---------- the shared sample documents ----------

class SampleLibrary:
    def __init__(self, docs: list[IndexedDoc], index: HybridIndex):
        self.docs, self.index = docs, index

    @classmethod
    def load(cls, settings, embedder) -> "SampleLibrary":
        docs = load_samples(settings.docs_dir)
        indexed = _load_prebuilt(settings, docs)
        if indexed is None:
            log.info("No prebuilt index for these documents - embedding %d documents now", len(docs))
            indexed = [embed_doc(d, embedder, settings) for d in docs]
            save_prebuilt(settings, indexed)
        index = HybridIndex.from_parts([(d.chunks, d.vectors) for d in indexed], embedder)
        if settings.qdrant_url and index.chunks:
            vectors = np.stack([v for d in indexed for v in d.vectors])
            index.dense_store = QdrantDense(settings.qdrant_url, settings.collection, vectors)
        log.info("Sample library: %d documents, %d chunks, dense backend=%s", len(indexed), len(index.chunks), index.backend)
        return cls(indexed, index)


def _fingerprint(settings, docs: list[Document]) -> str:
    h = hashlib.sha256(f"{TEXT_VERSION}|{settings.embed_model}|{settings.chunk_size}|{settings.chunk_overlap}".encode())
    for d in docs:
        h.update(d.id.encode() + b"\0" + d.text.encode())
    return h.hexdigest()[:16]


def save_prebuilt(settings, indexed: list[IndexedDoc]) -> None:
    try:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        meta = {"fingerprint": _fingerprint(settings, [d.doc for d in indexed]),
                "docs": [{"id": d.doc.id, "chunks": [c.text for c in d.chunks]} for d in indexed]}
        vectors = np.concatenate([d.vectors for d in indexed if len(d.vectors)]) if indexed else np.zeros((0, 1))
        np.save(settings.index_dir / "vectors.npy", vectors.astype(np.float32))
        (settings.index_dir / "index.json").write_text(json.dumps(meta))
    except OSError as exc:  # read-only filesystem etc. - not fatal, we just embed again next start
        log.warning("Could not save the prebuilt index: %s", exc)


def _load_prebuilt(settings, docs: list[Document]) -> list[IndexedDoc] | None:
    try:
        meta = json.loads((settings.index_dir / "index.json").read_text())
        vectors = np.load(settings.index_dir / "vectors.npy")
    except (OSError, ValueError):
        return None
    if meta.get("fingerprint") != _fingerprint(settings, docs):
        log.info("Prebuilt index is out of date (documents or settings changed)")
        return None
    out, row = [], 0
    for doc, entry in zip(docs, meta["docs"]):
        n = len(entry["chunks"])
        chunks = [Chunk(i, t, doc.id, doc.title) for i, t in enumerate(entry["chunks"])]
        out.append(IndexedDoc(doc, chunks, vectors[row : row + n]))
        row += n
    return out


# ---------- the visitor sandbox ----------

@dataclass
class Session:
    docs: "OrderedDict[str, IndexedDoc]"
    index: HybridIndex
    last_used: float
    uploads: int = field(default=0)


class Sessions:
    def __init__(self, library: SampleLibrary, embedder, settings, clock=time.monotonic):
        self.library, self.embedder, self.s, self.clock = library, embedder, settings, clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._sessions)

    # --- reads ---
    def index(self, sid: str | None) -> tuple[HybridIndex, bool]:
        """(index to search, is_private). Visitors without changes share the sample index."""
        s = self._get(sid)
        return (s.index, True) if s else (self.library.index, False)

    def document(self, sid: str | None, doc_id: str) -> IndexedDoc:
        for d in self._docs(sid):
            if d.doc.id == doc_id:
                return d
        raise KeyError(doc_id)

    def state(self, sid: str | None) -> dict:
        s = self._get(sid)
        return {
            "custom": s is not None,
            "documents": [d.summary() for d in (s.docs.values() if s else self.library.docs)],
            "limits": {"max_uploads": self.s.max_uploads, "max_upload_mb": round(self.s.max_upload_bytes / 1e6, 1),
                       "ttl_minutes": self.s.session_ttl_minutes, "uploads_left": self.s.max_uploads - (s.uploads if s else 0)},
        }

    # --- changes (the first one copies the samples into a private session) ---
    def remove(self, sid: str, doc_id: str) -> dict:
        with self._lock:
            s = self._get_or_create(sid)
            if doc_id not in s.docs:
                raise KeyError(doc_id)
            del s.docs[doc_id]
            s.index = self._build(s)
        return self.state(sid)

    def add(self, sid: str, filename: str, data: bytes) -> dict:
        self._check(sid)
        name = safe_filename(filename)
        if len(data) > self.s.max_upload_bytes:
            raise ValueError(f"File too large (max {self.s.max_upload_bytes / 1e6:.1f} MB).")
        current = self._get(sid)
        if current and current.uploads >= self.s.max_uploads:
            raise ValueError(f"Upload limit reached ({self.s.max_uploads} files). Reset to start over.")
        text = extract_upload_text(name, data)
        title = name[: -len(suffix(name))] if suffix(name) else name
        doc_id = f"upload-{uuid.uuid4().hex[:8]}-{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')[:40]}"
        doc = Document(doc_id, name, title, text[: self.s.max_doc_chars], "uploaded", len(text) > self.s.max_doc_chars)
        indexed = embed_doc(doc, self.embedder, self.s)  # the slow part: outside the lock
        if not indexed.chunks:
            raise ValueError("That file has too little text to index.")
        with self._lock:
            s = self._get_or_create(sid)
            if s.uploads >= self.s.max_uploads:
                raise ValueError(f"Upload limit reached ({self.s.max_uploads} files). Reset to start over.")
            s.uploads += 1
            s.docs[doc_id] = indexed
            s.index = self._build(s)
        return self.state(sid)

    def reset(self, sid: str) -> dict:
        self._check(sid)
        with self._lock:
            self._sessions.pop(sid, None)
        return self.state(None)

    # --- internals ---
    def _build(self, s: Session) -> HybridIndex:
        return HybridIndex.from_parts([(d.chunks, d.vectors) for d in s.docs.values()], self.embedder)

    def _docs(self, sid):
        s = self._get(sid)
        return list(s.docs.values()) if s else self.library.docs

    def _check(self, sid):
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
        """Caller holds the lock."""
        self._check(sid)
        now = self.clock()
        self._expire(now)
        s = self._sessions.get(sid)
        if s is None:
            while len(self._sessions) >= self.s.max_sessions:
                self._sessions.popitem(last=False)
            s = Session(OrderedDict((d.doc.id, d) for d in self.library.docs), self.library.index, now)
            self._sessions[sid] = s
        s.last_used = now
        self._sessions.move_to_end(sid)
        return s
