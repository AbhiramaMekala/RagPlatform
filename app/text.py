"""Everything about turning files into clean, chunked text.

  sample docs (data/docs/*.md)  ─┐
                                 ├─> clean_text -> chunk_text (LangChain splitter) -> Chunks
  visitor uploads (.pdf .docx …) ┘   (uploads go through extract_upload_text first)
"""
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .models import Chunk, Document

SAMPLE_TYPES = {".md", ".txt", ".rst"}
TEXT_UPLOADS = {".txt", ".md", ".markdown", ".rst", ".csv"}
UPLOAD_TYPES = TEXT_UPLOADS | {".pdf", ".docx"}
TEXT_VERSION = "2"  # bump when cleaning/chunking changes, so prebuilt indexes are rebuilt
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


# ---------- sample documents ----------

def load_samples(docs_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SAMPLE_TYPES:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(path.name, path.name, extract_title(raw, path.stem), raw, "sample"))
    return docs


def extract_title(raw: str, fallback: str) -> str:
    """Title from front matter (`title: Time off`) or a `Title:` header line; else the file name."""
    m = re.search(r"^title:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
    if not m:
        return fallback.replace("-", " ").replace("_", " ").strip().title()
    return m.group(1).strip().strip("\"'").replace("``", "")


def clean_text(raw: str) -> str:
    """Strip Markdown / reStructuredText markup so chunks read like plain prose."""
    text = FRONT_MATTER.sub("", raw)
    text = re.sub(r"^import .*$", "", text, flags=re.MULTILINE)  # MDX imports
    text = re.sub(r"<[^>\n]+/?>", "", text)  # HTML / JSX tags
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [link text](url) -> link text
    text = re.sub(r":[\w-]+:`([^`<]*?)(?:\s*<[^>]*>)?`", r"\1", text)  # rst :role:`x` -> x
    text = re.sub(r"`{1,2}([^`\n]+)`{1,2}_?", r"\1", text)  # `code` -> code
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*([=\-~^#*+])\1{3,}\s*", line):  # heading underlines / rules
            continue
        if re.match(r"\s*\.\. (\w[\w-]*::|_|\[)", line):  # rst directives
            continue
        lines.append(re.sub(r"^#{1,6}\s+", "", line.rstrip()))  # "## Heading" -> "Heading"
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def chunk_text(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split one document into overlapping chunks (ids are local to the document)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = [p for p in splitter.split_text(clean_text(doc.text)) if len(p) > 80]  # drop tiny fragments
    return [Chunk(i, p, doc.id, doc.title) for i, p in enumerate(pieces)]


# ---------- visitor uploads ----------

def suffix(filename: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]+$", filename or "")
    return m.group(0).lower() if m else ""


def safe_filename(filename: str) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", (filename or "").split("/")[-1].split("\\")[-1]).strip(" .")
    return (name or "document")[:80]


def extract_upload_text(filename: str, data: bytes) -> str:
    """Plain text from an uploaded file. Raises ValueError with a message meant for the visitor."""
    ext = suffix(filename)
    if ext not in UPLOAD_TYPES:
        raise ValueError(f"Unsupported file type '{ext or '?'}'. Use: {', '.join(sorted(UPLOAD_TYPES))}")
    if ext in TEXT_UPLOADS:
        text = _decode(data)
    elif ext == ".pdf":
        text = _pdf_text(data)
    else:
        text = _docx_text(data)
    text = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\x00", ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 30:
        raise ValueError("Couldn't find readable text in that file (scanned PDFs and images aren't supported).")
    return text


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages[:200])
    except Exception as exc:
        raise ValueError("Couldn't read that PDF.") from exc


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except Exception as exc:
        raise ValueError("Couldn't read that Word file.") from exc
    paragraphs = ("".join(t.text or "" for t in p.iter(f"{_W}t")) for p in root.iter(f"{_W}p"))
    return "\n\n".join(p for p in paragraphs if p.strip())
