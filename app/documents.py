"""Turn an uploaded file into plain text. Supports .txt, .md, .pdf and .docx (no extra services needed)."""
import io
import re
import zipfile
from xml.etree import ElementTree

TEXT_TYPES = {".txt", ".md", ".markdown", ".rst", ".csv"}
SUPPORTED_UPLOADS = TEXT_TYPES | {".pdf", ".docx"}


def suffix(filename: str) -> str:
    m = re.search(r"\.[A-Za-z0-9]+$", filename or "")
    return m.group(0).lower() if m else ""


def safe_filename(filename: str) -> str:
    name = re.sub(r"[^\w.\- ]+", "_", (filename or "").split("/")[-1].split("\\")[-1]).strip(" .")
    return (name or "document")[:80]


def extract_text(filename: str, data: bytes) -> str:
    """Raises ValueError with a user-facing message if the file can't be read."""
    ext = suffix(filename)
    if ext not in SUPPORTED_UPLOADS:
        raise ValueError(f"Unsupported file type '{ext or '?'}'. Use: {', '.join(sorted(SUPPORTED_UPLOADS))}")
    if ext in TEXT_TYPES:
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

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages[:200])
    except Exception as exc:
        raise ValueError("Couldn't read that PDF.") from exc


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
    except Exception as exc:
        raise ValueError("Couldn't read that Word file.") from exc
    paragraphs = []
    for p in root.iter(f"{W}p"):
        paragraphs.append("".join(t.text or "" for t in p.iter(f"{W}t")))
    return "\n\n".join(p for p in paragraphs if p.strip())
