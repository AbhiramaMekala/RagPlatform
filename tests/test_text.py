import io
import zipfile
from pathlib import Path

import pytest

from app.models import Document
from app.text import clean_text, extract_title, extract_upload_text, load_samples, safe_filename

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_title_from_front_matter_header_or_filename():
    assert extract_title("---\ntitle: Time off\n---\nbody", "x") == "Time off"
    assert extract_title("PEP: 8\nTitle: Style Guide for Python Code\n", "x") == "Style Guide for Python Code"
    assert extract_title("no title here", "my-notes_v2") == "My Notes V2"


def test_clean_text_strips_markup():
    raw = "---\ntitle: T\n---\n\nSee [the calendar](https://x.com) <Calc job=\"a\" /> and ``print()``.\n\nHeading\n=======\n"
    out = clean_text(raw)
    assert "calendar" in out and "print()" in out
    assert "https://" not in out and "<Calc" not in out and "=====" not in out and "title:" not in out
    assert clean_text("## Approval limits\n\nText") == "Approval limits\n\nText"


def test_samples_load():
    docs = load_samples(DOCS)
    assert len(docs) >= 5 and all(d.title and d.text and d.origin == "sample" for d in docs)


def test_chunking():
    pytest.importorskip("langchain_text_splitters")
    from app.text import chunk_text

    doc = load_samples(DOCS)[0]
    chunks = chunk_text(doc, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 2 and all(len(c.text) <= 400 and c.source == doc.id for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert chunk_text(Document("e", "e.txt", "E", "tiny", "uploaded"), 400, 50) == []


DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Refund policy for the Lumen store.</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Refunds are accepted within </w:t></w:r><w:r><w:t>30 days of purchase.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def test_upload_text_formats():
    assert extract_upload_text("a.md", b"# Title\r\n\r\n\r\n\r\nBody text that is long enough.") == "# Title\n\nBody text that is long enough."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", DOCX_XML)
    assert "accepted within 30 days of purchase" in extract_upload_text("p.docx", buf.getvalue())


def test_upload_rejections():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_upload_text("x.exe", b"abc")
    with pytest.raises(ValueError):
        extract_upload_text("x.txt", b"   ")
    with pytest.raises(ValueError):
        extract_upload_text("x.docx", b"definitely not a zip file at all")


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("My <Report>.pdf") == "My _Report_.pdf"
