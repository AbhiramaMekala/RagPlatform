import io
import zipfile

import pytest

from app.documents import extract_text, safe_filename

DOCX_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Refund policy for the Lumen store.</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Refunds are accepted within </w:t></w:r><w:r><w:t>30 days of purchase.</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def test_plain_text_and_markdown():
    assert extract_text("a.md", "# Title\r\n\r\n\r\n\r\nBody text that is long enough.".encode()) == "# Title\n\nBody text that is long enough."


def test_docx():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", DOCX_XML)
    text = extract_text("policy.docx", buf.getvalue())
    assert "Refunds are accepted within 30 days of purchase." in text


def test_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("x.exe", b"abc")
    with pytest.raises(ValueError):
        extract_text("x.txt", b"   ")
    with pytest.raises(ValueError):
        extract_text("x.docx", b"not a zip file at all, definitely")


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("My <Report>.pdf") == "My _Report_.pdf"
