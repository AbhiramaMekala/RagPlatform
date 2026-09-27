from pathlib import Path

import pytest

from app.ingest import clean_text, extract_title, load_documents

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_extract_title_from_pep_header():
    raw = "PEP: 8\nTitle: Style Guide for Python Code\nAuthor: Guido\n"
    assert extract_title(raw, "x") == "Style Guide for Python Code"


def test_clean_text_strips_rst_markup():
    raw = "Introduction\n============\n\nSee :pep:`style guide <7>` and ``print()``.\n\n.. code-block:: python\n"
    out = clean_text(raw)
    assert "====" not in out and ":pep:" not in out and "``" not in out
    assert "style guide" in out and "print()" in out


def test_markdown_front_matter_and_links_cleaned():
    raw = "---\ntitle: Time off\nsidebar: Handbook\n---\n\nSee [the calendar](https://x.com) <Calc job=\"a\" /> now."
    assert extract_title(raw, "x") == "Time off"
    out = clean_text(raw)
    assert out == "See the calendar  now."


def test_corpus_loads():
    docs = load_documents(DOCS)
    assert len(docs) >= 5
    assert all(title and text for _, title, text in docs)
    assert not any(text.startswith("---") for _, _, text in docs)


def test_chunking():
    pytest.importorskip("langchain_text_splitters")
    from app.ingest import chunk_documents

    chunks = chunk_documents(load_documents(DOCS)[:2], chunk_size=800, chunk_overlap=120)
    assert chunks and all(len(c.text) <= 800 for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))
