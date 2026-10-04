"""Shared data types."""
from dataclasses import dataclass


@dataclass
class Chunk:
    id: int  # unique within one index
    text: str
    source: str  # id of the document it came from (sample file name, or an upload id)
    title: str  # document title, e.g. "Quillfeather Labs - Expense and Spending Policy"


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Document:
    id: str
    name: str  # file name shown to the visitor
    title: str
    text: str  # full original text (shown in the document viewer)
    origin: str  # "sample" or "uploaded"
    truncated: bool = False
