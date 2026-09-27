"""Shared data types used across modules."""
from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    source: str  # file name, e.g. quillfeather-expense-policy.md
    title: str  # document title, e.g. "Quillfeather Labs - Expense and Spending Policy"


@dataclass
class Hit:
    chunk: Chunk
    score: float
