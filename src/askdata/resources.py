"""Loaders for the knowledge files and the evaluation sets."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import EVAL_DIR, KNOWLEDGE_DIR


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=None)
def load_glossary(path: Path = KNOWLEDGE_DIR / "glossary.md") -> str:
    return Path(path).read_text(encoding="utf-8")


def load_examples(path: Path = KNOWLEDGE_DIR / "examples.jsonl") -> list[dict]:
    """Few-shot question -> SQL examples (kept separate from the evaluation set)."""
    return _jsonl(Path(path))


def load_gold(path: Path = EVAL_DIR / "gold.jsonl") -> list[dict]:
    return _jsonl(Path(path))


def load_adversarial(path: Path = EVAL_DIR / "adversarial.jsonl") -> list[dict]:
    return _jsonl(Path(path))
