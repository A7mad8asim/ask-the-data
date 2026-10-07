"""Pick the few-shot examples most similar to the user's question.

Default: character n-gram TF-IDF. It needs no model download, works the same for
Arabic and English, and tolerates spelling variants. Optional: bge-m3 embeddings
served by Ollama (EMBEDDINGS=bge-m3), with automatic fallback to TF-IDF.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from typing import Callable, Sequence

log = logging.getLogger(__name__)

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟـ]")  # harakat and tatweel
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_text(text: str) -> str:
    """Case-fold, strip Arabic diacritics and unify common letter variants."""
    t = unicodedata.normalize("NFKC", text or "").casefold().translate(_DIGITS)
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter:
    t = f" {normalize_text(text)} "
    grams: Counter = Counter()
    for n in range(n_min, n_max + 1):
        for i in range(len(t) - n + 1):
            grams[t[i : i + n]] += 1
    return grams


class ExampleRetriever:
    def __init__(
        self,
        examples: Sequence[dict],
        method: str = "tfidf",
        embed: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        self.examples = list(examples)
        self.method = "tfidf"
        self._embed = None
        if method != "tfidf" and embed is not None:
            try:
                self._matrix = [self._unit(v) for v in embed([e["question"] for e in self.examples])]
                self._embed = embed
                self.method = method
            except Exception as e:  # e.g. Ollama not running, or the model not pulled
                log.warning("Embedding retrieval unavailable (%s); using TF-IDF.", e)
        if self.method == "tfidf":
            grams = [char_ngrams(e["question"]) for e in self.examples]
            df = Counter(g for doc in grams for g in doc)
            n = len(grams)
            self._idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
            self._vectors = [self._tfidf(doc) for doc in grams]

    @staticmethod
    def _unit(v: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _tfidf(self, grams: Counter) -> dict[str, float]:
        vec = {g: (1 + math.log(c)) * self._idf.get(g, 0.0) for g, c in grams.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {g: w / norm for g, w in vec.items() if w}

    def top_k(self, question: str, k: int = 4) -> list[dict]:
        if k <= 0 or not self.examples:
            return []
        if self._embed is not None:
            q = self._unit(self._embed([question])[0])
            scores = [sum(a * b for a, b in zip(q, row)) for row in self._matrix]
        else:
            q = self._tfidf(char_ngrams(question))
            scores = [sum(w * v.get(g, 0.0) for g, w in q.items()) for v in self._vectors]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.examples[i] for i in order[:k]]
