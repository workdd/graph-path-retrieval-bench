"""Sentence embedders used by question-aware retrievers."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised row per text (zero rows stay zero)."""
        ...


def relation_text(name: str) -> str:
    """Turn a relation name into plain words (underscores become spaces)."""
    return name.replace("_", " ")


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


class FixedEmbedder:
    """Lookup-table embedder for tests; unknown sentences map to the zero vector."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.dim = len(next(iter(table.values()))) if table else 1

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = [np.asarray(self.table.get(t, [0.0] * self.dim), dtype=np.float64) for t in texts]
        if not rows:
            return np.zeros((0, self.dim))
        return _normalise(np.vstack(rows))


class SentenceTransformerEmbedder:
    """sentence-transformers wrapper with an in-memory per-sentence cache.

    Needs the optional ``[embed]`` extra; raises ``ImportError`` without it.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._name = model
        self._model = SentenceTransformer(model)
        self._cache: dict[str, np.ndarray] = {}

    def __getstate__(self) -> dict:
        # The torch model cannot be pickled to spawned workers; each worker
        # reloads it from the model name and starts with an empty cache.
        return {"_name": self._name}

    def __setstate__(self, state: dict) -> None:
        from sentence_transformers import SentenceTransformer

        self._name = state["_name"]
        self._model = SentenceTransformer(self._name)
        self._cache = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = np.asarray(self._model.encode(missing), dtype=np.float64)
            for t, v in zip(missing, _normalise(vecs)):
                self._cache[t] = v
        if not texts:
            return np.zeros((0, 0))
        return np.vstack([self._cache[t] for t in texts])
