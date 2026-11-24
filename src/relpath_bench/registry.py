"""Name-based retriever registry."""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator: register ``cls`` under its ``name`` attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def names() -> list[str]:
    return sorted(_REGISTRY)


def make(name: str, embedder: "Embedder | None" = None, **kwargs):  # noqa: F821
    """Instantiate a registered retriever; ``kwargs`` go to its constructor."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown retriever {name!r}; known: {names()}")
    cls = _REGISTRY[name]
    if getattr(cls, "needs_embedder", False):
        if embedder is None:
            raise ValueError(f"{name} needs an embedder")
        return cls(embedder=embedder, **kwargs)
    return cls(**kwargs)
