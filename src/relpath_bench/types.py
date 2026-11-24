"""Shared value types and the retriever protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from relpath_bench.graph import Graph


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    seeds: tuple[int, ...]
    answers: frozenset[int]
    hops: int | None = None


@dataclass(frozen=True)
class ScoredPath:
    nodes: tuple[int, ...]
    rels: tuple[int, ...]
    score: float

    @property
    def end(self) -> int:
        return self.nodes[-1]

    @property
    def length(self) -> int:
        return len(self.rels)


@dataclass(frozen=True)
class Limits:
    max_hops: int = 3
    max_expanded: int = 10_000


@dataclass
class RetrievalResult:
    paths: list[ScoredPath]
    capped: bool


class Retriever(Protocol):
    name: str
    needs_embedder: bool

    def retrieve(self, g: "Graph", q: Query, k: int, limits: Limits) -> RetrievalResult: ...
