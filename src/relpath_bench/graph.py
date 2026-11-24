"""Immutable CSR knowledge graph with explicit inverse edges.

Relation ids: forward relations are ``0..R-1`` (sorted by name), inverse
relations are ``R..2R-1`` and named ``"inverse of " + forward name``. Node ids
follow sorted node names, so ids do not depend on input order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Graph:
    node_names: list[str]
    rel_names: list[str]
    indptr: np.ndarray
    nbr: np.ndarray
    rel: np.ndarray
    n_forward_rels: int

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    def node_id(self, name: str) -> int | None:
        index = self.__dict__.get("_index")
        if index is None:
            index = {n: i for i, n in enumerate(self.node_names)}
            object.__setattr__(self, "_index", index)
        return index.get(name)

    def degree(self, u: int) -> int:
        return int(self.indptr[u + 1] - self.indptr[u])

    def neighbours(self, u: int) -> list[tuple[int, int]]:
        """Return ``(rel_id, v)`` pairs of ``u``, sorted ascending."""
        lo, hi = int(self.indptr[u]), int(self.indptr[u + 1])
        return list(zip(self.rel[lo:hi].tolist(), self.nbr[lo:hi].tolist()))

    def canonical(self, h: int, r: int, t: int) -> tuple[int, int, int]:
        """Map an edge to its forward triple (inverse edges are flipped)."""
        if r >= self.n_forward_rels:
            return (t, r - self.n_forward_rels, h)
        return (h, r, t)


def build_graph(triples: Iterable[tuple[str, str, str]]) -> Graph:
    unique = set(triples)
    node_names = sorted({x for h, _, t in unique for x in (h, t)})
    fwd_names = sorted({r for _, r, _ in unique})
    n_rel = len(fwd_names)
    rel_names = fwd_names + ["inverse of " + r for r in fwd_names]
    nid = {n: i for i, n in enumerate(node_names)}
    rid = {r: i for i, r in enumerate(fwd_names)}

    edges: set[tuple[int, int, int]] = set()
    for h, r, t in unique:
        hi, ri, ti = nid[h], rid[r], nid[t]
        edges.add((hi, ri, ti))
        if hi != ti:  # a self loop is a single edge, not a forward/inverse pair
            edges.add((ti, ri + n_rel, hi))

    rows = sorted(edges)
    n = len(node_names)
    indptr = np.zeros(n + 1, dtype=np.int64)
    for u, _, _ in rows:
        indptr[u + 1] += 1
    np.cumsum(indptr, out=indptr)
    # rows are sorted by (u, rel, v); neighbours() must return (rel, v) ascending
    rel = np.array([r for _, r, _ in rows], dtype=np.int32)
    nbr = np.array([v for _, _, v in rows], dtype=np.int64)
    return Graph(node_names, rel_names, indptr, nbr, rel, n_rel)
