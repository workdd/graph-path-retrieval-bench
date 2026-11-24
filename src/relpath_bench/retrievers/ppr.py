"""Personalized PageRank retriever.

Algorithm: power iteration of ``x <- d * P^T x + (1 - d) * p`` on the
undirected graph (the CSR already holds inverse edges), where ``p`` is uniform
over the query seeds. ``P`` is the row-normalised adjacency; parallel edges
between the same pair of nodes count once per edge. Mass of nodes without
edges teleports back to ``p``. Iteration stops when the L1 change drops below
``tol`` or after ``max_iter`` rounds. The transition matrix is built once per
graph and cached (keyed by ``id(g)``, dropped when the graph is collected).

Paths: only non-seed nodes inside the ``max_hops`` BFS tree (honouring
``max_expanded``, which sets ``capped``) become paths, each with its BFS-tree
path as the relation path.

Scoring rule: ``score`` is the node's PPR value. Ties are broken by the
harness.

Hyperparameters: ``damping`` default 0.85 (standard PageRank); HippoRAG value
to be confirmed from arXiv:2405.14831 and tuned on validation in Task 12.
``tol`` 1e-8 and ``max_iter`` 100 are convergence settings, not tuned.
"""

from __future__ import annotations

import weakref

import numpy as np
from scipy.sparse import csr_matrix

from relpath_bench.graph import Graph
from relpath_bench.harness import bfs_tree
from relpath_bench.registry import register
from relpath_bench.types import Limits, Query, RetrievalResult, ScoredPath


@register
class PprRetriever:
    name = "ppr"
    needs_embedder = False

    def __init__(
        self, damping: float = 0.85, tol: float = 1e-8, max_iter: int = 100
    ) -> None:
        self.damping = damping
        self.tol = tol
        self.max_iter = max_iter
        self._cache: dict[int, csr_matrix] = {}

    def _transpose(self, g: Graph) -> csr_matrix:
        key = id(g)
        mat = self._cache.get(key)
        if mat is None:
            n = g.n_nodes
            deg = np.diff(g.indptr).astype(float)
            rows = np.repeat(np.arange(n), np.diff(g.indptr))
            inv = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
            # Entry (v, u) = 1/deg(u) per edge u->v, i.e. P transposed.
            mat = csr_matrix((inv[rows], (g.nbr, rows)), shape=(n, n))
            self._cache[key] = mat
            weakref.finalize(g, self._cache.pop, key, None)
        return mat

    def _scores(self, g: Graph, seeds: tuple[int, ...]) -> np.ndarray:
        n = g.n_nodes
        p = np.zeros(n)
        np.add.at(p, list(seeds), 1.0)
        p /= p.sum()
        mat = self._transpose(g)
        dangling = np.diff(g.indptr) == 0
        x = p.copy()
        for _ in range(self.max_iter):
            nxt = mat @ x
            nxt += x[dangling].sum() * p
            nxt = self.damping * nxt + (1.0 - self.damping) * p
            done = np.abs(nxt - x).sum() < self.tol
            x = nxt
            if done:
                break
        return x

    def retrieve(self, g: Graph, q: Query, k: int, limits: Limits) -> RetrievalResult:
        if not q.seeds:
            return RetrievalResult([], False)
        x = self._scores(g, q.seeds)
        tree, capped = bfs_tree(g, q.seeds, limits.max_hops, limits.max_expanded)
        paths = [
            ScoredPath(nodes, rels, float(x[nodes[-1]]))
            for nodes, rels in tree.values()
            if rels
        ]
        return RetrievalResult(paths, capped)
