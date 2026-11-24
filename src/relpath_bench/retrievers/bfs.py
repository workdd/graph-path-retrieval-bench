"""k-hop BFS retriever: the baseline.

Algorithm: breadth-first search from the query seeds up to ``max_hops``,
stopping when ``max_expanded`` nodes have been expanded (``capped=True``).
Each reached non-seed node yields its single BFS-tree (shortest) path.

Scoring rule: ``score = -length``, so shorter paths rank higher. Ties are
broken by the harness, not here.

Hyperparameters: none (hop and expansion limits come from ``Limits``).
"""

from __future__ import annotations

from relpath_bench.graph import Graph
from relpath_bench.harness import bfs_tree
from relpath_bench.registry import register
from relpath_bench.types import Limits, Query, RetrievalResult, ScoredPath


@register
class BfsRetriever:
    name = "bfs"
    needs_embedder = False

    def retrieve(self, g: Graph, q: Query, k: int, limits: Limits) -> RetrievalResult:
        tree, capped = bfs_tree(g, q.seeds, limits.max_hops, limits.max_expanded)
        paths = [
            ScoredPath(nodes, rels, -float(len(rels)))
            for nodes, rels in tree.values()
            if rels
        ]
        return RetrievalResult(paths, capped)
