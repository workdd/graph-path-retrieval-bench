"""PathRAG-style flow-based pruning retriever.

Paper: PathRAG, arXiv:2502.14902 (resource propagation with flow-based path
pruning). Implemented from the paper's description only.

Difference from the paper: the paper scores relational paths between pairs of
retrieved nodes. Here every query seed is the single source of the flow, and
each node that ends up holding resource yields one path, the propagation-tree
path from the seed to that node.

Algorithm, per seed ``s``: ``S[s] = 1`` and layer 0 is ``{s}``. For layer
``d -> d + 1`` every node ``v`` that holds no resource yet receives
``S[v] = sum over u in layer d adjacent to v of alpha * S[u] / degree(u)``
(one term per edge; ``degree`` counts all incident edges including inverse
edges). ``prev[v]`` is the ``u`` with the largest single contribution, ties
broken by ``(rel, u)`` ascending. A node with ``S[v] < theta`` is pruned: it
gets no path, joins no layer and propagates nothing, and it is not revisited
by later layers. Propagation stops after ``max_hops`` layers.

Scoring rule: a path follows ``prev`` from ``s`` to ``v``; its score is the
mean of ``S`` over the path's nodes excluding ``s``. With several seeds each
seed is computed on its own and identical ``(nodes, rels)`` keep the higher
score. Paths are simple because a node receives resource only once.

Expansion budget: each node whose neighbours are scanned counts as one
expansion, over all seeds together; reaching ``max_expanded`` with nodes still
to expand returns ``capped=True``.

Hyperparameters: grid ``alpha`` {0.8, 0.9} x ``theta`` {0.0, 0.001, 0.01,
0.05}; chosen ``alpha=0.9``, ``theta=0.001`` by mean of MetaQA and WebQSP
validation hit@10 (see docs/tuning.md).
"""

from __future__ import annotations

from relpath_bench.graph import Graph
from relpath_bench.registry import register
from relpath_bench.types import Limits, Query, RetrievalResult, ScoredPath


@register
class PathRagRetriever:
    name = "pathrag"
    needs_embedder = False

    def __init__(self, alpha: float = 0.9, theta: float = 0.001) -> None:
        self.alpha = alpha
        self.theta = theta

    def _flow(
        self, g: Graph, seed: int, limits: Limits, budget: list[int]
    ) -> tuple[list[ScoredPath], bool]:
        resource: dict[int, float] = {seed: 1.0}
        path: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {seed: ((seed,), ())}
        total: dict[int, float] = {seed: 0.0}  # sum of S along the path, excl. seed
        out: list[ScoredPath] = []
        layer = [seed]
        for _ in range(limits.max_hops):
            if not layer:
                break
            # v -> (summed resource, best contribution key, best u)
            incoming: dict[int, list] = {}
            for u in layer:
                if budget[0] >= limits.max_expanded:
                    return out, True
                budget[0] += 1
                deg = g.degree(u)
                for r, v in g.neighbours(u):
                    if v in resource:
                        continue
                    c = self.alpha * resource[u] / deg
                    acc = incoming.get(v)
                    if acc is None:
                        incoming[v] = [c, c, r, u]
                        continue
                    acc[0] += c
                    if c > acc[1] or (c == acc[1] and (r, u) < (acc[2], acc[3])):
                        acc[1], acc[2], acc[3] = c, r, u
            nxt: list[int] = []
            for v in sorted(incoming):
                s_v, _, r, u = incoming[v]
                resource[v] = s_v
                if s_v < self.theta:
                    continue
                nodes, rels = path[u]
                path[v] = (nodes + (v,), rels + (r,))
                total[v] = total[u] + s_v
                out.append(ScoredPath(path[v][0], path[v][1], total[v] / len(rels + (r,))))
                nxt.append(v)
            layer = nxt
        return out, False

    def retrieve(self, g: Graph, q: Query, k: int, limits: Limits) -> RetrievalResult:
        if not q.seeds:
            return RetrievalResult([], False)
        budget = [0]
        best: dict[tuple[tuple[int, ...], tuple[int, ...]], ScoredPath] = {}
        capped = False
        for seed in dict.fromkeys(q.seeds):
            paths, hit = self._flow(g, seed, limits, budget)
            for p in paths:
                key = (p.nodes, p.rels)
                if key not in best or p.score > best[key].score:
                    best[key] = p
            if hit:
                capped = True
                break
        return RetrievalResult(list(best.values()), capped)
