"""Prize-collecting Steiner tree retriever, question-aware.

Paper: G-Retriever, arXiv:2402.07630 (retrieval as a prize-collecting Steiner
tree over a question-relevant subgraph). Solver: ``pcst_fast`` (optional
dependency, imported lazily so ``relpath-bench list`` works without it).
The GNN and LLM parts of G-Retriever are excluded; only the retrieval step is
reproduced.

Algorithm:
1. ``bfs_tree`` keeps the nodes within ``max_hops`` of the seeds (expansion
   cap applied, ``capped`` is propagated). The subgraph is every graph edge
   whose two ends are both kept. Each unordered node pair becomes one
   undirected edge that keeps the direction-aware relation with the highest
   question similarity (ties: relation id ascending); self loops are dropped.
2. Node prizes: the ``top_nodes`` nodes most similar to the question (cosine
   between the question and the node name; ties by node id) get prizes
   ``top_nodes, top_nodes - 1, ..., 1``. Each seed's prize is the total of all
   prizes plus 1, so seeds always stay in the tree.
3. Edge cost: ``max(0, edge_cost - similarity(relation))``. This replaces
   G-Retriever's edge prizes (virtual edge nodes) by a cost reduction, an
   approximation. Zero costs are passed to the solver unchanged.
4. ``pcst_fast(edges, prizes, costs, -1, 1, "gw", 0)`` returns the tree.
5. Inside the tree, a BFS from each seed (neighbours by ascending node id)
   gives one path to every other tree node (paths longer than ``max_hops``
   are dropped). Score = sum of the prizes of the
   path nodes excluding the seed, divided by the path length. With several
   seeds the higher score is kept per identical (nodes, rels).

Hyperparameters: ``top_nodes=5`` is fixed. ``edge_cost`` grid {0.25, 0.5, 1.0};
chosen 0.25 by mean of MetaQA and WebQSP validation hit@10 (see
docs/tuning.md).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from relpath_bench.embed import Embedder, relation_text
from relpath_bench.graph import Graph
from relpath_bench.harness import bfs_tree
from relpath_bench.registry import register
from relpath_bench.types import Limits, Query, RetrievalResult, ScoredPath


@register
class PcstRetriever:
    name = "pcst"
    needs_embedder = True

    def __init__(self, embedder: Embedder, top_nodes: int = 5, edge_cost: float = 0.25) -> None:
        self.embedder = embedder
        self.top_nodes = top_nodes
        self.edge_cost = edge_cost

    def _sims(self, question: str, texts: list[str]) -> list[float]:
        vecs = self.embedder.encode([question] + texts)
        return (vecs[1:] @ vecs[0]).tolist() if texts else []

    def retrieve(self, g: Graph, q: Query, k: int, limits: Limits) -> RetrievalResult:
        if not q.seeds:
            return RetrievalResult([], False)
        from pcst_fast import pcst_fast

        seeds = tuple(dict.fromkeys(q.seeds))
        tree, capped = bfs_tree(g, seeds, limits.max_hops, limits.max_expanded)
        nodes = sorted(tree)
        local = {n: i for i, n in enumerate(nodes)}

        # candidate relations per unordered pair (lo < hi), as lo -> hi rels
        pair_rels: dict[tuple[int, int], set[int]] = {}
        for u in nodes:
            for r, v in g.neighbours(u):
                if v > u and v in local:
                    pair_rels.setdefault((u, v), set()).add(r)
        rel_ids = sorted({r for rs in pair_rels.values() for r in rs})
        rel_sim = dict(
            zip(rel_ids, self._sims(q.question, [relation_text(g.rel_names[r]) for r in rel_ids]))
        )

        pairs = sorted(pair_rels)
        kept = {p: min(pair_rels[p], key=lambda r: (-rel_sim[r], r)) for p in pairs}
        costs = [max(0.0, self.edge_cost - rel_sim[kept[p]]) for p in pairs]
        edges = [(local[a], local[b]) for a, b in pairs]

        node_sim = self._sims(q.question, [g.node_names[n] for n in nodes])
        top = sorted(range(len(nodes)), key=lambda i: (-node_sim[i], nodes[i]))[: self.top_nodes]
        prizes = np.zeros(len(nodes))
        for rank_, i in enumerate(top):
            prizes[i] = self.top_nodes - rank_
        seed_prize = float(prizes.sum()) + 1.0
        for s in seeds:
            prizes[local[s]] = seed_prize

        if not edges:
            return RetrievalResult([], capped)
        _, tree_edges = pcst_fast(
            np.asarray(edges, dtype=np.int64),
            prizes,
            np.asarray(costs, dtype=np.float64),
            -1,
            1,
            "gw",
            0,
        )
        adj: dict[int, list[int]] = {}
        for ei in tree_edges.tolist():
            a, b = pairs[ei]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        n_fwd = g.n_forward_rels

        def step_rel(a: int, b: int) -> int:
            if a < b:
                return kept[(a, b)]
            r = kept[(b, a)]
            return r + n_fwd if r < n_fwd else r - n_fwd

        out: dict[tuple[tuple[int, ...], tuple[int, ...]], ScoredPath] = {}
        for s in seeds:
            if s not in adj:
                continue
            seen = {s: ((s,), (), 0.0)}
            queue = deque([s])
            while queue:
                u = queue.popleft()
                path_nodes, path_rels, total = seen[u]
                for v in sorted(adj.get(u, ())):
                    if v in seen:
                        continue
                    t = total + prizes[local[v]]
                    seen[v] = (path_nodes + (v,), path_rels + (step_rel(u, v),), t)
                    queue.append(v)
                    if len(seen[v][1]) > limits.max_hops:
                        continue  # a tree path can be longer than the BFS depth
                    sp = ScoredPath(seen[v][0], seen[v][1], float(t) / len(seen[v][1]))
                    key = (sp.nodes, sp.rels)
                    if key not in out or sp.score > out[key].score:
                        out[key] = sp
        return RetrievalResult(list(out.values()), capped)
