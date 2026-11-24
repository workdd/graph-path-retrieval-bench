"""Choose retriever hyperparameters on validation data only.

Selection metric per setting: mean of (MetaQA validation hit@10 averaged over
hops) and (WebQSP validation hit@10). Ties go to the setting listed first.

Run: .venv/bin/python scripts/tune.py --workers 4 | tee docs/tuning.md
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import time
import warnings
from collections import defaultdict

import numpy as np

from relpath_bench import registry
from relpath_bench.datasets import load
from relpath_bench.harness import rank
from relpath_bench.types import Limits

GRIDS: dict[str, dict[str, list]] = {
    "bfs": {},
    "ppr": {"damping": [0.5, 0.85]},
    "pathrag": {"alpha": [0.8, 0.9], "theta": [0.0, 0.001, 0.01, 0.05]},
    "beam": {"width": [1, 3, 5]},
    "pcst": {"edge_cost": [0.25, 0.5, 1.0]},
}
SAMPLE = {"metaqa": 300, "webqsp": None}
_DATA: dict = {}
_EMB = []


def settings(algo: str) -> list[dict]:
    grid = GRIDS[algo]
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


def _embedder():
    if not _EMB:
        from relpath_bench.embed import SentenceTransformerEmbedder

        _EMB.append(SentenceTransformerEmbedder())
    return _EMB[0]


def _init() -> None:
    import relpath_bench.retrievers  # noqa: F401

    warnings.simplefilter("ignore")
    for ds in ("metaqa", "webqsp"):
        _DATA[ds] = list(load(ds, "validation", SAMPLE[ds], seed=0))


def evaluate(task: tuple[str, dict]) -> tuple[str, dict, float, float, float]:
    algo, params = task
    emb = _embedder() if getattr(registry._REGISTRY[algo], "needs_embedder", False) else None
    out = {}
    for ds, examples in _DATA.items():
        r = registry.make(algo, embedder=emb, **params)
        by_hop = defaultdict(list)
        for ex in examples:
            res = r.retrieve(ex.graph, ex.query, 20, Limits())
            top = rank(res.paths, 10)
            hit = bool({p.end for p in top} & ex.query.answers)
            by_hop[ex.query.hops if ds == "metaqa" else 0].append(hit)
        out[ds] = float(np.mean([np.mean(v) for v in by_hop.values()]))
    return algo, params, out["metaqa"], out["webqsp"], (out["metaqa"] + out["webqsp"]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    _init()
    tasks = [(a, s) for a in GRIDS for s in settings(a)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        results = pool.map(evaluate, tasks, chunksize=1)
    print("# Hyperparameter tuning (validation only)\n")
    print("Selection metric: mean of MetaQA validation hit@10 (300 questions per hop, seed 0,")
    print("averaged over hops) and WebQSP validation hit@10 (all questions). Ties go to the")
    print("setting listed first in the grid. Max path length 3 hops.\n")
    print(f"Questions: MetaQA {len(_DATA['metaqa'])}, WebQSP {len(_DATA['webqsp'])}. "
          f"Wall time {time.time() - t0:.0f} s.\n")
    print("| algorithm | setting | MetaQA hit@10 | WebQSP hit@10 | selection | chosen |")
    print("|---|---|---|---|---|---|")
    for algo in GRIDS:
        rows = [r for r in results if r[0] == algo]
        best = max(rows, key=lambda r: r[4] + 0)  # max() keeps the first of equal values
        for a, p, m, w, s in rows:
            tag = "yes" if (a, p) == (best[0], best[1]) else ""
            label = ", ".join(f"{k}={v}" for k, v in p.items()) or "(none)"
            print(f"| {a} | {label} | {m:.4f} | {w:.4f} | {s:.4f} | {tag} |")


if __name__ == "__main__":
    main()
