# Hyperparameter tuning (validation only)

Selection metric: mean of MetaQA validation hit@10 (300 questions per hop, seed 0,
averaged over hops) and WebQSP validation hit@10 (all questions). Ties go to the
setting listed first in the grid. Max path length 3 hops.

Questions: MetaQA 900, WebQSP 246. Wall time 568 s.

| algorithm | setting | MetaQA hit@10 | WebQSP hit@10 | selection | chosen |
|---|---|---|---|---|---|
| bfs | (none) | 0.6533 | 0.1911 | 0.4222 | yes |
| ppr | damping=0.5 | 0.7133 | 0.2886 | 0.5010 |  |
| ppr | damping=0.85 | 0.7278 | 0.3333 | 0.5306 | yes |
| pathrag | alpha=0.8, theta=0.0 | 0.7178 | 0.2439 | 0.4808 |  |
| pathrag | alpha=0.8, theta=0.001 | 0.7211 | 0.2520 | 0.4866 |  |
| pathrag | alpha=0.8, theta=0.01 | 0.7189 | 0.1504 | 0.4346 |  |
| pathrag | alpha=0.8, theta=0.05 | 0.5944 | 0.0366 | 0.3155 |  |
| pathrag | alpha=0.9, theta=0.0 | 0.7178 | 0.2480 | 0.4829 |  |
| pathrag | alpha=0.9, theta=0.001 | 0.7211 | 0.2602 | 0.4906 | yes |
| pathrag | alpha=0.9, theta=0.01 | 0.7256 | 0.1707 | 0.4481 |  |
| pathrag | alpha=0.9, theta=0.05 | 0.6378 | 0.0447 | 0.3412 |  |
| beam | width=1 | 0.3389 | 0.4390 | 0.3890 |  |
| beam | width=3 | 0.3722 | 0.4756 | 0.4239 |  |
| beam | width=5 | 0.3767 | 0.4797 | 0.4282 | yes |
| pcst | edge_cost=0.25 | 0.4067 | 0.3089 | 0.3578 | yes |
| pcst | edge_cost=0.5 | 0.3911 | 0.2398 | 0.3155 |  |
| pcst | edge_cost=1.0 | 0.3333 | 0.2317 | 0.2825 |  |

Command: `.venv/bin/python scripts/tune.py --workers 5` (9.5 min, 8 cores, 16 GB, MiniLM embedder `all-MiniLM-L6-v2`).
