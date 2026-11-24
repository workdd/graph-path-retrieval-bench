# relpath-bench

A small benchmark that compares relational path retrieval algorithms on knowledge graphs under one harness, one budget and one set of scoring rules.

## Results

<!-- RESULTS:START -->
### metaqa (test-full)

| algorithm | hops | hit@1 | hit@10 | recall@10 | context@10 | p50 ms | capped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beam | 1 | 40.7% | 52.3% | 51.2% | 14.1 | 68.3 | 8 |
| beam | 2 | 12.6% | 28.3% | 25.3% | 12.3 | 46.0 | 2 |
| beam | 3 | 12.1% | 26.1% | 12.2% | 13.9 | 93.5 | 5 |
| bfs | 1 | 40.3% | 97.8% | 95.8% | 10.0 | 21.6 | 0 |
| bfs | 2 | 0.0% | 64.3% | 54.5% | 10.0 | 10.8 | 0 |
| bfs | 3 | 6.2% | 43.6% | 12.6% | 10.0 | 34.5 | 0 |
| pathrag | 1 | 47.7% | 90.8% | 87.7% | 9.7 | 2.2 | 0 |
| pathrag | 2 | 0.3% | 81.3% | 65.5% | 9.7 | 2.9 | 0 |
| pathrag | 3 | 13.6% | 44.6% | 15.8% | 10.0 | 2.3 | 0 |
| pcst | 1 | 22.9% | 63.1% | 50.2% | 5.2 | 216.3 | 0 |
| pcst | 2 | 3.3% | 42.8% | 22.5% | 5.2 | 52.4 | 0 |
| pcst | 3 | 7.6% | 19.3% | 6.4% | 4.0 | 180.9 | 0 |
| ppr | 1 | 47.2% | 98.1% | 95.4% | 10.7 | 62.8 | 0 |
| ppr | 2 | 0.7% | 75.8% | 61.9% | 10.2 | 57.2 | 0 |
| ppr | 3 | 18.5% | 50.8% | 18.1% | 10.7 | 74.9 | 0 |

![metaqa recall versus context](docs/figures/metaqa_context_recall.svg)

### webqsp (test-full)

reachable_ratio (answer within 3 hops of a seed): beam 95.3%, bfs 95.3%, pathrag 95.3%, pcst 95.3%, ppr 95.3%

Subset `all`

| algorithm | hops | hit@1 | hit@10 | recall@10 | context@10 | p50 ms | capped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beam | all | 26.7% | 46.5% | 40.1% | 11.3 | 25.0 | 0 |
| bfs | all | 3.6% | 19.7% | 12.7% | 10.0 | 4.0 | 0 |
| pathrag | all | 6.3% | 24.1% | 14.9% | 12.2 | 1.6 | 0 |
| pcst | all | 8.1% | 29.2% | 17.5% | 6.0 | 513.3 | 0 |
| ppr | all | 8.4% | 33.8% | 22.3% | 12.4 | 6.9 | 0 |

Subset `reachable`

| algorithm | hops | hit@1 | hit@10 | recall@10 | context@10 | p50 ms | capped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| beam | all | 28.0% | 48.8% | 42.1% | 11.3 | 25.0 | 0 |
| bfs | all | 3.8% | 20.6% | 13.3% | 10.0 | 4.0 | 0 |
| pathrag | all | 6.6% | 25.3% | 15.6% | 12.2 | 1.6 | 0 |
| pcst | all | 8.5% | 30.6% | 18.4% | 6.1 | 513.5 | 0 |
| ppr | all | 8.8% | 35.4% | 23.4% | 12.4 | 6.9 | 0 |

![webqsp recall versus context](docs/figures/webqsp_context_recall.svg)
<!-- RESULTS:END -->

## Install

```bash
pip install -e .[embed,metaqa]
```

## Quickstart

```bash
relpath-bench list                                   # registered algorithms
relpath-bench run --dataset metaqa                   # 1,000-question sample per hop count, seed 0
relpath-bench run --dataset metaqa --full            # every question
relpath-bench report --out results --readme README.md
```

The README tables are built from `--full` runs only. Sample runs are for quick checks.

## Algorithms

| Name | Paper | Differences from the paper |
| --- | --- | --- |
| bfs | none (baseline) | Breadth-first search; every reached node yields its shortest path. |
| ppr | HippoRAG, [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) | Personalized PageRank only; nodes inside the hop limit become paths along their BFS-tree path. |
| pathrag | PathRAG, [arXiv:2502.14902](https://arxiv.org/abs/2502.14902) | Single-seed adaptation: flow starts at each query seed and every node holding resource yields its propagation-tree path. |
| beam | Think-on-Graph, [arXiv:2307.07697](https://arxiv.org/abs/2307.07697) | Relation selection uses embedding similarity between question and relation name instead of an LLM. |
| pcst | G-Retriever, [arXiv:2402.07630](https://arxiv.org/abs/2402.07630) | Retrieval step only, no GNN and no LLM; prizes sit on edges (approximation). |

The PathRAG implementation was written from the paper and does not use code from the original repository.

## Fairness rules

- Every algorithm starts from the same seeds, the topic entities given by the dataset.
- Budgets are k in {1, 5, 10, 20}; a retriever records its top 20 paths.
- Ties are broken by the harness: score descending, then path length ascending, then node id tuple.
- An answer counts only when it is the end node of a path. Zero-length paths are dropped from the ranking.
- Paths have at most 3 hops.
- At most 10,000 nodes are expanded per query; hitting the cap is reported in the `capped` column.
- Hyperparameters are tuned on the validation split only. Defaults and sources are in each retriever's module docstring.

## Datasets

| Dataset | Source | License |
| --- | --- | --- |
| MetaQA | [yuyuz/MetaQA](https://github.com/yuyuz/MetaQA) | Creative Commons, per its README |
| WebQSP | Hugging Face `rmanluo/RoG-webqsp` | The dataset card states no license |

Both are downloaded on demand and are not redistributed here. The original MetaQA Google Drive folder returned HTTP 401 on 2026-11-24, so the loader falls back to the Hugging Face dataset `camazlucas/MetaQA` pinned at revision `f83854092893610686ab19f2038c039afa9bc643` (it ships `kb/kb.txt` instead of `kb.txt`). Its `kb.txt` has the same SHA-256 (`9b4d3b3066266075ad1577b4beed68e6e112e76e1a29ed2cb00a2896d2b792af`) as the `kb.txt` in the independent copy [brandenkang/MetaQA_PBG](https://github.com/brandenkang/MetaQA_PBG). The question files were checked by line count against the original README table only; they were not byte-compared with the original.

## Adding an algorithm

Add one file under `src/relpath_bench/retrievers/`, decorate the class with `@register`, give it a `name` and a `needs_embedder` flag, and implement `retrieve(g, q, k, limits)`. Import it in `retrievers/__init__.py`; `relpath-bench list` then shows it and `report` requires its results.

## License

Apache-2.0
