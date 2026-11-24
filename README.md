# relpath-bench

A small benchmark that compares relational path retrieval algorithms on knowledge graphs under one harness, one budget and one set of scoring rules.

## Results

<!-- RESULTS:START -->
Results pending: run `relpath-bench run --dataset metaqa --full` and `relpath-bench report`.
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
- Paths have at most 4 hops.
- At most 10,000 nodes are expanded per query; hitting the cap is reported in the `capped` column.
- Hyperparameters are tuned on the validation split only. Defaults and sources are in each retriever's module docstring.

## Datasets

| Dataset | Source | License |
| --- | --- | --- |
| MetaQA | [yuyuz/MetaQA](https://github.com/yuyuz/MetaQA) | Creative Commons, per its README |
| WebQSP | Hugging Face `rmanluo/RoG-webqsp` | The dataset card states no license |

Both are downloaded on demand and are not redistributed here. If the MetaQA Google Drive folder cannot be fetched (it answers HTTP 401 to anonymous listing), the loader falls back to the Hugging Face mirror `camazlucas/MetaQA`, which carries the same vanilla files (`kb/kb.txt` instead of `kb.txt`).

## Adding an algorithm

Add one file under `src/relpath_bench/retrievers/`, decorate the class with `@register`, give it a `name` and a `needs_embedder` flag, and implement `retrieve(g, q, k, limits)`. Import it in `retrievers/__init__.py`; `relpath-bench list` then shows it and `report` requires its results.

## License

Apache-2.0
