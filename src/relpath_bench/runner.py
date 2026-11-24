"""Run retrievers over a dataset, writing per-query jsonl with resume."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from relpath_bench import registry
from relpath_bench.datasets import LoadStats, load
from relpath_bench.harness import rank
from relpath_bench.metrics import QueryRecord, score_record, summarize
from relpath_bench.types import Limits

BATCH = 64
EMBED_HINT = "pip install relpath-bench[embed]"


def run_dir(out: Path, dataset: str, split: str, sample: int | None, seed: int) -> Path:
    tag = f"{split}-full" if sample is None else f"{split}-sample{sample}-seed{seed}"
    return Path(out) / dataset / tag


def _import_retrievers() -> None:
    import relpath_bench.retrievers  # noqa: F401  (registers every retriever)


def _unavailable(name: str, embedder) -> bool:
    cls = registry._REGISTRY[name]
    if getattr(cls, "needs_embedder", False) and embedder is None:
        return True
    return name == "pcst" and importlib.util.find_spec("pcst_fast") is None


def _read_done(path: Path) -> set[str]:
    """Return finished qids; drop everything from the first unparsable line on."""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    good: list[bytes] = []
    done: set[str] = set()
    for line in raw.splitlines(keepends=True):
        try:
            done.add(json.loads(line)["qid"])
        except (ValueError, KeyError, TypeError):
            break
        good.append(line if line.endswith(b"\n") else line + b"\n")
    clean = b"".join(good)
    if clean != raw:
        path.write_bytes(clean)
    return done


def _evaluate(retriever, name: str, ex, limits: Limits) -> str:
    t0 = time.perf_counter()
    res = retriever.retrieve(ex.graph, ex.query, 20, limits)
    ranked = rank(res.paths, 20)
    latency_ms = (time.perf_counter() - t0) * 1000
    rec = score_record(
        ex.graph, ex.query, ranked, algorithm=name, latency_ms=latency_ms, capped=res.capped
    )
    return rec.to_json()


# --- parallel worker state -------------------------------------------------

_W: dict = {}


def _init_worker(dataset, split, sample, seed, data_dir, embedder, limits) -> None:
    import warnings

    _import_retrievers()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _W["examples"] = {
            e.query.qid: e for e in load(dataset, split, sample, seed=seed, data_dir=data_dir)
        }
    _W.update(embedder=embedder, limits=limits, retrievers={})


def _work(name: str, qids: list[str]) -> list[str]:
    r = _W["retrievers"].get(name)
    if r is None:
        r = _W["retrievers"][name] = registry.make(name, embedder=_W["embedder"])
    return [_evaluate(r, name, _W["examples"][q], _W["limits"]) for q in qids]


def run(
    dataset: str,
    split: str,
    algorithms: list[str],
    sample: int | None,
    seed: int,
    out: Path,
    workers: int = 1,
    embedder=None,
    data_dir: Path | None = None,
    limits: Limits = Limits(),
) -> Path:
    _import_retrievers()
    d = run_dir(out, dataset, split, sample, seed)
    d.mkdir(parents=True, exist_ok=True)

    active = [a for a in algorithms if not _unavailable(a, embedder)]
    skipped = sorted(set(algorithms) - set(active))
    if skipped:
        print(f"skipped {', '.join(skipped)}: {EMBED_HINT}", file=sys.stderr)

    stats = LoadStats()
    examples = list(load(dataset, split, sample, seed=seed, data_dir=data_dir, stats=stats))
    pending: dict[str, list] = {}
    for name in active:
        done = _read_done(d / f"{name}.jsonl")
        pending[name] = [e for e in examples if e.query.qid not in done]

    if workers > 1 and any(pending.values()):
        _run_parallel(d, pending, workers, dataset, split, sample, seed, data_dir, embedder, limits)
    else:
        for name, todo in pending.items():
            if not todo:
                continue
            r = registry.make(name, embedder=embedder)
            with open(d / f"{name}.jsonl", "a", encoding="utf-8") as f:
                for ex in todo:
                    f.write(_evaluate(r, name, ex, limits) + "\n")
                    f.flush()

    records: list[QueryRecord] = []
    for name in active:
        for line in (d / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            records.append(QueryRecord.from_json(line))
    summary = {
        "dataset": dataset,
        "split": split,
        "full": sample is None,
        "sample": sample,
        "seed": seed,
        "algorithms": active,
        "load_stats": asdict(stats),
        "rows": summarize(records, by_hops=(dataset == "metaqa")),
    }
    (d / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return d


def _run_parallel(d, pending, workers, dataset, split, sample, seed, data_dir, embedder, limits):
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(dataset, split, sample, seed, data_dir, embedder, limits),
    ) as pool:
        futures = []
        for name, todo in pending.items():
            qids = [e.query.qid for e in todo]
            for i in range(0, len(qids), BATCH):
                futures.append((name, pool.submit(_work, name, qids[i : i + BATCH])))
        files = {n: open(d / f"{n}.jsonl", "a", encoding="utf-8") for n in pending}
        try:
            from concurrent.futures import as_completed

            by_future = {f: n for n, f in futures}
            for fut in as_completed(by_future):
                f = files[by_future[fut]]
                for line in fut.result():
                    f.write(line + "\n")
                f.flush()
        finally:
            for f in files.values():
                f.close()
