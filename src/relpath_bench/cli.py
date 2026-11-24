"""Command line interface: list, run, report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from relpath_bench import registry


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relpath-bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list retriever names")

    r = sub.add_parser("run", help="run retrievers over a dataset")
    r.add_argument("--dataset", choices=["metaqa", "webqsp"], required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--algorithms", default="all", help="all or comma separated names")
    r.add_argument("--full", action="store_true", help="use every question")
    r.add_argument("--sample", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--out", type=Path, default=Path("results"))
    r.add_argument("--data-dir", type=Path, default=None)

    rp = sub.add_parser("report", help="render result tables into the README")
    rp.add_argument("--out", type=Path, default=Path("results"))
    rp.add_argument("--readme", type=Path, default=Path("README.md"))
    return p


def main(argv: list[str] | None = None) -> int:
    import relpath_bench.retrievers  # noqa: F401  (registers every retriever)

    args = _parser().parse_args(argv)
    if args.cmd == "list":
        print("\n".join(registry.names()))
        return 0
    if args.cmd == "report":
        from relpath_bench.report import render

        render(args.out, args.readme)
        return 0

    algos = registry.names() if args.algorithms == "all" else args.algorithms.split(",")
    unknown = [a for a in algos if a not in registry.names()]
    if unknown:
        print(f"unknown algorithms: {', '.join(unknown)}", file=sys.stderr)
        return 2
    embedder = None
    if any(getattr(registry._REGISTRY[a], "needs_embedder", False) for a in algos):
        try:
            from relpath_bench.embed import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        except ImportError:
            embedder = None
    from relpath_bench.runner import run

    d = run(
        args.dataset, args.split, algos, None if args.full else args.sample, args.seed,
        args.out, workers=args.workers, embedder=embedder, data_dir=args.data_dir,
    )
    print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
