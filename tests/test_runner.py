import json
import sys

import pytest

from relpath_bench import registry
from relpath_bench.types import RetrievalResult

from relpath_bench.runner import run, run_dir


def _strip(path):
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        d.pop("latency_ms")
        out.append(d)
    return sorted(out, key=lambda d: d["qid"])


def test_run_dir_names(tmp_path):
    assert run_dir(tmp_path, "metaqa", "test", None, 0) == tmp_path / "metaqa" / "test-full"
    assert (
        run_dir(tmp_path, "webqsp", "test", 5, 1) == tmp_path / "webqsp" / "test-sample5-seed1"
    )


def test_run_writes_and_resumes(metaqa_fixture_dir, tmp_path):
    d = run("metaqa", "test", ["bfs", "ppr", "pathrag"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    lines = (d / "bfs.jsonl").read_text().splitlines()
    summary = json.loads((d / "summary.json").read_text())
    assert len(lines) == 3 and summary["full"] is True
    assert summary["algorithms"] == ["bfs", "pathrag", "ppr"]
    assert {r["algorithm"] for r in summary["rows"]} == {"bfs", "ppr", "pathrag"}
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert (d / "bfs.jsonl").read_text().splitlines() == lines


def test_truncated_line_rerun(metaqa_fixture_dir, tmp_path):
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    good = (d / "bfs.jsonl").read_text()
    (d / "bfs.jsonl").write_text(good[: len(good) // 2])
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert len((d / "bfs.jsonl").read_text().splitlines()) == 3


def test_deterministic_records(metaqa_fixture_dir, tmp_path):
    a = run("metaqa", "test", ["bfs", "ppr"], None, 0, tmp_path / "a", data_dir=metaqa_fixture_dir)
    b = run("metaqa", "test", ["bfs", "ppr"], None, 0, tmp_path / "b", data_dir=metaqa_fixture_dir)
    for n in ("bfs", "ppr"):
        assert _strip(a / f"{n}.jsonl") == _strip(b / f"{n}.jsonl")


def test_parallel_matches_serial(metaqa_fixture_dir, tmp_path):
    a = run("metaqa", "test", ["bfs", "pathrag"], None, 0, tmp_path / "a", data_dir=metaqa_fixture_dir)
    b = run("metaqa", "test", ["bfs", "pathrag"], None, 0, tmp_path / "b", workers=2,
            data_dir=metaqa_fixture_dir)
    for n in ("bfs", "pathrag"):
        assert _strip(a / f"{n}.jsonl") == _strip(b / f"{n}.jsonl")


def test_skips_embedder_algorithms(metaqa_fixture_dir, tmp_path, capsys):
    d = run("metaqa", "test", ["bfs", "pcst", "beam"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert capsys.readouterr().err.strip() == "skipped beam, pcst: pip install relpath-bench[embed]"
    assert not (d / "beam.jsonl").exists()
    assert json.loads((d / "summary.json").read_text())["algorithms"] == ["bfs"]


def test_missing_pcst_fast_skips(metaqa_fixture_dir, tmp_path, capsys, monkeypatch):
    from relpath_bench.embed import FixedEmbedder

    monkeypatch.setitem(sys.modules, "pcst_fast", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda n: None)
    run("metaqa", "test", ["pcst"], None, 0, tmp_path, embedder=FixedEmbedder({}),
        data_dir=metaqa_fixture_dir)
    assert "skipped pcst" in capsys.readouterr().err


def test_partial_resume_keeps_good_prefix(metaqa_fixture_dir, tmp_path):
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    lines = (d / "bfs.jsonl").read_text().splitlines()
    (d / "bfs.jsonl").write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2])
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    out = (d / "bfs.jsonl").read_text().splitlines()
    qids = [json.loads(x)["qid"] for x in out]
    assert len(out) == 3 and len(set(qids)) == 3 and out[0] == lines[0]


def test_summary_covers_all_jsonl_in_dir(metaqa_fixture_dir, tmp_path):
    run("metaqa", "test", ["bfs", "ppr"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    s = json.loads((d / "summary.json").read_text())
    assert s["algorithms"] == ["bfs", "ppr"]
    assert {r["algorithm"] for r in s["rows"]} == {"bfs", "ppr"}


def test_summary_ignores_unregistered_jsonl(metaqa_fixture_dir, tmp_path):
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    (d / "stray.jsonl").write_text("garbage\n")
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert json.loads((d / "summary.json").read_text())["algorithms"] == ["bfs"]


def test_zero_example_dataset(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    (root / "kb.txt").write_text("a|r|b\n")
    for hop in (1, 2, 3):
        (root / f"{hop}-hop" / "vanilla").mkdir(parents=True)
        (root / f"{hop}-hop" / "vanilla" / "qa_test.txt").write_text("")
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path / "o", data_dir=root)
    s = json.loads((d / "summary.json").read_text())
    assert s["rows"] == [] and s["algorithms"] == []


def test_parallel_small_batches(metaqa_fixture_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("relpath_bench.runner.BATCH", 1)
    a = run("metaqa", "test", ["bfs"], None, 0, tmp_path / "a", data_dir=metaqa_fixture_dir)
    b = run("metaqa", "test", ["bfs"], None, 0, tmp_path / "b", workers=2,
            data_dir=metaqa_fixture_dir)
    assert _strip(a / "bfs.jsonl") == _strip(b / "bfs.jsonl")


@pytest.fixture
def boom_retriever():
    import multiprocessing

    saved = dict(registry._REGISTRY)

    @registry.register
    class Boom:
        name = "boom"
        needs_embedder = False

        def retrieve(self, g, q, k, limits):
            raise RuntimeError("boom")

    import relpath_bench.runner as runner

    old = runner._MP_CONTEXT
    runner._MP_CONTEXT = multiprocessing.get_context("fork")  # workers must see the registry
    yield
    runner._MP_CONTEXT = old
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_worker_exception_propagates(metaqa_fixture_dir, tmp_path, boom_retriever, monkeypatch):
    monkeypatch.setattr("relpath_bench.runner.BATCH", 1)
    with pytest.raises(RuntimeError, match="boom"):
        run("metaqa", "test", ["boom"], None, 0, tmp_path, workers=2, data_dir=metaqa_fixture_dir)
