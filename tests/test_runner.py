import json
import sys

import pytest

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
    assert len(lines) == 1 and summary["full"] is True
    assert summary["algorithms"] == ["bfs", "ppr", "pathrag"]
    assert {r["algorithm"] for r in summary["rows"]} == {"bfs", "ppr", "pathrag"}
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert (d / "bfs.jsonl").read_text().splitlines() == lines


def test_truncated_line_rerun(metaqa_fixture_dir, tmp_path):
    d = run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    good = (d / "bfs.jsonl").read_text()
    (d / "bfs.jsonl").write_text(good[: len(good) // 2])
    run("metaqa", "test", ["bfs"], None, 0, tmp_path, data_dir=metaqa_fixture_dir)
    assert len((d / "bfs.jsonl").read_text().splitlines()) == 1
    json.loads((d / "bfs.jsonl").read_text())


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
