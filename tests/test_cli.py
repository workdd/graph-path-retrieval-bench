import json

import pytest

from relpath_bench.cli import main


def test_cli_list(capsys):
    main(["list"])
    assert capsys.readouterr().out.split() == ["beam", "bfs", "pathrag", "pcst", "ppr"]


def test_cli_run(metaqa_fixture_dir, tmp_path, capsys):
    rc = main(["run", "--dataset", "metaqa", "--algorithms", "bfs,ppr", "--full",
               "--out", str(tmp_path), "--data-dir", str(metaqa_fixture_dir)])
    assert rc == 0
    d = tmp_path / "metaqa" / "test-full"
    assert json.loads((d / "summary.json").read_text())["algorithms"] == ["bfs", "ppr"]
    assert str(d) in capsys.readouterr().out


def test_cli_run_sample_dir_and_unknown(metaqa_fixture_dir, tmp_path, capsys):
    main(["run", "--dataset", "metaqa", "--algorithms", "bfs", "--sample", "5", "--seed", "2",
          "--out", str(tmp_path), "--data-dir", str(metaqa_fixture_dir)])
    assert (tmp_path / "metaqa" / "test-sample5-seed2" / "bfs.jsonl").exists()
    assert main(["run", "--dataset", "metaqa", "--algorithms", "nope", "--out", str(tmp_path)]) == 2


def test_cli_run_all_without_embed_extra(metaqa_fixture_dir, tmp_path, capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    main(["run", "--dataset", "metaqa", "--full", "--out", str(tmp_path),
          "--data-dir", str(metaqa_fixture_dir)])
    assert "skipped beam, pcst: pip install relpath-bench[embed]" in capsys.readouterr().err


def test_cli_report_delegates(monkeypatch, tmp_path):
    import sys, types

    called = []
    mod = types.ModuleType("relpath_bench.report")
    mod.render = lambda out, readme: called.append((out, readme))
    monkeypatch.setitem(sys.modules, "relpath_bench.report", mod)
    main(["report", "--out", str(tmp_path), "--readme", "R.md"])
    assert called and called[0][0] == tmp_path
