"""Integration tests for ScarTissue MCP Server tools and 5-scenario benchmark."""

import inspect
from pathlib import Path

from benchmarks.evaluate_5_scenarios import run_benchmark
from benchmarks.evaluate_real_oss_commits import evaluate_real_oss_commits
from scar_tissue.server import _get_store, check_scar_tissue, record_dead_end


def test_mcp_tools_record_and_check(tmp_path: Path, monkeypatch):
    storage = tmp_path / ".scartissue"
    monkeypatch.setenv("SCAR_TISSUE_DIR", str(storage))

    # Security guardrail: `storage_dir` must NOT be exposed to LLM clients in tool schemas
    assert "storage_dir" not in inspect.signature(record_dead_end).parameters
    assert "storage_dir" not in inspect.signature(check_scar_tissue).parameters

    rec_res = record_dead_end(
        target_file="trainer/ddp_runner.py",
        target_symbol="sync_shard_metrics",
        attempted_change="if rank == 0: dist.barrier()",
        error_root_cause="NCCL collective timeout deadlock",
        dead_end_approach="在 rank==0 分支调用 dist.barrier()",
        safe_pivot="全卡无条件 all_reduce 归约",
        raw_context_chars=1800,
    )
    assert rec_res["status"] == "recorded"
    assert rec_res["alert_char_length"] <= 50
    assert "dist.barrier" not in rec_res["alert_50c"]

    chk_res = check_scar_tissue(
        target_file="trainer/ddp_runner.py",
        target_symbol="sync_shard_metrics",
        planned_change="Add barrier in rank 0 evaluation",
    )
    assert chk_res["has_scars"] is True
    assert len(chk_res["alerts"]) == 1
    assert len(chk_res["alerts"][0]) <= 50
    assert chk_res["tokens_saved_estimate"] > 400


def test_get_store_walks_up_to_git_root_for_relative_target(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SCAR_TISSUE_DIR", raising=False)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    subdir = repo / "src" / "pkg"
    subdir.mkdir(parents=True)
    monkeypatch.chdir(subdir)

    store = _get_store(target_file="src/pkg/module.py")
    assert store.storage_dir == (repo / ".scartissue").resolve()
    assert store.repo_root == repo.resolve()


def test_five_real_scenarios_benchmark_end_to_end():
    report = run_benchmark()
    summary = report["summary"]
    assert summary["interception_recall"] == "5/5 (100%)"
    assert summary["unrelated_false_positives"] == "0/3 (0%)"
    assert summary["all_alerts_under_50_chars"] is True
    assert summary["max_alert_chars"] <= 50
    assert summary["avg_pink_elephant_risk_pivot"] == 0.0


def test_real_oss_commits_benchmark_end_to_end():
    oss_report = evaluate_real_oss_commits()
    summary = oss_report["summary"]
    assert summary["interception_recall"] == "5/5 (100%)"
    assert summary["all_alerts_under_50_chars"] is True
    assert summary["max_alert_chars"] <= 50
    assert summary["avg_pink_elephant_risk_pivot"] == 0.0
