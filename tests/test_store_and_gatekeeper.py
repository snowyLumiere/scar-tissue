"""Unit tests for ScarStore SQLite + JSONL bidirectional sync, path normalization, and Gatekeeper filtering."""

import json
from pathlib import Path
import pytest
from scar_tissue.store import ScarStore, normalize_rel_path


def test_store_record_check_and_human_editability(tmp_path: Path):
    repo_root = tmp_path / "my_project"
    repo_root.mkdir()
    store = ScarStore(repo_root / ".scartissue", repo_root=repo_root)

    rec = store.record_dead_end(
        target_file="storage/async_sqlite.py",
        target_symbol="write_audit_event",
        attempted_change="Wrap connection with asyncio.Lock()",
        error_root_cause="sqlite3.OperationalError: database is locked",
        dead_end_approach="用 asyncio.Lock() 包装跨线程连接",
        safe_pivot="开启WAL日志并由单写协程队列串行落盘",
        raw_context_chars=2400,
    )

    assert len(rec.alert_50c) <= 50
    assert (repo_root / ".scartissue" / "scars.db").exists()
    assert (repo_root / ".scartissue" / "scars.jsonl").exists()

    # 1. Query using absolute file path (as Claude Code / Cursor does) -> must normalize and intercept!
    abs_query_path = str(repo_root / "storage" / "async_sqlite.py")
    hit = store.check_scar_tissue(
        target_file=abs_query_path,
        target_symbol="write_audit_event",
        planned_change="Add asyncio lock around sqlite write",
    )
    assert hit.has_scars is True
    assert hit.target_file == "storage/async_sqlite.py"
    assert hit.matches[0].match_reason.startswith("exact_file")
    assert len(hit.alerts[0]) <= 50
    assert hit.tokens_saved_estimate > 500

    # 2. Unrelated file -> must NOT trigger false alarm
    miss = store.check_scar_tissue(
        target_file="frontend/Navbar.tsx",
        planned_change="Adjust logo margin",
    )
    assert miss.has_scars is False
    assert miss.alerts == []

    # 3. Human editability via API deletion
    deleted = store.delete_scar(rec.scar_id)
    assert deleted is True
    after_del = store.check_scar_tissue(target_file="storage/async_sqlite.py")
    assert after_del.has_scars is False


def test_bidirectional_jsonl_human_text_editor_sync(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = ScarStore(repo_root / ".scartissue", repo_root=repo_root)

    r1 = store.record_dead_end(
        target_file="a/foo.py",
        attempted_change="bad a",
        error_root_cause="crash a",
        dead_end_approach="dead end a",
        safe_pivot="pivot a",
    )
    r2 = store.record_dead_end(
        target_file="b/bar.py",
        attempted_change="bad b",
        error_root_cause="crash b",
        dead_end_approach="dead end b",
        safe_pivot="pivot b",
    )
    assert len(store.list_all_scars()) == 2

    # Simulate a human opening `.scartissue/scars.jsonl` in a text editor:
    # - deleting `r2`
    # - customizing `r1.alert_50c` to a hand-tuned redline rule
    jsonl_path = repo_root / ".scartissue" / "scars.jsonl"
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [obj for obj in lines if obj["scar_id"] == r1.scar_id]
    kept[0]["alert_50c"] = "⛔忌手工修订死路(并发崩溃)→用人工审核正路"
    jsonl_path.write_text(json.dumps(kept[0], ensure_ascii=False) + "\n", encoding="utf-8")

    # Next query must honor the human's JSONL edit and NOT overwrite it with old SQLite data!
    scars_after = store.list_all_scars()
    assert len(scars_after) == 1
    assert scars_after[0].scar_id == r1.scar_id
    assert scars_after[0].alert_50c == "⛔忌手工修订死路(并发崩溃)→用人工审核正路"

    chk_b = store.check_scar_tissue(target_file="b/bar.py")
    assert chk_b.has_scars is False

    chk_a = store.check_scar_tissue(target_file="a/foo.py")
    assert chk_a.has_scars is True
    assert chk_a.alerts[0] == "⛔忌手工修订死路(并发崩溃)→用人工审核正路"


def test_ambiguous_basename_in_different_dirs_does_not_collide(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = ScarStore(repo_root / ".scartissue", repo_root=repo_root)

    store.record_dead_end(
        target_file="trainer/utils.py",
        target_symbol="ddp_sync_helper",
        attempted_change="dist.barrier() in rank 0",
        error_root_cause="NCCL deadlock",
        dead_end_approach="rank 0 dist.barrier()",
        safe_pivot="all_reduce",
    )

    # Editing `frontend/utils.py` (same basename `utils.py`, different module dir) must NOT false-alarm!
    res = store.check_scar_tissue(
        target_file="frontend/utils.py",
        target_symbol="format_currency",
        planned_change="Format USD string with 2 decimals",
    )
    assert res.has_scars is False


def test_short_english_symbol_does_not_false_match_prose(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = ScarStore(repo_root / ".scartissue", repo_root=repo_root)

    store.record_dead_end(
        target_file="trainer/runner.py",
        target_symbol="run",
        attempted_change="dist.barrier() deadlock",
        error_root_cause="NCCL lock deadlock",
        dead_end_approach="rank 0 dist.barrier()",
        safe_pivot="all_reduce",
    )

    # Different file whose `planned_change` contains the English word 'run' + 'lock' keywords
    # -> Must NOT trigger cross_file_symbol_guard just because `def run` is a 3-letter word!
    res = store.check_scar_tissue(
        target_file="scripts/deploy.py",
        planned_change="run deployment check with lock file",
    )
    assert res.has_scars is False


def test_malformed_jsonl_fails_closed_without_data_loss(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = ScarStore(repo_root / ".scartissue", repo_root=repo_root)

    r1 = store.record_dead_end("a.py", "try a", "err a", "dead a", "pivot a")
    r2 = store.record_dead_end("b.py", "try b", "err b", "dead b", "pivot b")
    jsonl_path = repo_root / ".scartissue" / "scars.jsonl"
    original_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(original_lines) == 2

    # Introduce a single-character JSON syntax typo on line 2
    corrupted = original_lines[0] + "\n" + original_lines[1][:-1] + ",}\n"
    jsonl_path.write_text(corrupted, encoding="utf-8")

    # Must fail closed with ValueError citing line 2 and preserve both JSONL and SQLite
    with pytest.raises(ValueError, match="line 2"):
        store.list_all_scars()
    with pytest.raises(ValueError, match="line 2"):
        store.check_scar_tissue("a.py")

    # Corrupted JSONL must NOT have been overwritten/truncated
    assert jsonl_path.read_text(encoding="utf-8") == corrupted

    # Restoring the valid JSONL recovers both records cleanly
    jsonl_path.write_text("\n".join(original_lines) + "\n", encoding="utf-8")
    recovered = {s.scar_id for s in store.list_all_scars()}
    assert recovered == {r1.scar_id, r2.scar_id}


def test_read_only_checks_do_not_create_store_or_rewrite_jsonl(tmp_path: Path):
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    store = ScarStore(empty_repo / ".scartissue", repo_root=empty_repo)

    assert store.list_all_scars() == []
    assert store.check_scar_tissue("app/main.py").has_scars is False
    assert not (empty_repo / ".scartissue").exists()

    # After recording a scar, read-only check hits must update SQLite telemetry
    # without rewriting/dirtying the tracked `scars.jsonl` file.
    store.record_dead_end("app/main.py", "bad", "crash", "bad api", "safe api")
    jsonl_path = empty_repo / ".scartissue" / "scars.jsonl"
    before_bytes = jsonl_path.read_bytes()

    hit = store.check_scar_tissue("app/main.py")
    assert hit.has_scars is True
    assert hit.matches[0].scar.hit_count == 1
    assert jsonl_path.read_bytes() == before_bytes
    assert store.get_stats()["total_interceptions"] == 1


def test_normalize_rel_path_scrubs_windows_and_posix_user_homes(tmp_path: Path):
    assert normalize_rel_path(r"C:\Users\alice\secret_proj\db.py", tmp_path) == "secret_proj/db.py"
    assert normalize_rel_path(r"C:\Users\alice\db.py", tmp_path) == "db.py"
    assert normalize_rel_path("/Users/bob/app.py", tmp_path) == "app.py"
    assert "alice" not in normalize_rel_path(r"C:\Users\alice\db.py", tmp_path)
    assert "bob" not in normalize_rel_path("/Users/bob/app.py", tmp_path)


