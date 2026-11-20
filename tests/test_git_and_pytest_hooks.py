"""Unit tests for Git Revert/Reset Hook Extractor and 3-Strike Pytest Crash-Loop Tracker."""

import os
import subprocess
from pathlib import Path
import pytest
from scar_tissue import __version__
from scar_tissue.cli import main as cli_main
from scar_tissue.hooks import HOOK_MARKER, GitScarExtractor, ThreeStrikeTestTracker
from scar_tissue.store import ScarStore


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


def test_git_revert_auto_extraction_and_non_destructive_hook_install(tmp_path: Path):
    repo = tmp_path / "demo_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")

    # Pre-create an existing user post-commit hook to verify we don't clobber it
    existing_hook = repo / ".git" / "hooks" / "post-commit"
    existing_hook.parent.mkdir(parents=True, exist_ok=True)
    existing_hook.write_text("#!/bin/sh\necho 'existing user hook'\n", encoding="utf-8")

    store = ScarStore(repo / ".scartissue", repo_root=repo)
    extractor = GitScarExtractor(repo, store)

    hooks = extractor.install_git_hooks()
    assert len(hooks) == 3  # post-commit, post-rewrite, post-checkout
    for h in hooks:
        assert h.exists()
        assert os.access(h, os.X_OK)
    # Existing user hook line must still be intact!
    post_commit_content = existing_hook.read_text(encoding="utf-8")
    assert "echo 'existing user hook'" in post_commit_content
    assert HOOK_MARKER in post_commit_content

    target = repo / "engine" / "worker.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def process_item(x):\n    return x + 1\n", encoding="utf-8")
    _git(repo, "add", "engine/worker.py")
    _git(repo, "commit", "-m", "feat: add worker")

    # Commit a normal feature that mentions the word 'reset' (e.g. 'feat: add password reset')
    # -> Must NOT false-trigger as a dead-end revert!
    auth_file = repo / "engine" / "auth.py"
    auth_file.write_text("def password_reset():\n    return True\n", encoding="utf-8")
    _git(repo, "add", "engine/auth.py")
    _git(repo, "commit", "-m", "feat(auth): add password reset endpoint")
    assert extractor.inspect_commit("HEAD") == []

    # Commit bad change
    target.write_text("def process_item(x):\n    proc.wait()\n    return x + 1\n", encoding="utf-8")
    _git(repo, "add", "engine/worker.py")
    _git(repo, "commit", "-m", "refactor: call proc.wait() directly")

    # Test real native `git revert HEAD --no-edit` (with default "This reverts commit <sha>.")
    _git(repo, "revert", "HEAD", "--no-edit")

    records = extractor.inspect_commit("HEAD")
    assert len(records) == 1
    rec = records[0]
    assert rec.target_file == "engine/worker.py"
    assert rec.target_symbol == "process_item"
    # Must NOT mistake "This reverts commit <sha>." for error_root_cause!
    assert "this reverts commit" not in rec.error_root_cause.lower()
    assert len(rec.alert_50c) <= 50
    assert "proc.wait" not in rec.alert_50c


def test_git_reset_reflog_capture(tmp_path: Path):
    repo = tmp_path / "reset_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")

    store = ScarStore(repo / ".scartissue", repo_root=repo)
    extractor = GitScarExtractor(repo, store)

    f = repo / "trainer" / "sync.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("def step():\n    return all_reduce()\n", encoding="utf-8")
    _git(repo, "add", "trainer/sync.py")
    _git(repo, "commit", "-m", "feat: initial step")

    # Commit a broken change and then `git reset --hard HEAD~1`
    f.write_text("def step():\n    dist.barrier()\n    return all_reduce()\n", encoding="utf-8")
    _git(repo, "add", "trainer/sync.py")
    _git(repo, "commit", "-m", "perf: add dist.barrier() in step")
    _git(repo, "reset", "--hard", "HEAD~1")

    reset_scars = extractor.inspect_latest_reflog_reset()
    assert len(reset_scars) == 1
    assert reset_scars[0].target_file == "trainer/sync.py"
    assert reset_scars[0].trigger_source.value == "git_reset"
    assert len(reset_scars[0].alert_50c) <= 50
    assert "dist.barrier" not in reset_scars[0].alert_50c


def test_three_strike_per_file_isolation_and_cli_runner(tmp_path: Path):
    store = ScarStore(tmp_path / ".scartissue", repo_root=tmp_path)
    tracker = ThreeStrikeTestTracker(tmp_path, store, strike_threshold=3)

    # 2 failures in `a.py` + 1 failure in `b.py` + 1 pass in `c.py` -> must NOT cross-contaminate!
    assert tracker.record_test_attempt(False, "a.py", "E RuntimeError: boom a1", "bad a") is None
    assert tracker.record_test_attempt(False, "a.py", "E RuntimeError: boom a2", "bad a") is None
    assert tracker.record_test_attempt(False, "b.py", "E RuntimeError: boom b1", "bad b") is None
    assert tracker.record_test_attempt(True, "c.py", working_fix_pivot="good c") is None
    assert len(store.list_all_scars()) == 0

    # 3rd failure in `a.py` followed by pass in `a.py` -> MUST create a scar for `a.py`!
    assert tracker.record_test_attempt(False, "a.py", "E RuntimeError: boom a3", "bad a") is None
    recovered_a = tracker.record_test_attempt(True, "a.py", working_fix_pivot="good a")
    assert recovered_a is not None
    assert recovered_a.target_file == "a.py"
    assert len(recovered_a.alert_50c) <= 50

    # Test CLI `scar-tissue test` subcommand with python -c exit codes
    for _ in range(3):
        rc = cli_main(
            [
                "test",
                "--repo",
                str(tmp_path),
                "--file",
                "inference/kv.py",
                "--attempted",
                "对 32K 序列调用 .unsqueeze(1).expand()",
                "--",
                "python3",
                "-c",
                "import sys; sys.stderr.write('RuntimeError: CUDA out of memory\\n'); sys.exit(1)",
            ]
        )
        assert rc == 1

    rc_pass = cli_main(
        [
            "test",
            "--repo",
            str(tmp_path),
            "--file",
            "inference/kv.py",
            "--pivot",
            "启用 SDPA is_causal 免实例化",
            "--",
            "python3",
            "-c",
            "import sys; sys.exit(0)",
        ]
    )
    assert rc_pass == 0
    chk = store.check_scar_tissue("inference/kv.py")
    assert chk.has_scars is True


def test_install_git_hooks_rejects_non_git_and_handles_worktree_and_non_shell_hooks(
    tmp_path: Path,
):
    # 1. Non-git directory must fail cleanly without creating a fake `.git/`
    non_git = tmp_path / "plain_dir"
    non_git.mkdir()
    extractor_ng = GitScarExtractor(non_git)
    with pytest.raises(ValueError, match="Not a git repository"):
        extractor_ng.install_git_hooks()
    assert not (non_git / ".git").exists()
    assert cli_main(["install-hooks", "--repo", str(non_git)]) == 1

    # 2. Real git repo with a non-shell hook (Python) -> refuse without force, replace with force
    repo = tmp_path / "main_repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")

    py_hook = repo / ".git" / "hooks" / "post-commit"
    py_content = "#!/usr/bin/env python3\nprint('custom hook')\n"
    py_hook.write_text(py_content, encoding="utf-8")

    extractor = GitScarExtractor(repo)
    with pytest.raises(ValueError, match="non-shell interpreter"):
        extractor.install_git_hooks()
    # Original Python hook must remain intact
    assert py_hook.read_text(encoding="utf-8") == py_content

    # With force=True, replaces cleanly with valid shell hook
    hooks = extractor.install_git_hooks(force=True)
    assert len(hooks) == 3
    assert py_hook.read_text(encoding="utf-8").startswith("#!/bin/sh\n")
    assert HOOK_MARKER in py_hook.read_text(encoding="utf-8")

    # 3. Git worktree (`.git` is a file, not a directory) must resolve hooks via git-path
    wt = tmp_path / "wt_branch"
    _git(repo, "worktree", "add", "-b", "wt-test", str(wt))
    assert (wt / ".git").is_file()
    wt_hooks = GitScarExtractor(wt).install_git_hooks()
    assert len(wt_hooks) == 3
    assert all(h.exists() for h in wt_hooks)

    # 4. Custom `core.hooksPath` is honored
    custom_hooks = repo / ".custom-hooks"
    _git(repo, "config", "core.hooksPath", ".custom-hooks")
    ch_hooks = GitScarExtractor(repo).install_git_hooks()
    assert all(h.parent == custom_hooks.resolve() for h in ch_hooks)


def test_cli_version_and_scar_tissue_dir_env(tmp_path: Path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out

    custom_store = tmp_path / "external_store"
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    monkeypatch.setenv("SCAR_TISSUE_DIR", str(custom_store))

    assert cli_main(["list", "--repo", str(repo_dir)]) == 0
    assert not custom_store.exists()
    assert not (repo_dir / ".scartissue").exists()

    rc = cli_main(
        [
            "record",
            "--repo",
            str(repo_dir),
            "--file",
            "app/worker.py",
            "--attempted",
            "proc.wait()",
            "--cause",
            "pipe deadlock",
            "--dead-end",
            "proc.wait()",
            "--pivot",
            "proc.communicate()",
        ]
    )
    assert rc == 0
    assert (custom_store / "scars.jsonl").exists()
    assert not (repo_dir / ".scartissue").exists()

