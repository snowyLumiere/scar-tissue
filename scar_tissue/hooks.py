"""
Git Revert/Reset Hook Extractor & 3-Strike Test Crash-Loop Detector.

Automatically captures `{attempted_change, error_root_cause, dead_end_approach, safe_pivot}`
when:
1. A developer or agent executes `git revert` / `git reset` or commits a rollback.
2. A test suite fails >= 3 consecutive times and is finally fixed on a subsequent run.
"""

from __future__ import annotations

import json
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scar_tissue.models import ScarRecord, TriggerSource
from scar_tissue.store import ScarStore, normalize_rel_path

HOOK_MARKER = "# [SCAR-TISSUE-AUTO-HOOK]"


def _run_git(repo_root: Path, args: List[str]) -> str:
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        return ""


def parse_diff_hunk_metadata(diff_text: str) -> Tuple[str, List[str], List[str]]:
    """
    Extract (symbol_name, added_lines, removed_lines) from a unified git diff.
    """
    symbol = ""
    added: List[str] = []
    removed: List[str] = []

    for line in diff_text.splitlines():
        if line.startswith("@@"):
            m = re.search(
                r"@@.*@@\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|const)\s+([a-zA-Z0-9_]+)",
                line,
            )
            if m and not symbol:
                symbol = m.group(1)
        elif not line.startswith("+++") and not line.startswith("---"):
            if not symbol:
                m_ctx = re.search(
                    r"^[+\-\s]\s*(?:export\s+)?(?:async\s+)?(?:def|class|function)\s+([a-zA-Z0-9_]+)",
                    line,
                )
                if m_ctx:
                    symbol = m_ctx.group(1)
            if line.startswith("+"):
                clean = line[1:].strip()
                if clean and not clean.startswith("#") and not clean.startswith("//"):
                    added.append(clean)
            elif line.startswith("-"):
                clean = line[1:].strip()
                if clean and not clean.startswith("#") and not clean.startswith("//"):
                    removed.append(clean)

    return symbol, added, removed


def _pick_salient_code_line(lines: List[str]) -> str:
    """
    Pick the most informative code line from diff additions/removals, skipping
    trivial boilerplate like bare variable declarations, imports, or single identifiers.
    """
    if not lines:
        return ""
    for line in lines:
        s = line.strip()
        if len(s) < 6:
            continue
        if re.match(r"^(?:import\s|from\s|export\s+\{|pass$|return$|break$|continue$|[{}()\[\],;]+$)", s):
            continue
        if re.match(r"^(?:let|var|const)\s+[a-zA-Z0-9_:]+\s*=\s*(?:\[\]|\{\}|false|true|null|undefined);?$", s):
            continue
        # Prefer lines with method/function calls, await, or control-flow conditions
        if "(" in s or "await " in s or "==" in s or "new " in s:
            return s[:60]
    for line in lines:
        if len(line.strip()) >= 6:
            return line.strip()[:60]
    return lines[0].strip()[:60]


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _infer_cause_from_code(text: str) -> str:
    """Infer a concise engineering root-cause label when commit body has no explicit note."""
    low = text.lower()
    cjk = _has_cjk(text)
    if any(k in low for k in ("barrier", "lock", "mutex", "deadlock", "死锁")):
        return "并发同步死锁冲突" if cjk else "concurrency deadlock"
    if any(k in low for k in ("wait(", "pipe", "stdout", "communicate", "管道")):
        return "管道缓冲满阻塞死锁" if cjk else "pipe buffer deadlock"
    if any(k in low for k in ("useeffect", "setsearchparams", "setstate", "rerender")):
        return "状态重渲染死循环" if cjk else "infinite rerender loop"
    if any(k in low for k in ("expand(", "unsqueeze", "oom", "cuda", "显存")):
        return "高维张量实例化OOM" if cjk else "tensor expansion OOM"
    if any(k in low for k in ("sqlite", "database is locked", "transaction")):
        return "数据库并发写锁冲突" if cjk else "DB write lock conflict"
    if any(k in low for k in ("sandbox", "/projects", "alloweddirectories", "readdir")):
        return "破坏CLI沙箱目录边界" if cjk else "CLI sandbox violation"
    if any(k in low for k in ("mount_path", "sse", "route", "subpath")):
        return "子路径路由挂载冲突" if cjk else "subpath mount conflict"
    return "引发回归故障被撤回" if cjk else "reverted regression"


def extract_structured_fields_from_commit(
    subject: str,
    body: str,
    diff_text: str,
) -> Dict[str, str]:
    """
    Parse structured tags (if present in commit body) or heuristically extract
    {attempted_change, error_root_cause, dead_end_approach, safe_pivot, target_symbol}
    from a git revert/rollback commit and its unified diff.
    Correctly ignores Git's default 'This reverts commit <sha>.' boilerplate line.
    """
    symbol, added, removed = parse_diff_hunk_metadata(diff_text)

    # Filter out git's default 'This reverts commit <sha>.' boilerplate from body
    meaningful_body_lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip()
        and not re.match(r"^This reverts commit [0-9a-f]{7,40}\.?$", line.strip(), re.I)
    ]
    clean_body = "\n".join(meaningful_body_lines)
    full_msg = f"{subject}\n{clean_body}".strip()

    dead_end = ""
    root_cause = ""
    safe_pivot = ""
    attempted = subject

    for line in full_msg.splitlines():
        low = line.lower().strip()
        if low.startswith(("dead-end:", "dead_end:", "死路:")):
            dead_end = line.split(":", 1)[-1].strip(" ：")
        elif low.startswith(("root-cause:", "root_cause:", "error:", "根因:", "报错:")):
            root_cause = line.split(":", 1)[-1].strip(" ：")
        elif low.startswith(("safe-pivot:", "pivot:", "fix:", "正路:", "替代方案:", "修复:")):
            safe_pivot = line.split(":", 1)[-1].strip(" ：")
        elif low.startswith(("attempted:", "尝试改动:")):
            attempted = line.split(":", 1)[-1].strip(" ：")

    salient_removed = _pick_salient_code_line(removed)
    salient_added = _pick_salient_code_line(added)

    # Heuristic fallback from commit subject/body and diff lines
    if not dead_end:
        rev_match = re.search(r'Revert\s+"([^"]+)"', subject, re.IGNORECASE)
        if rev_match:
            orig_title = rev_match.group(1).strip()
            orig_title = re.sub(r"^[a-z]+(?:\([^)]+\))?:\s*", "", orig_title, flags=re.I)
            dead_end = orig_title
        else:
            # Handle unquoted Revert subjects like:
            # - "Revert `mount_path` parameter from `FastMCP` (#1881)"
            # - "fix(memory): revert back instantiation of KnowledgeGraphManager"
            cleaned_subj = re.sub(r"\s*\(#\d+\)\s*$", "", subject).strip()
            cleaned_subj = re.sub(r"^[a-z]+(?:\([^)]+\))?:\s*", "", cleaned_subj, flags=re.I)
            cleaned_subj = re.sub(
                r"^(?:revert\s+back\s+to\s+|revert\s+back\s+|revert\s+|rollback\s+|reset\s+|回滚[:\s]*|撤回[:\s]*)",
                "",
                cleaned_subj,
                flags=re.I,
            ).strip()
            if salient_removed and (
                len(cleaned_subj) < 8 or "original" in cleaned_subj.lower()
            ):
                dead_end = salient_removed
            elif cleaned_subj and salient_removed and cleaned_subj.lower() not in salient_removed.lower():
                dead_end = f"{cleaned_subj} ({salient_removed[:28]})"
            else:
                dead_end = cleaned_subj or salient_removed

    if not root_cause:
        cause_match = re.search(
            r"(?:由于|因为|导致|触发|caused by|due to|because of|triggers)\s*([^。.;；\n]{4,50})",
            full_msg,
            re.IGNORECASE,
        )
        if cause_match:
            root_cause = cause_match.group(1).strip()
        elif meaningful_body_lines:
            root_cause = meaningful_body_lines[0][:45]
        else:
            root_cause = _infer_cause_from_code(f"{subject} {dead_end} {' '.join(removed)}")

    if not safe_pivot and salient_added:
        safe_pivot = salient_added

    return {
        "target_symbol": symbol,
        "attempted_change": attempted or dead_end,
        "error_root_cause": root_cause,
        "dead_end_approach": dead_end,
        "safe_pivot": safe_pivot,
    }


class GitScarExtractor:
    """Inspects Git commits, reflog resets, and history to capture dead ends."""

    def __init__(self, repo_root: Path, store: Optional[ScarStore] = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.store = store or ScarStore(self.repo_root / ".scartissue", repo_root=self.repo_root)

    def inspect_commit(self, commit_ref: str = "HEAD") -> List[ScarRecord]:
        """
        Inspect a specific commit. If it is a revert, rollback, or hotfix-after-failure,
        extract dead-end scar records per affected code file.
        """
        meta = _run_git(self.repo_root, ["log", "-1", "--format=%H%n%s%n%b", commit_ref])
        if not meta:
            return []

        lines = meta.splitlines()
        sha = lines[0].strip() if lines else commit_ref
        subject = lines[1].strip() if len(lines) > 1 else ""
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""

        combined_msg = f"{subject}\n{body}"
        # Only trigger on genuine revert/rollback commits or commits with explicit Dead-End/Scar tags.
        # Do NOT false-trigger on normal feature commits like "feat: add password reset"!
        is_revert_subject = bool(
            re.match(
                r'^(?:revert\b|rollback\b|回滚|撤回|fix\([^)]*\):\s*revert)',
                subject.strip(),
                re.IGNORECASE,
            )
        )
        has_revert_body_or_tag = bool(
            re.search(
                r"(?:^This reverts commit [0-9a-f]{7,40}|^(?:dead[-_]?end|死路|scar)\s*:)",
                combined_msg,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        if not (is_revert_subject or has_revert_body_or_tag):
            return []

        files_out = _run_git(
            self.repo_root, ["diff-tree", "--no-commit-id", "--name-only", "-r", sha]
        )
        all_changed = [
            f.strip()
            for f in files_out.splitlines()
            if f.strip() and not f.strip().startswith(".scartissue")
        ]
        # Filter out pure documentation or lock files when actual source code files are present
        code_files = [
            f
            for f in all_changed
            if not f.lower().endswith((".md", ".rst", ".txt", ".lock"))
            and not f.startswith(("docs/", "tests/", "test_"))
        ]
        changed_files = code_files if code_files else all_changed
        created_records: List[ScarRecord] = []

        for rel_file in changed_files:
            diff_text = _run_git(self.repo_root, ["show", "--format=", sha, "--", rel_file])
            extracted = extract_structured_fields_from_commit(
                subject=subject,
                body=body,
                diff_text=diff_text,
            )
            raw_chars = len(combined_msg) + len(diff_text)
            rec = self.store.record_dead_end(
                target_file=normalize_rel_path(rel_file, self.repo_root),
                target_symbol=extracted["target_symbol"],
                attempted_change=extracted["attempted_change"],
                error_root_cause=extracted["error_root_cause"],
                dead_end_approach=extracted["dead_end_approach"],
                safe_pivot=extracted["safe_pivot"],
                trigger_source=TriggerSource.GIT_REVERT,
                commit_or_run_ref=sha[:8],
                raw_context_chars=max(raw_chars, 500),
            )
            created_records.append(rec)

        return created_records

    def inspect_latest_reflog_reset(self) -> List[ScarRecord]:
        """
        Detect if the latest Git operation in reflog was `git reset` moving away from
        a discarded commit (`HEAD@{1}` -> `HEAD@{0}`), and extract the discarded commit
        as a `GIT_RESET` dead-end scar!
        """
        reflog_out = _run_git(self.repo_root, ["reflog", "-n", "2", "--format=%H|%gs"])
        if not reflog_out:
            return []
        entries = [line.strip() for line in reflog_out.splitlines() if "|" in line]
        if len(entries) < 2:
            return []

        head0_sha, head0_action = entries[0].split("|", 1)
        head1_sha, _ = entries[1].split("|", 1)

        if not head0_action.lower().startswith("reset:") or head0_sha == head1_sha:
            return []

        # Verify this reset actually discarded/rolled back `head1_sha`
        # (i.e., `head1_sha` is NOT an ancestor of `head0_sha`, preventing false triggers on fast-forward resets)
        is_fast_forward = subprocess.run(
            ["git", "merge-base", "--is-ancestor", head1_sha, head0_sha],
            cwd=str(self.repo_root),
            capture_output=True,
            check=False,
        ).returncode == 0
        if is_fast_forward:
            return []

        # `head1_sha` is the commit that was discarded by `git reset`!
        meta = _run_git(self.repo_root, ["log", "-1", "--format=%s%n%b", head1_sha])
        subject = meta.splitlines()[0].strip() if meta else "discarded commit"
        files_out = _run_git(
            self.repo_root, ["diff-tree", "--no-commit-id", "--name-only", "-r", head1_sha]
        )
        changed_files = [f.strip() for f in files_out.splitlines() if f.strip()]
        created: List[ScarRecord] = []

        for rel_file in changed_files:
            if rel_file.startswith(".scartissue"):
                continue
            # Diff from discarded head1_sha to restored head0_sha:
            # '-' lines are the discarded dead-end lines, '+' lines are the restored baseline
            diff_text = _run_git(
                self.repo_root, ["diff", head1_sha, head0_sha, "--", rel_file]
            )
            symbol, added, removed = parse_diff_hunk_metadata(diff_text)
            dead_end = removed[0][:60] if removed else subject
            safe_pivot = added[0][:50] if added else ""
            cause = _infer_cause_from_code(f"{subject} {' '.join(removed)}")

            rec = self.store.record_dead_end(
                target_file=normalize_rel_path(rel_file, self.repo_root),
                target_symbol=symbol,
                attempted_change=f"git reset 丢弃提交: {subject}",
                error_root_cause=cause,
                dead_end_approach=dead_end,
                safe_pivot=safe_pivot,
                trigger_source=TriggerSource.GIT_RESET,
                commit_or_run_ref=head1_sha[:8],
                raw_context_chars=max(len(diff_text) + len(meta), 500),
            )
            created.append(rec)

        return created

    def scan_recent_history(self, max_commits: int = 50) -> List[ScarRecord]:
        """Scan up to `max_commits` in git history to cold-start the ScarTissue database."""
        shas_out = _run_git(
            self.repo_root, ["log", f"-n{max_commits}", "--format=%H"]
        )
        if not shas_out:
            return []
        records: List[ScarRecord] = []
        for sha in reversed(shas_out.splitlines()):
            sha = sha.strip()
            if sha:
                records.extend(self.inspect_commit(sha))
        return records

    def install_git_hooks(self, force: bool = False) -> List[Path]:
        """
        Idempotently install `post-commit`, `post-rewrite`, and `post-checkout` hooks
        in the repository's resolved Git hooks directory (supporting worktrees,
        submodules, and `core.hooksPath`) without clobbering existing user hooks.
        Refuses to mutate non-shell hooks unless `force=True`.
        """
        git_dir = _run_git(self.repo_root, ["rev-parse", "--git-dir"])
        if not git_dir:
            raise ValueError(f"Not a git repository: {self.repo_root}")

        configured_hooks = _run_git(self.repo_root, ["config", "--get", "core.hooksPath"])
        if configured_hooks:
            hooks_dir = Path(configured_hooks).expanduser()
            if not hooks_dir.is_absolute():
                hooks_dir = (self.repo_root / hooks_dir).resolve()
        else:
            hooks_path_str = _run_git(self.repo_root, ["rev-parse", "--git-path", "hooks"])
            if not hooks_path_str:
                raise ValueError(
                    f"Unable to resolve git hooks directory for: {self.repo_root}"
                )
            hooks_dir = Path(hooks_path_str)
            if not hooks_dir.is_absolute():
                hooks_dir = (self.repo_root / hooks_dir).resolve()

        hooks_dir.mkdir(parents=True, exist_ok=True)
        installed: List[Path] = []

        hook_snippet = (
            f"\n{HOOK_MARKER}\n"
            'REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"\n'
            'if command -v scar-tissue >/dev/null 2>&1; then\n'
            '  scar-tissue capture-git --repo "$REPO_ROOT" >/dev/null 2>&1 || true\n'
            'else\n'
            '  python3 -m scar_tissue.cli capture-git --repo "$REPO_ROOT" >/dev/null 2>&1 || true\n'
            'fi\n'
        )
        shell_shebang_re = re.compile(
            r"^#!\s*(?:/usr/bin/env\s+(?:-S\s+)?)?(?:/[^\s]*/)?(?:sh|bash|dash|zsh)\b"
        )

        for hook_name in ("post-commit", "post-rewrite", "post-checkout"):
            hook_path = hooks_dir / hook_name
            if hook_path.exists():
                existing = hook_path.read_text(encoding="utf-8")
                if HOOK_MARKER not in existing:
                    lines = existing.splitlines()
                    first_line = lines[0].strip() if lines else ""
                    is_shell_hook = (not first_line.startswith("#!")) or bool(
                        shell_shebang_re.match(first_line)
                    )
                    if not is_shell_hook:
                        if not force:
                            raise ValueError(
                                f"Existing hook {hook_path} uses a non-shell interpreter "
                                f"({first_line!r}); pass --force to replace it."
                            )
                        hook_path.write_text(
                            f"#!/bin/sh\n{hook_snippet.lstrip()}", encoding="utf-8"
                        )
                    else:
                        hook_path.write_text(
                            existing.rstrip() + "\n" + hook_snippet, encoding="utf-8"
                        )
            else:
                with open(hook_path, "x", encoding="utf-8") as fh:
                    fh.write(f"#!/bin/sh\n{hook_snippet.lstrip()}")
            hook_path.chmod(
                hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
            installed.append(hook_path)

        return installed


class ThreeStrikeTestTracker:
    """
    Monitors consecutive test failures per target file.
    When a file's test suite fails >= `strike_threshold` (default 3) consecutive times
    and then passes on the recovery run, automatically extracts the dead-end scar!
    """

    def __init__(
        self,
        repo_root: Path,
        store: Optional[ScarStore] = None,
        strike_threshold: int = 3,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.store = store or ScarStore(
            self.repo_root / ".scartissue", repo_root=self.repo_root
        )
        self.strike_threshold = strike_threshold
        self.state_file = self.store.storage_dir / "test_streak.json"

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "streaks" in data:
                    return data
            except Exception:
                pass
        return {"streaks": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def parse_error_from_test_output(output: str) -> str:
        """Extract concise root cause exception or assertion failure from pytest/unittest logs."""
        for line in reversed(output.splitlines()):
            line_s = line.strip()
            m = re.match(
                r"^(?:E\s+)?([A-Za-z0-9_.]*(?:Error|Exception|Deadlock|Fault|Timeout|OOM)[A-Za-z0-9_.]*:\s*.+)$",
                line_s,
            )
            if m:
                return m.group(1)[:65]
        for line in output.splitlines():
            if "FAILED" in line or "ERROR" in line:
                return line.strip()[:65]
        return "单测断言/运行时崩溃"

    def record_test_attempt(
        self,
        passed: bool,
        target_file: str,
        test_output: str = "",
        attempted_approach: str = "",
        target_symbol: str = "",
        working_fix_pivot: str = "",
    ) -> Optional[ScarRecord]:
        """
        Record a single test execution outcome isolated by `target_file`.
        - If `passed == False`: increments consecutive failure count for `target_file`.
        - If `passed == True` and `consecutive_failures >= self.strike_threshold`:
          automatically crystallizes the crash loop into a ScarRecord and resets the file's streak!
        """
        state = self._load_state()
        streaks: Dict[str, Any] = state.setdefault("streaks", {})
        norm_file = normalize_rel_path(target_file, self.repo_root) or "__global__"
        bucket = streaks.setdefault(
            norm_file, {"consecutive_failures": 0, "failed_runs": []}
        )

        if not passed:
            error_cause = self.parse_error_from_test_output(test_output)
            bucket["consecutive_failures"] = int(bucket.get("consecutive_failures", 0)) + 1
            bucket.setdefault("failed_runs", []).append(
                {
                    "target_file": norm_file,
                    "target_symbol": target_symbol,
                    "attempted_approach": attempted_approach or "高危实现写法",
                    "error_cause": error_cause,
                    "raw_chars": len(test_output) + len(attempted_approach),
                }
            )
            self._save_state(state)
            return None

        # Test passed for `norm_file`! Check if it recovered from >= 3 consecutive failures
        failures = int(bucket.get("consecutive_failures", 0))
        failed_runs = bucket.get("failed_runs", [])

        # Reset only this file's streak
        streaks[norm_file] = {"consecutive_failures": 0, "failed_runs": []}
        self._save_state(state)

        if failures < self.strike_threshold or not failed_runs:
            return None

        first_fail = failed_runs[0]
        last_fail = failed_runs[-1]
        total_raw_chars = sum(int(r.get("raw_chars", 400)) for r in failed_runs) + len(
            working_fix_pivot
        )

        dead_end_desc = last_fail.get("attempted_approach") or first_fail.get(
            "attempted_approach"
        )
        error_root_cause = last_fail.get("error_cause") or first_fail.get("error_cause")
        symbol = target_symbol or last_fail.get("target_symbol", "")

        return self.store.record_dead_end(
            target_file=norm_file,
            target_symbol=symbol,
            attempted_change=f"连续{failures}次单测失败尝试: {dead_end_desc}",
            error_root_cause=error_root_cause,
            dead_end_approach=dead_end_desc,
            safe_pivot=working_fix_pivot or "通过单测验证的隔离/无锁实现",
            trigger_source=TriggerSource.PYTEST_3_STRIKE,
            commit_or_run_ref=f"{failures}x_fail_recovered",
            raw_context_chars=max(total_raw_chars, 900),
        )

    # Alias for backward compatibility
    record_test_Attempt = record_test_attempt

    def run_test_command(
        self,
        command: List[str],
        target_file: str = "",
        attempted_approach: str = "",
        working_fix_pivot: str = "",
        return_output: bool = False,
    ) -> Any:
        """
        Execute a test command (e.g. `['pytest', '-q']`), inspect working tree `git diff`
        if `target_file` is not explicitly specified, and feed the outcome into the 3-strike tracker.
        """
        res = subprocess.run(
            command,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        output = f"{res.stdout}\n{res.stderr}".strip()
        passed = res.returncode == 0

        inferred_file = target_file
        inferred_symbol = ""
        inferred_added: List[str] = []

        if not inferred_file:
            diff_files = _run_git(self.repo_root, ["diff", "--name-only", "HEAD"])
            candidates = [
                f.strip()
                for f in diff_files.splitlines()
                if f.strip() and not f.strip().startswith(("tests/", "test_", ".scartissue"))
            ]
            if candidates:
                inferred_file = candidates[0]
            elif diff_files.splitlines():
                inferred_file = diff_files.splitlines()[0].strip()
            else:
                inferred_file = "workspace"

        diff_text = _run_git(self.repo_root, ["diff", "HEAD", "--", inferred_file])
        if diff_text:
            inferred_symbol, inferred_added, _ = parse_diff_hunk_metadata(diff_text)

        approach = attempted_approach or (inferred_added[0][:60] if inferred_added else "")
        pivot = working_fix_pivot or (
            inferred_added[0][:50] if passed and inferred_added else ""
        )

        scar = self.record_test_attempt(
            passed=passed,
            target_file=inferred_file,
            test_output=output,
            attempted_approach=approach,
            target_symbol=inferred_symbol,
            working_fix_pivot=pivot,
        )
        if return_output:
            return res.returncode, scar, output
        return res.returncode, scar
