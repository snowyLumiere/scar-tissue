"""Command-line interface for ScarTissue (MCP Server, Git/Pytest Hooks, & CLI Inspector)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from scar_tissue import __version__
from scar_tissue.hooks import GitScarExtractor, ThreeStrikeTestTracker
from scar_tissue.store import ScarStore


def _resolve_cli_storage_dir(repo_root: Path) -> Path:
    """Resolve storage directory, honoring SCAR_TISSUE_DIR when set."""
    env_dir = os.environ.get("SCAR_TISSUE_DIR", "").strip()
    if env_dir:
        env_path = Path(env_dir).expanduser()
        return env_path if env_path.is_absolute() else (repo_root / env_path)
    return repo_root / ".scartissue"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scar-tissue",
        description="ScarTissue: Anti-Memory MCP Server & Git/Test Hook Gatekeeper for Coding Agents",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="cmd")

    # 1. serve (default if no subcommand)
    sub.add_parser("serve", help="Run the ScarTissue MCP server over stdio")

    # 2. install-hooks
    p_hooks = sub.add_parser(
        "install-hooks",
        help="Idempotently install Git post-commit, post-rewrite, and post-checkout hooks",
    )
    p_hooks.add_argument("--repo", default=".", help="Target git repository root")
    p_hooks.add_argument(
        "--force",
        action="store_true",
        help="Replace existing incompatible non-shell hooks instead of refusing",
    )

    # 3. capture-git
    p_cap = sub.add_parser(
        "capture-git",
        help="Extract dead-end scars from Git revert commits, reflog resets, or history",
    )
    p_cap.add_argument("--repo", default=".", help="Target git repository root")
    p_cap.add_argument("--commit", default="HEAD", help="Specific commit ref to inspect")
    p_cap.add_argument("--scan", type=int, default=0, help="Scan last N commits in history")

    # 4. test (3-strike crash-loop wrapper)
    p_test = sub.add_parser(
        "test",
        help="Run a test command (e.g. pytest) with 3-strike crash-loop auto-capture",
    )
    p_test.add_argument("--repo", default=".", help="Target repository root")
    p_test.add_argument("--file", default="", help="Target file being tested/modified")
    p_test.add_argument("--attempted", default="", help="Description of attempted approach")
    p_test.add_argument("--pivot", default="", help="Description of working fix")
    p_test.add_argument(
        "test_cmd",
        nargs=argparse.REMAINDER,
        help="Test command to run (e.g. -- pytest -q)",
    )

    # 5. record
    p_rec = sub.add_parser("record", help="Manually record a dead-end scar")
    p_rec.add_argument("--repo", default=".", help="Repository root")
    p_rec.add_argument("--file", required=True, help="Target file path")
    p_rec.add_argument("--symbol", default="", help="Function or class symbol")
    p_rec.add_argument("--attempted", required=True, help="What change was attempted")
    p_rec.add_argument("--cause", required=True, help="Error root cause")
    p_rec.add_argument("--dead-end", required=True, help="Proven infeasible approach")
    p_rec.add_argument("--pivot", default="", help="Verified working alternative")

    # 6. check
    p_chk = sub.add_parser("check", help="Check scar tissue before modifying a file")
    p_chk.add_argument("file", help="Target file path")
    p_chk.add_argument("--repo", default=".", help="Repository root")
    p_chk.add_argument("--plan", default="", help="Planned code modification description")
    p_chk.add_argument("--symbol", default="", help="Target function or class symbol")

    # 7. list / stats
    p_list = sub.add_parser("list", help="List all recorded scars and telemetry stats")
    p_list.add_argument("--repo", default=".", help="Repository root")

    args = parser.parse_args(argv)

    if not args.cmd or args.cmd == "serve":
        from scar_tissue.server import main as run_mcp_server

        run_mcp_server()
        return 0

    repo_root = Path(getattr(args, "repo", ".")).resolve()
    storage_dir = _resolve_cli_storage_dir(repo_root)

    try:
        store = ScarStore(storage_dir, repo_root=repo_root)

        if args.cmd == "install-hooks":
            extractor = GitScarExtractor(repo_root, store)
            hooks = extractor.install_git_hooks(force=args.force)
            print(
                json.dumps(
                    {"installed_hooks": [str(h) for h in hooks]},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        if args.cmd == "capture-git":
            extractor = GitScarExtractor(repo_root, store)
            if args.scan > 0:
                records = extractor.scan_recent_history(max_commits=args.scan)
            else:
                records = extractor.inspect_commit(args.commit)
                records.extend(extractor.inspect_latest_reflog_reset())
            print(
                json.dumps(
                    {
                        "captured_count": len(records),
                        "scars": [
                            {
                                "scar_id": r.scar_id,
                                "target_file": r.target_file,
                                "trigger_source": r.trigger_source.value,
                                "alert_50c": r.alert_50c,
                                "char_len": len(r.alert_50c),
                            }
                            for r in records
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        if args.cmd == "test":
            cmd_list = list(args.test_cmd)
            if cmd_list and cmd_list[0] == "--":
                cmd_list = cmd_list[1:]
            if not cmd_list:
                cmd_list = ["pytest", "-q"]
            tracker = ThreeStrikeTestTracker(repo_root, store)
            rc, scar, test_out = tracker.run_test_command(
                command=cmd_list,
                target_file=args.file,
                attempted_approach=args.attempted,
                working_fix_pivot=args.pivot,
                return_output=True,
            )
            if test_out:
                print(test_out, file=sys.stderr if rc != 0 else sys.stdout)
            if scar is not None:
                print(
                    json.dumps(
                        {
                            "scar_captured": True,
                            "scar_id": scar.scar_id,
                            "alert_50c": scar.alert_50c,
                        },
                        ensure_ascii=False,
                    )
                )
            return rc

        if args.cmd == "record":
            rec = store.record_dead_end(
                target_file=args.file,
                target_symbol=args.symbol,
                attempted_change=args.attempted,
                error_root_cause=args.cause,
                dead_end_approach=args.dead_end,
                safe_pivot=args.pivot,
            )
            print(rec.model_dump_json(indent=2))
            return 0

        if args.cmd == "check":
            res = store.check_scar_tissue(
                target_file=args.file,
                planned_change=args.plan,
                target_symbol=args.symbol,
            )
            print(res.model_dump_json(indent=2))
            return 0

        if args.cmd == "list":
            scars = store.list_all_scars()
            stats = store.get_stats()
            print(
                json.dumps(
                    {
                        "stats": stats,
                        "scars": [
                            {
                                "scar_id": s.scar_id,
                                "target_file": s.target_file,
                                "target_symbol": s.target_symbol,
                                "alert_50c": s.alert_50c,
                                "char_len": len(s.alert_50c),
                                "trigger_source": s.trigger_source.value,
                                "hit_count": s.hit_count,
                            }
                            for s in scars
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
