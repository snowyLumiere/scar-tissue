"""
ScarTissue MCP Server (~150 lines core).

Exposes two core Anti-Memory tools to Claude Code / Cursor / Codex:
1. `record_dead_end`: Records a proven dead end (from git revert, 3x test crash, or agent failure)
   and compiles it into a 16-dim latent scar vector + strictly <=50-char redline alert.
2. `check_scar_tissue`: Mandatory pre-edit gatekeeper that checks a file/symbol/plan before
   modifying code, popping up <=50-char negative constraints only when stepping into a minefield.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from scar_tissue.models import TriggerSource
from scar_tissue.store import ScarStore

mcp = FastMCP(
    "ScarTissue-AntiMemory",
    instructions=(
        "Before editing any file, ALWAYS call `check_scar_tissue(target_file, planned_change)` "
        "to check if your planned approach matches a previously reverted dead end. "
        "Whenever a git revert happens or a test fails 3 times before succeeding, "
        "call `record_dead_end` to immunize future sessions."
    ),
)


def _find_git_or_store_root(start: Path, rel_store_name: str) -> Optional[Path]:
    """Walk upward from `start` to find the nearest repo/store root."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / rel_store_name).exists() or (parent / ".git").exists():
            return parent
    return None


def _get_store(target_file: Optional[str] = None) -> ScarStore:
    """
    Resolve the ScarStore strictly from server-side configuration (`SCAR_TISSUE_DIR`
    or nearest Git/`.scartissue` root). Never accepts untrusted storage paths from MCP clients.
    """
    cwd = Path.cwd().resolve()
    env_dir = os.environ.get("SCAR_TISSUE_DIR", "").strip()
    env_path = Path(env_dir).expanduser() if env_dir else None
    rel_store_name = (
        env_dir if (env_path is not None and not env_path.is_absolute()) else ".scartissue"
    )

    detected_root: Optional[Path] = None
    if target_file:
        try:
            cand = Path(target_file)
            if cand.is_absolute():
                detected_root = _find_git_or_store_root(cand.parent, rel_store_name)
        except Exception:
            pass
    if detected_root is None:
        detected_root = _find_git_or_store_root(cwd, rel_store_name)

    repo_root = detected_root or cwd

    if env_path is not None and env_path.is_absolute():
        return ScarStore(env_path, repo_root=repo_root)

    return ScarStore(repo_root / rel_store_name, repo_root=repo_root)


@mcp.tool()
def record_dead_end(
    target_file: str,
    attempted_change: str,
    error_root_cause: str,
    dead_end_approach: str,
    safe_pivot: str = "",
    target_symbol: str = "",
    trigger_source: str = "manual_mcp",
    commit_or_run_ref: str = "",
    raw_context_chars: int = 0,
) -> Dict[str, Any]:
    """
    Record a proven dead-end code change (Anti-Memory / Scar Tissue).

    Extracts {attempted_change + error_root_cause + dead_end_approach + safe_pivot},
    encodes a 16-dimensional latent scar vector (Memorilla-inspired gatekeeper),
    and compiles a strictly <= 50 character Contrastive Pivot redline alert.
    """
    store = _get_store(target_file=target_file)
    try:
        source_enum = TriggerSource(trigger_source)
    except ValueError:
        source_enum = TriggerSource.MANUAL_MCP

    record = store.record_dead_end(
        target_file=target_file,
        attempted_change=attempted_change,
        error_root_cause=error_root_cause,
        dead_end_approach=dead_end_approach,
        safe_pivot=safe_pivot,
        target_symbol=target_symbol,
        trigger_source=source_enum,
        commit_or_run_ref=commit_or_run_ref,
        raw_context_chars=raw_context_chars,
    )

    return {
        "status": "recorded",
        "scar_id": record.scar_id,
        "target_file": record.target_file,
        "target_symbol": record.target_symbol,
        "alert_50c": record.alert_50c,
        "alert_char_length": len(record.alert_50c),
        "archetype": record.archetype.value,
        "latent_16d_preview": record.latent_16d[:4],
    }


@mcp.tool()
def check_scar_tissue(
    target_file: str,
    planned_change: str = "",
    target_symbol: str = "",
    top_k: int = 3,
    threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Mandatory pre-edit Gatekeeper: Check historical dead ends before modifying `target_file`.

    Instead of dumping thousands of lines of CLAUDE.md history into the context window,
    returns strictly <= 50-char high-priority redline alerts ONLY when the target file,
    symbol, or 16-dim semantic intent matches a historical minefield.
    """
    store = _get_store(target_file=target_file)
    result = store.check_scar_tissue(
        target_file=target_file,
        planned_change=planned_change,
        target_symbol=target_symbol,
        top_k=top_k,
        threshold=threshold,
    )

    return {
        "has_scars": result.has_scars,
        "target_file": result.target_file,
        "alerts": result.alerts,
        "tokens_saved_estimate": result.tokens_saved_estimate,
        "matches": [
            {
                "scar_id": m.scar.scar_id,
                "alert_50c": m.scar.alert_50c,
                "similarity_score": m.similarity_score,
                "match_reason": m.match_reason,
                "commit_or_run_ref": m.scar.commit_or_run_ref,
            }
            for m in result.matches
        ],
    }


def main() -> None:
    """Entry point for running ScarTissue MCP Server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
