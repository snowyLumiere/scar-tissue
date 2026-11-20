"""Data models for ScarTissue Anti-Memory records and retrieval results."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TriggerSource(str, Enum):
    """How a dead-end scar record was captured."""

    GIT_REVERT = "git_revert"
    GIT_RESET = "git_reset"
    PYTEST_3_STRIKE = "pytest_3_strike"
    MANUAL_MCP = "manual_mcp"
    GIT_HISTORY_SCAN = "git_history_scan"


class FailureArchetype(str, Enum):
    """Primary failure category for the 16-dim Scar Gatekeeper."""

    CONCURRENCY_LOCK = "concurrency_lock"
    MEMORY_OOM_SHAPE = "memory_oom_shape"
    IO_BUFFER_DEADLOCK = "io_buffer_deadlock"
    STATE_SYNC_LOOP = "state_sync_loop"
    DATABASE_TX_LOCK = "database_tx_lock"
    PATH_FS_BOUNDARY = "path_fs_boundary"
    SCHEMA_CONTRACT_DRIFT = "schema_contract_drift"
    PERF_TIMEOUT_COMPLEXITY = "perf_timeout_complexity"
    GENERAL_REGRESSION = "general_regression"


class ScarRecord(BaseModel):
    """
    A single 'Scar Tissue' (Anti-Memory) entry representing a proven dead end.
    """

    scar_id: str = Field(description="Short deterministic 8-char hex fingerprint of the dead end")
    target_file: str = Field(description="Normalized relative file path where the dead end occurred")
    target_symbol: str = Field(default="", description="Function, class, or module symbol affected")
    attempted_change: str = Field(description="What the agent/developer tried to do that failed")
    error_root_cause: str = Field(description="Root cause exception, deadlock, OOM, or test failure")
    dead_end_approach: str = Field(description="The specific approach proven infeasible (Dead End)")
    safe_pivot: str = Field(
        default="",
        description="The verified working alternative or positive architectural direction",
    )
    alert_50c: str = Field(
        description="High-priority Negative Constraint + Positive Pivot alert strictly <= 50 chars"
    )
    raw_negative_alert: str = Field(
        default="",
        description="Unfiltered negative warning (kept only for Pink-Elephant ablation comparison)",
    )
    latent_16d: List[float] = Field(
        default_factory=list,
        description="16-dimensional normalized semantic & structural scar vector (Memorilla-inspired gatekeeper)",
    )
    forbidden_tokens: List[str] = Field(
        default_factory=list,
        description="Code tokens/APIs from the dead-end diff that must not be echoed in the prompt",
    )
    archetype: FailureArchetype = Field(default=FailureArchetype.GENERAL_REGRESSION)
    trigger_source: TriggerSource = Field(default=TriggerSource.MANUAL_MCP)
    commit_or_run_ref: str = Field(default="", description="Git commit SHA or test run ID")
    raw_context_chars: int = Field(
        default=0,
        description="Total character length of raw git diff + error traceback before compression",
    )
    hit_count: int = Field(default=0, description="Number of times this scar intercepted a repeat edit")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @staticmethod
    def compute_id(target_file: str, dead_end_approach: str, error_root_cause: str) -> str:
        norm = f"{target_file.strip().lower()}::{dead_end_approach.strip().lower()}::{error_root_cause.strip().lower()}"
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:8]


class ScarMatch(BaseModel):
    """A matched scar record with gatekeeper relevance score."""

    scar: ScarRecord
    similarity_score: float = Field(description="Combined path, symbol, and 16-dim latent similarity [0..1]")
    match_reason: str = Field(description="Why this scar was triggered (e.g., file+symbol+latent)")


class ScarCheckResponse(BaseModel):
    """Response returned by check_scar_tissue to the coding agent."""

    has_scars: bool
    target_file: str
    alerts: List[str] = Field(
        default_factory=list,
        description="List of <=50-char redline alerts ready for zero-bloat context injection",
    )
    matches: List[ScarMatch] = Field(default_factory=list)
    tokens_saved_estimate: int = Field(
        default=0,
        description="Estimated prompt tokens saved vs. dumping raw failure logs / CLAUDE.md",
    )
