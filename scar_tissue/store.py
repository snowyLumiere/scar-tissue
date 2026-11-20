"""
ScarStore: Hybrid SQLite + Human-Auditable JSONL Storage & 16-Dim Memory Gatekeeper.

Solves both:
1. CLAUDE.md bloat (>200 lines truncation & attention dilution) by returning only matched <=50-char alerts.
2. Memorilla's black-box uneditability by pairing the 16-float latent gatekeeper vector with
   a bidirectionally synchronized SQLite + JSONL store that humans can edit or prune in any text editor.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from scar_tissue.encoder import (
    MAX_ALERT_CHARS,
    compile_negative_constraint,
    cosine_similarity_16d,
    encode_16d_scar_vector,
    extract_forbidden_tokens,
    scrub_sensitive_info,
)
from scar_tissue.models import (
    FailureArchetype,
    ScarCheckResponse,
    ScarMatch,
    ScarRecord,
    TriggerSource,
)

# Generic filenames that exist in many directories and must never match by bare basename alone
AMBIGUOUS_BASENAMES: Set[str] = {
    "__init__.py",
    "__main__.py",
    "utils.py",
    "helpers.py",
    "common.py",
    "base.py",
    "config.py",
    "models.py",
    "types.py",
    "views.py",
    "cli.py",
    "main.py",
    "server.py",
    "index.ts",
    "index.tsx",
    "index.js",
    "index.jsx",
    "types.ts",
    "utils.ts",
}


def normalize_rel_path(file_path: str, repo_root: Optional[Path] = None) -> str:
    """
    Normalize a file path (absolute or relative) to a clean POSIX path relative to `repo_root`.
    """
    cleaned = file_path.strip().replace("\\", "/")
    if not cleaned:
        return ""

    if repo_root is not None:
        try:
            root_resolved = repo_root.resolve()
            candidate = Path(cleaned)
            if candidate.is_absolute():
                abs_p = candidate.resolve()
                if abs_p.is_relative_to(root_resolved):
                    cleaned = abs_p.relative_to(root_resolved).as_posix()
        except Exception:
            pass

    while cleaned.startswith("./"):
        cleaned = cleaned[2:]

    # Privacy guard: never persist absolute OS user paths (macOS/Linux/Windows) outside repo_root
    home_match = re.match(
        r"^(?:[A-Za-z]:)?/(?:Users|home)/[^/]+(?:/(.*))?$", cleaned, re.IGNORECASE
    )
    if home_match:
        tail = (home_match.group(1) or "").strip("/")
        parts = [p for p in tail.split("/") if p]
        cleaned = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")
    elif re.match(r"^[A-Za-z]:/", cleaned):
        cleaned = cleaned[3:]
    elif cleaned.startswith("/"):
        cleaned = cleaned.lstrip("/")

    return cleaned


class ScarStore:
    """Local Anti-Memory store and 16-dim Gatekeeper with bidirectional JSONL sync."""

    def __init__(
        self,
        storage_dir: Optional[str | Path] = None,
        repo_root: Optional[str | Path] = None,
    ) -> None:
        if storage_dir is None:
            env_dir = os.environ.get("SCAR_TISSUE_DIR")
            storage_dir = Path(env_dir) if env_dir else Path.cwd() / ".scartissue"
        self.storage_dir = Path(storage_dir).resolve()
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else self.storage_dir.parent
        )
        self.db_path = self.storage_dir / "scars.db"
        self.jsonl_path = self.storage_dir / "scars.jsonl"
        self._db_initialized = False
        if self.db_path.exists() or self.jsonl_path.exists():
            self._ensure_db()
            self._reconcile_from_jsonl_if_edited()

    def _has_backing_files(self) -> bool:
        return self.db_path.exists() or self.jsonl_path.exists()

    def _ensure_db(self) -> None:
        if self._db_initialized and self.db_path.exists():
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._db_initialized = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scars (
                    scar_id TEXT PRIMARY KEY,
                    target_file TEXT NOT NULL,
                    target_symbol TEXT NOT NULL,
                    attempted_change TEXT NOT NULL,
                    error_root_cause TEXT NOT NULL,
                    dead_end_approach TEXT NOT NULL,
                    safe_pivot TEXT NOT NULL,
                    alert_50c TEXT NOT NULL,
                    raw_negative_alert TEXT NOT NULL,
                    latent_16d_json TEXT NOT NULL,
                    forbidden_tokens_json TEXT NOT NULL,
                    archetype TEXT NOT NULL,
                    trigger_source TEXT NOT NULL,
                    commit_or_run_ref TEXT NOT NULL,
                    raw_context_chars INTEGER NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scars_target_file ON scars(target_file);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_meta (
                    meta_key TEXT PRIMARY KEY,
                    meta_val TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        if not path.exists():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _reconcile_from_jsonl_if_edited(self) -> None:
        """
        If a human developer edited or deleted lines in `.scartissue/scars.jsonl`
        (or pulled updated `scars.jsonl` from Git), sync those changes back into SQLite.
        Fails closed (raises `ValueError`) on any malformed non-empty line before touching SQLite.
        """
        if not self.jsonl_path.exists():
            return

        self._ensure_db()
        current_hash = self._file_sha256(self.jsonl_path)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT meta_val FROM sync_meta WHERE meta_key = 'jsonl_sha256'"
            ).fetchone()
            recorded_hash = row["meta_val"] if row else ""

        if current_hash == recorded_hash:
            return

        # Parse human-edited JSONL lines (fail-closed: never silently drop malformed lines)
        raw_text = self.jsonl_path.read_text(encoding="utf-8")
        parsed_records: List[ScarRecord] = []
        for lineno, line in enumerate(raw_text.splitlines(), start=1):
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = ScarRecord.model_validate_json(line_s)
            except Exception as exc:
                raise ValueError(
                    f"Malformed JSONL record in {self.jsonl_path} at line {lineno}: {exc}"
                ) from exc
            if len(rec.alert_50c) > MAX_ALERT_CHARS:
                rec.alert_50c = rec.alert_50c[: MAX_ALERT_CHARS - 1] + "…"
            if not rec.latent_16d or len(rec.latent_16d) != 16:
                rec.latent_16d, rec.archetype = encode_16d_scar_vector(
                    rec.target_file,
                    rec.target_symbol,
                    rec.attempted_change,
                    rec.error_root_cause,
                    rec.dead_end_approach,
                )
            parsed_records.append(rec)

        with self._connect() as conn:
            existing_hits = {
                r["scar_id"]: int(r["hit_count"])
                for r in conn.execute("SELECT scar_id, hit_count FROM scars").fetchall()
            }
            conn.execute("DELETE FROM scars;")
            for record in parsed_records:
                merged_hits = max(record.hit_count, existing_hits.get(record.scar_id, 0))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scars (
                        scar_id, target_file, target_symbol, attempted_change,
                        error_root_cause, dead_end_approach, safe_pivot,
                        alert_50c, raw_negative_alert, latent_16d_json,
                        forbidden_tokens_json, archetype, trigger_source,
                        commit_or_run_ref, raw_context_chars, hit_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.scar_id,
                        normalize_rel_path(record.target_file, self.repo_root),
                        record.target_symbol,
                        record.attempted_change,
                        record.error_root_cause,
                        record.dead_end_approach,
                        record.safe_pivot,
                        record.alert_50c,
                        record.raw_negative_alert,
                        json.dumps(record.latent_16d),
                        json.dumps(record.forbidden_tokens, ensure_ascii=False),
                        record.archetype.value,
                        record.trigger_source.value,
                        record.commit_or_run_ref,
                        record.raw_context_chars,
                        merged_hits,
                        record.created_at,
                    ),
                )
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (meta_key, meta_val) VALUES ('jsonl_sha256', ?)",
                (current_hash,),
            )

    def _sync_jsonl(self) -> None:
        """Atomically write current SQLite state to `.scartissue/scars.jsonl` and update checksum."""
        self._ensure_db()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scars ORDER BY created_at DESC"
            ).fetchall()
        records = [self._row_to_record(r) for r in rows]
        lines = [rec.model_dump_json() for rec in records]
        content = "\n".join(lines) + ("\n" if lines else "")
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".scars.", suffix=".jsonl.tmp", dir=str(self.storage_dir)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.jsonl_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        new_hash = self._file_sha256(self.jsonl_path)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (meta_key, meta_val) VALUES ('jsonl_sha256', ?)",
                (new_hash,),
            )

    def record_dead_end(
        self,
        target_file: str,
        attempted_change: str,
        error_root_cause: str,
        dead_end_approach: str,
        safe_pivot: str = "",
        target_symbol: str = "",
        trigger_source: TriggerSource = TriggerSource.MANUAL_MCP,
        commit_or_run_ref: str = "",
        raw_context_chars: int = 0,
    ) -> ScarRecord:
        """
        Compress and store a proven dead end into the SQLite + JSONL ScarTissue store.
        Automatically synthesizes the 16-dim latent vector and <=50-char Anti-Pink-Elephant alert.
        """
        self._ensure_db()
        self._reconcile_from_jsonl_if_edited()
        attempted_change = scrub_sensitive_info(attempted_change)
        error_root_cause = scrub_sensitive_info(error_root_cause)
        dead_end_approach = scrub_sensitive_info(dead_end_approach)
        safe_pivot = scrub_sensitive_info(safe_pivot)
        norm_file = normalize_rel_path(target_file, self.repo_root)
        scar_id = ScarRecord.compute_id(norm_file, dead_end_approach, error_root_cause)

        latent_16d, archetype = encode_16d_scar_vector(
            target_file=norm_file,
            target_symbol=target_symbol,
            attempted_change=attempted_change,
            error_root_cause=error_root_cause,
            dead_end_approach=dead_end_approach,
        )

        alert_50c, raw_negative = compile_negative_constraint(
            dead_end_approach=dead_end_approach,
            error_root_cause=error_root_cause,
            safe_pivot=safe_pivot,
            max_chars=MAX_ALERT_CHARS,
        )

        forbidden = extract_forbidden_tokens(f"{attempted_change} {dead_end_approach}")
        if raw_context_chars <= 0:
            raw_context_chars = max(
                len(attempted_change) + len(error_root_cause) + len(dead_end_approach) + len(safe_pivot),
                450,
            )

        record = ScarRecord(
            scar_id=scar_id,
            target_file=norm_file,
            target_symbol=target_symbol.strip(),
            attempted_change=attempted_change.strip(),
            error_root_cause=error_root_cause.strip(),
            dead_end_approach=dead_end_approach.strip(),
            safe_pivot=safe_pivot.strip(),
            alert_50c=alert_50c,
            raw_negative_alert=raw_negative,
            latent_16d=latent_16d,
            forbidden_tokens=forbidden,
            archetype=archetype,
            trigger_source=trigger_source,
            commit_or_run_ref=commit_or_run_ref.strip(),
            raw_context_chars=raw_context_chars,
        )

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT hit_count FROM scars WHERE scar_id = ?", (scar_id,)
            ).fetchone()
            hit_count = int(existing["hit_count"]) if existing else 0
            record.hit_count = hit_count

            conn.execute(
                """
                INSERT OR REPLACE INTO scars (
                    scar_id, target_file, target_symbol, attempted_change,
                    error_root_cause, dead_end_approach, safe_pivot,
                    alert_50c, raw_negative_alert, latent_16d_json,
                    forbidden_tokens_json, archetype, trigger_source,
                    commit_or_run_ref, raw_context_chars, hit_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.scar_id,
                    record.target_file,
                    record.target_symbol,
                    record.attempted_change,
                    record.error_root_cause,
                    record.dead_end_approach,
                    record.safe_pivot,
                    record.alert_50c,
                    record.raw_negative_alert,
                    json.dumps(record.latent_16d),
                    json.dumps(record.forbidden_tokens, ensure_ascii=False),
                    record.archetype.value,
                    record.trigger_source.value,
                    record.commit_or_run_ref,
                    record.raw_context_chars,
                    record.hit_count,
                    record.created_at,
                ),
            )

        self._sync_jsonl()
        return record

    def _row_to_record(self, row: sqlite3.Row) -> ScarRecord:
        return ScarRecord(
            scar_id=row["scar_id"],
            target_file=row["target_file"],
            target_symbol=row["target_symbol"],
            attempted_change=row["attempted_change"],
            error_root_cause=row["error_root_cause"],
            dead_end_approach=row["dead_end_approach"],
            safe_pivot=row["safe_pivot"],
            alert_50c=row["alert_50c"],
            raw_negative_alert=row["raw_negative_alert"],
            latent_16d=json.loads(row["latent_16d_json"]),
            forbidden_tokens=json.loads(row["forbidden_tokens_json"]),
            archetype=FailureArchetype(row["archetype"]),
            trigger_source=TriggerSource(row["trigger_source"]),
            commit_or_run_ref=row["commit_or_run_ref"],
            raw_context_chars=int(row["raw_context_chars"]),
            hit_count=int(row["hit_count"]),
            created_at=row["created_at"],
        )

    def list_all_scars(self) -> List[ScarRecord]:
        if not self._has_backing_files():
            return []
        self._ensure_db()
        self._reconcile_from_jsonl_if_edited()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scars ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete_scar(self, scar_id: str) -> bool:
        """Allow humans to easily delete/edit out any outdated scar rule."""
        if not self._has_backing_files():
            return False
        self._ensure_db()
        self._reconcile_from_jsonl_if_edited()
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM scars WHERE scar_id = ?", (scar_id,))
            deleted = cur.rowcount > 0
        if deleted:
            self._sync_jsonl()
        return deleted

    def check_scar_tissue(
        self,
        target_file: str,
        planned_change: str = "",
        target_symbol: str = "",
        top_k: int = 3,
        threshold: float = 0.45,
    ) -> ScarCheckResponse:
        """
        Pre-edit Memory Gatekeeper:
        Checks whether `target_file` (and optional `target_symbol` / `planned_change`)
        steps into any recorded dead-end scar tissue.
        Returns strictly <= 50 char alerts only for genuine matches.
        """
        norm_file = normalize_rel_path(target_file, self.repo_root)
        query_base = Path(norm_file).name
        query_dir = str(Path(norm_file).parent)

        all_scars = self.list_all_scars()
        if not all_scars:
            return ScarCheckResponse(has_scars=False, target_file=norm_file)

        has_intent = bool(planned_change.strip() or target_symbol.strip())
        query_16d: List[float] = []
        if has_intent:
            query_16d, _ = encode_16d_scar_vector(
                target_file=norm_file,
                target_symbol=target_symbol,
                attempted_change=planned_change,
                error_root_cause=planned_change,
                dead_end_approach=planned_change,
            )

        matched: List[ScarMatch] = []
        for scar in all_scars:
            scar_norm = normalize_rel_path(scar.target_file, self.repo_root)
            scar_base = Path(scar_norm).name
            scar_dir = str(Path(scar_norm).parent)

            # 1. File affinity (prevents collisions on generic basenames like utils.py in different dirs)
            if scar_norm == norm_file and norm_file != "":
                file_score = 1.0
                reason_parts = ["exact_file"]
            elif (
                scar_base == query_base
                and query_base != ""
                and query_base.lower() not in AMBIGUOUS_BASENAMES
                and (scar_dir == "." or query_dir == ".")
            ):
                # Bare filename query matching a specific non-ambiguous file
                file_score = 0.90
                reason_parts = ["unambiguous_basename"]
            elif query_dir != "." and scar_dir == query_dir:
                file_score = 0.45
                reason_parts = ["same_module_dir"]
            else:
                file_score = 0.0
                reason_parts = []

            # 2. Symbol affinity (word-boundary & identifier specificity guard)
            symbol_score = 0.0
            if scar.target_symbol:
                sym_raw = scar.target_symbol.strip()
                sym_low = sym_raw.lower()
                if target_symbol and sym_low == target_symbol.strip().lower():
                    symbol_score = 1.0
                    reason_parts.append(f"symbol:{sym_raw}")
                elif (
                    planned_change
                    and (len(sym_low) >= 5 or "_" in sym_low or any(c.isupper() for c in sym_raw))
                    and re.search(rf"\b{re.escape(sym_low)}\b", planned_change.lower())
                ):
                    symbol_score = 0.9
                    reason_parts.append(f"symbol_in_plan:{sym_raw}")

            # 3. 16-dim Latent Scar Similarity
            latent_sim = (
                cosine_similarity_16d(query_16d, scar.latent_16d)
                if has_intent
                else 0.0
            )
            if latent_sim >= 0.45:
                reason_parts.append(f"latent_16d:{latent_sim:.2f}")

            # Combine via Gatekeeper policy:
            # Never pollute unrelated files unless same file/module or extreme semantic+symbol match
            if file_score >= 0.90:
                if has_intent:
                    combined = 0.55 * file_score + 0.20 * symbol_score + 0.25 * latent_sim
                else:
                    combined = 0.80 * file_score
            elif file_score > 0.0:
                # Same module directory: only trigger if symbol or 16-dim intent aligns
                combined = 0.25 * file_score + 0.35 * symbol_score + 0.40 * latent_sim
            else:
                # Different module: only trigger if both symbol and high latent similarity match
                if symbol_score >= 0.9 and latent_sim >= 0.65:
                    combined = 0.45 * symbol_score + 0.55 * latent_sim
                    reason_parts.append("cross_file_symbol_guard")
                else:
                    combined = 0.0

            if combined >= threshold:
                matched.append(
                    ScarMatch(
                        scar=scar,
                        similarity_score=round(min(1.0, combined), 4),
                        match_reason="+".join(reason_parts) or "file_match",
                    )
                )

        matched.sort(key=lambda m: m.similarity_score, reverse=True)
        top_matches = matched[: max(0, top_k)]

        if top_matches:
            with self._connect() as conn:
                for m in top_matches:
                    conn.execute(
                        "UPDATE scars SET hit_count = hit_count + 1 WHERE scar_id = ?",
                        (m.scar.scar_id,),
                    )
                    m.scar.hit_count += 1

        alerts = [m.scar.alert_50c for m in top_matches]
        raw_chars_total = sum(m.scar.raw_context_chars for m in top_matches)
        alert_chars_total = sum(len(a) for a in alerts)
        tokens_saved = max(0, int((raw_chars_total - alert_chars_total) / 3.0))

        return ScarCheckResponse(
            has_scars=len(top_matches) > 0,
            target_file=norm_file,
            alerts=alerts,
            matches=top_matches,
            tokens_saved_estimate=tokens_saved,
        )

    def get_stats(self) -> Dict[str, float | int]:
        """Compute repository-wide ScarTissue telemetry."""
        scars = self.list_all_scars()
        total_scars = len(scars)
        if total_scars == 0:
            return {
                "total_scars": 0,
                "total_interceptions": 0,
                "avg_alert_chars": 0.0,
                "max_alert_chars": 0,
                "compression_ratio_x": 0.0,
            }
        total_hits = sum(s.hit_count for s in scars)
        total_alert_chars = sum(len(s.alert_50c) for s in scars)
        total_raw_chars = sum(s.raw_context_chars for s in scars)
        max_len = max(len(s.alert_50c) for s in scars)
        return {
            "total_scars": total_scars,
            "total_interceptions": total_hits,
            "avg_alert_chars": round(total_alert_chars / total_scars, 1),
            "max_alert_chars": max_len,
            "compression_ratio_x": round(total_raw_chars / max(1, total_alert_chars), 1),
        }
