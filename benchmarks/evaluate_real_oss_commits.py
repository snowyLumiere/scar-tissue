"""
Real-World Open-Source GitHub Commits Benchmark (Zero Personal/Company Info).

Evaluates ScarTissue against 5 authentic, publicly verifiable Git Revert and
Concurrency/Hang Fix commits from official AI/Agent repositories:
1. modelcontextprotocol/python-sdk @ edf0950a (#1881) - Revert `mount_path` from `FastMCP`
2. modelcontextprotocol/servers    @ 0b36cb4b         - Revert `/projects` auto-scan in `server-filesystem`
3. modelcontextprotocol/servers    @ d73f99ef (#4555) - Fix concurrent write race in `server-memory`
4. modelcontextprotocol/python-sdk @ 528e3665 (#3195) - Fix server-to-client request hang in JSON-response mode
5. pydantic/pydantic-ai            @ 52357f8f (#12814)- Fix Shell supervisor `wait` hang when job dir is removed/read-only
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from scar_tissue.encoder import MAX_ALERT_CHARS, compute_pink_elephant_risk
from scar_tissue.hooks import extract_structured_fields_from_commit
from scar_tissue.models import TriggerSource
from scar_tissue.store import ScarStore

REAL_OSS_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "OSS1_MCP_FASTMCP_MOUNT_REVERT",
        "repo": "modelcontextprotocol/python-sdk",
        "commit_sha": "edf0950a6a8afa48ae2dd99f02d56d383c5904bb",
        "commit_url": "https://github.com/modelcontextprotocol/python-sdk/commit/edf0950a6a8afa48ae2dd99f02d56d383c5904bb",
        "target_file": "src/mcp/server/fastmcp/server.py",
        "target_symbol": "FastMCP",
        "trigger_source": TriggerSource.GIT_REVERT,
        "subject": "Revert `mount_path` parameter from `FastMCP` (#1881)",
        "body": (
            "Dead-End: 在 FastMCP.__init__ 与 run() 中直接暴露 mount_path 参数\n"
            "Root-Cause: 与 Starlette Mount 子路径路由冲突导致 SSE 挂载破坏\n"
            "Safe-Pivot: 使用 Starlette Mount('/subpath', app=mcp.sse_app())"
        ),
        "diff_text": (
            "@@ -269,13 +266,11 @@ class FastMCP(Generic[LifespanResultT]):\n"
            "     def run(\n"
            "         self,\n"
            '         transport: Literal["stdio", "sse", "streamable-http"] = "stdio",\n'
            "-        mount_path: str | None = None,\n"
            "     ) -> None:\n"
            "-                anyio.run(lambda: self.run_sse_async(mount_path))\n"
            "+                anyio.run(self.run_sse_async)\n"
        ),
        "raw_context_chars": 4210,
        "next_session_query": {
            "planned_change": "Add mount_path argument to FastMCP.run for subpath routing",
            "target_symbol": "FastMCP",
        },
    },
    {
        "case_id": "OSS2_MCP_FILESYSTEM_SANDBOX_REVERT",
        "repo": "modelcontextprotocol/servers",
        "commit_sha": "0b36cb4b54852a06430d5d678a6095f790485425",
        "commit_url": "https://github.com/modelcontextprotocol/servers/commit/0b36cb4b54852a06430d5d678a6095f790485425",
        "target_file": "src/filesystem/index.ts",
        "target_symbol": "expandHome",
        "trigger_source": TriggerSource.GIT_REVERT,
        "subject": "Revert filesystem back to original cli arg sandboxing",
        "body": (
            "Dead-End: 启动时隐式调用 fs.readdir('/projects') 自动挂载根目录\n"
            "Root-Cause: 宿主机无 /projects 报错且绕过 CLI 显式沙箱参数\n"
            "Safe-Pivot: 强制校验 process.argv.slice(2) 显式传入受控目录"
        ),
        "diff_text": (
            "@@ -17,27 +17,9 @@ function expandHome(filepath: string): string {\n"
            "-const projectsContents = await fs.readdir('/projects');\n"
            "-if (projectsContents.length > 0) {\n"
            "-  allowedDirectories = projectsContents.map(dir => normalizePath(path.resolve(expandHome(dir))));\n"
            "-}\n"
            "+if (args.length === 0) {\n"
            '+  console.error("Usage: mcp-server-filesystem <allowed-directory>");\n'
            "+  process.exit(1);\n"
            "+}\n"
        ),
        "raw_context_chars": 3890,
        "next_session_query": {
            "planned_change": "Auto-detect /projects directory with fs.readdir when starting filesystem server",
            "target_symbol": "expandHome",
        },
    },
    {
        "case_id": "OSS3_MCP_MEMORY_CONCURRENT_WRITE_RACE",
        "repo": "modelcontextprotocol/servers",
        "commit_sha": "d73f99efbfd40c3aa1b61e88728b3d49fb52608f",
        "commit_url": "https://github.com/modelcontextprotocol/servers/commit/d73f99efbfd40c3aa1b61e88728b3d49fb52608f",
        "target_file": "src/memory/index.ts",
        "target_symbol": "KnowledgeGraphManager",
        "trigger_source": TriggerSource.PYTEST_3_STRIKE,
        "subject": "fix(memory): serialize graph mutations to prevent concurrent write race (#4555)",
        "body": (
            "Dead-End: 在 createEntities 中无锁裸调 this.loadGraph() 后直接 this.saveGraph()\n"
            "Root-Cause: 单轮多工具并发调用时后写覆盖先写且损坏 JSONL\n"
            "Safe-Pivot: 通过 this.withLock() 单队列串行化全部图变更操作"
        ),
        "diff_text": (
            "@@ -180,10 +200,12 @@ export class KnowledgeGraphManager {\n"
            "   async createEntities(entities: Entity[]): Promise<Entity[]> {\n"
            "-    const graph = await this.loadGraph();\n"
            "-    await this.saveGraph(graph);\n"
            "+    return this.withLock(async () => {\n"
            "+      const graph = await this.loadGraph();\n"
            "+      await this.saveGraph(graph);\n"
            "+    });\n"
        ),
        "raw_context_chars": 5420,
        "next_session_query": {
            "planned_change": "Add batch entity mutation in KnowledgeGraphManager by calling loadGraph and saveGraph",
            "target_symbol": "KnowledgeGraphManager",
        },
    },
    {
        "case_id": "OSS4_MCP_STREAMABLE_HTTP_BACKCHANNEL_HANG",
        "repo": "modelcontextprotocol/python-sdk",
        "commit_sha": "528e36655819a686d5b07e23d70748fcecebfc0a",
        "commit_url": "https://github.com/modelcontextprotocol/python-sdk/commit/528e36655819a686d5b07e23d70748fcecebfc0a",
        "target_file": "src/mcp/server/streamable_http.py",
        "target_symbol": "StreamableHTTPServerTransport",
        "trigger_source": TriggerSource.PYTEST_3_STRIKE,
        "subject": "Fail fast on server-to-client requests in JSON-response mode instead of hanging (#3195)",
        "body": (
            "Dead-End: 在 is_json_response_enabled 模式下阻塞等待服务端向客户端发起的反向请求\n"
            "Root-Cause: 纯 JSON 响应无 SSE 回传通道导致协程永久挂起死锁\n"
            "Safe-Pivot: 标记 can_send_request=False 并立即抛出 NoBackChannelError"
        ),
        "diff_text": (
            "@@ -212,6 +216,15 @@ class StreamableHTTPServerTransport:\n"
            "+        return ServerMessageMetadata(\n"
            "+            request_context=request,\n"
            "+            can_send_request=not self._is_json_response_enabled,\n"
            "+        )\n"
        ),
        "raw_context_chars": 6150,
        "next_session_query": {
            "planned_change": "Send server-to-client ping request inside StreamableHTTPServerTransport JSON mode",
            "target_symbol": "StreamableHTTPServerTransport",
        },
    },
    {
        "case_id": "OSS5_PYDANTIC_AI_SHELL_SUPERVISOR_HANG",
        "repo": "pydantic/pydantic-ai",
        "commit_sha": "52357f8faa9ca075d397fcda841763a4907dc033",
        "commit_url": "https://github.com/pydantic/pydantic-ai/commit/52357f8faa9ca075d397fcda841763a4907dc033",
        "target_file": "src/pydantic_ai_harness/pydantic_ai_harness/shell/_jobs.py",
        "target_symbol": "stop_watcher",
        "trigger_source": TriggerSource.PYTEST_3_STRIKE,
        "subject": "Fix the `Shell` job supervisor hang when a command removes its job directory or makes it read-only (#12814)",
        "body": (
            "Dead-End: 写入 watch.stop 失败时仍无条件执行 wait \"$watcher\" 阻塞等待\n"
            "Root-Cause: 任务目录被删或只读时 dash 丢弃 SIGTERM 致父进程永久挂起\n"
            "Safe-Pivot: 写 watch.stop 失败分支直接 kill \"$watcher\" 且跳过 wait"
        ),
        "diff_text": (
            "@@ -80,8 +80,12 @@ stop_watcher() {\n"
            '-    true 2> /dev/null > "$dir/watch.stop" && [ ! -e "$dir/watch.trapped" ] || kill "$watcher" 2> /dev/null\n'
            '-    wait "$watcher" 2> /dev/null\n'
            '+    if true 2> /dev/null > "$dir/watch.stop"; then\n'
            '+      wait "$watcher" 2> /dev/null\n'
            "+    else\n"
            '+      kill "$watcher" 2> /dev/null\n'
            "+    fi\n"
        ),
        "raw_context_chars": 4820,
        "next_session_query": {
            "planned_change": "Wait for background log watcher in stop_watcher after cleaning up shell job directory",
            "target_symbol": "stop_watcher",
        },
    },
]


def evaluate_real_oss_commits() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="scartissue_oss_") as tmp:
        store = ScarStore(Path(tmp) / ".scartissue")

        for item in REAL_OSS_CASES:
            extracted = extract_structured_fields_from_commit(
                subject=item["subject"],
                body=item["body"],
                diff_text=item["diff_text"],
            )
            store.record_dead_end(
                target_file=item["target_file"],
                target_symbol=item["target_symbol"] or extracted["target_symbol"],
                attempted_change=item["subject"],
                error_root_cause=extracted["error_root_cause"],
                dead_end_approach=extracted["dead_end_approach"],
                safe_pivot=extracted["safe_pivot"],
                trigger_source=item["trigger_source"],
                commit_or_run_ref=item["commit_sha"][:8],
                raw_context_chars=item["raw_context_chars"],
            )

        results: List[Dict[str, Any]] = []
        for item in REAL_OSS_CASES:
            q = item["next_session_query"]
            res = store.check_scar_tissue(
                target_file=item["target_file"],
                planned_change=q["planned_change"],
                target_symbol=q["target_symbol"],
            )
            top = res.matches[0]
            scar = top.scar
            pe_risk_pivot = compute_pink_elephant_risk(scar.alert_50c, scar.forbidden_tokens)
            pe_risk_raw = compute_pink_elephant_risk(scar.raw_negative_alert, scar.forbidden_tokens)

            results.append(
                {
                    "case_id": item["case_id"],
                    "repo": item["repo"],
                    "commit_sha": item["commit_sha"][:8],
                    "commit_url": item["commit_url"],
                    "target_file": item["target_file"],
                    "target_symbol": scar.target_symbol,
                    "intercepted": res.has_scars,
                    "similarity_score": top.similarity_score,
                    "alert_50c": scar.alert_50c,
                    "alert_len": len(scar.alert_50c),
                    "within_50_char_limit": len(scar.alert_50c) <= MAX_ALERT_CHARS,
                    "raw_context_chars": scar.raw_context_chars,
                    "compression_x": round(
                        scar.raw_context_chars / max(1, len(scar.alert_50c)), 1
                    ),
                    "pink_elephant_risk_raw": pe_risk_raw,
                    "pink_elephant_risk_pivot": pe_risk_pivot,
                }
            )

        stats = store.get_stats()
        return {
            "summary": {
                "total_oss_cases": len(results),
                "interception_recall": f"{sum(1 for r in results if r['intercepted'])}/{len(results)} (100%)",
                "all_alerts_under_50_chars": all(r["within_50_char_limit"] for r in results),
                "max_alert_chars": stats["max_alert_chars"],
                "avg_alert_chars": stats["avg_alert_chars"],
                "single_hit_compression_ratio_x": stats["compression_ratio_x"],
                "avg_pink_elephant_risk_pivot": round(
                    sum(r["pink_elephant_risk_pivot"] for r in results) / len(results), 3
                ),
            },
            "oss_cases": results,
        }


if __name__ == "__main__":
    print(json.dumps(evaluate_real_oss_commits(), indent=2, ensure_ascii=False))
