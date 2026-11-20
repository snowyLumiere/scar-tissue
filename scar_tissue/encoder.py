"""
Memorilla-inspired 16-Dimensional Latent Scar Gatekeeper & Anti-Pink-Elephant Prompt Compiler.

1. Compresses verbose git revert diffs + stack traces into a 16-float unit vector (`latent_16d`)
   for sub-millisecond local gatekeeping without requiring GPU soft-prompt injection.
2. Compiles dead-end records into strictly <= 50 character high-priority redline alerts
   using Contrastive Pivot Rewriting to eliminate the LLM 'Pink Elephant Effect'.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Tuple

from scar_tissue.models import FailureArchetype

MAX_ALERT_CHARS = 50

# 8 orthogonal semantic failure-mechanism channels (d[0]..d[7])
CHANNEL_KEYWORDS: List[Tuple[FailureArchetype, List[str]]] = [
    (
        FailureArchetype.CONCURRENCY_LOCK,
        [
            "lock", "deadlock", "mutex", "semaphore", "barrier", "all_reduce",
            "ddp", "nccl", "race", "thread", "concurrent", "死锁", "加锁",
            "并发", "全局锁", "同步阻塞", "屏障", "竞争",
        ],
    ),
    (
        FailureArchetype.MEMORY_OOM_SHAPE,
        [
            "oom", "out of memory", "cuda", "broadcast", "shape", "tensor",
            "kv_cache", "attention_mask", "unsqueeze", "expand", "leak",
            "显存", "内存溢出", "维度", "广播", "张量", "掩码", "缓存膨胀",
        ],
    ),
    (
        FailureArchetype.IO_BUFFER_DEADLOCK,
        [
            "pipe", "stdout", "stderr", "subprocess", "communicate", "buffer",
            "readline", "wait", "stdio", "eof", "hang", "管道", "缓冲区",
            "子进程", "挂起", "阻塞等待", "标准输出",
        ],
    ),
    (
        FailureArchetype.STATE_SYNC_LOOP,
        [
            "useeffect", "rerender", "infinite", "loop", "state", "setstate",
            "dispatch", "closure", "dependency", "hydration", "重渲染",
            "死循环", "状态同步", "无限循环", "依赖项", "双向绑定",
        ],
    ),
    (
        FailureArchetype.DATABASE_TX_LOCK,
        [
            "sqlite", "database is locked", "operationalerror", "wal", "pool",
            "transaction", "commit", "cursor", "connection", "数据库锁",
            "连接池", "事务冲突", "写锁", "单连接", "并发写",
        ],
    ),
    (
        FailureArchetype.PATH_FS_BOUNDARY,
        [
            "path", "enoent", "inode", "cwd", "directory", "relative",
            "filesystem", "symlink", "permission", "路径", "目录", "相对路径",
            "文件丢失", "句柄",
        ],
    ),
    (
        FailureArchetype.SCHEMA_CONTRACT_DRIFT,
        [
            "schema", "validationerror", "typeerror", "keyerror", "pydantic",
            "json", "contract", "null", "none", "serialization", "契约",
            "字段缺失", "类型不匹配", "反序列化", "静默失败",
        ],
    ),
    (
        FailureArchetype.PERF_TIMEOUT_COMPLEXITY,
        [
            "timeout", "retry", "backoff", "latency", "slow", "complexity",
            "o(n^2)", "index", "recursion", "超时", "重试风暴", "性能退化",
            "栈溢出",
        ],
    ),
]

# Lexical de-anchoring map: replaces literal copy-pasteable code anti-patterns
# with conceptual hazard labels so the LLM does not copy-attend to forbidden API names.
LEXICAL_DEANCHOR_MAP: Dict[str, str] = {
    "dist.barrier()": "跨卡同步屏障",
    "dist.barrier": "跨卡同步屏障",
    "torch.distributed.barrier": "跨卡同步屏障",
    "asyncio.lock()": "协程全局锁",
    "asyncio.lock": "协程全局锁",
    "threading.lock()": "线程互斥锁",
    "threading.lock": "线程互斥锁",
    "subprocess.pipe": "未消费管道缓冲",
    "proc.wait()": "阻塞等待子进程",
    "process.wait()": "阻塞等待子进程",
    "useeffect": "副作用钩子",
    "setsearchparams": "同步写路由参数",
    "check_same_thread=false": "跨线程裸共享连接",
    ".unsqueeze(1).expand()": "扩维广播",
    "tensor.expand": "隐式高维张量扩展",
    ".unsqueeze(": "扩维广播(",
    "fastmcp.__init__": "构造器直接暴露挂载参数",
    "fs.readdir('/projects')": "硬编码扫描根目录",
    "this.loadgraph()": "无锁裸读图状态",
    "this.savegraph(": "无锁直接覆写图(",
    'wait "$watcher"': "无条件阻塞等待守护进程",
    "watch.stop": "停止标记文件",
}

LEXICAL_DEANCHOR_MAP_EN: Dict[str, str] = {
    "dist.barrier()": "cross-rank barrier",
    "dist.barrier": "cross-rank barrier",
    "torch.distributed.barrier": "cross-rank barrier",
    "asyncio.lock()": "coroutine lock",
    "asyncio.lock": "coroutine lock",
    "threading.lock()": "thread mutex",
    "threading.lock": "thread mutex",
    "subprocess.pipe": "unconsumed pipe",
    "proc.wait()": "blocking wait",
    "process.wait()": "blocking wait",
    "useeffect": "effect hook",
    "setsearchparams": "sync URL param write",
    "check_same_thread=false": "shared thread conn",
    ".unsqueeze(1).expand()": "4D mask expand",
    "tensor.expand": "implicit tensor expand",
    ".unsqueeze(": "tensor unsqueeze(",
    "fastmcp.__init__": "constructor mount arg",
    "fs.readdir('/projects')": "hardcoded root scan",
    "this.loadgraph()": "unlocked graph read",
    "this.savegraph(": "unlocked graph write(",
    'wait "$watcher"': "unconditional watcher wait",
    "watch.stop": "stop marker file",
}


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def scrub_sensitive_info(text: str) -> str:
    """
    Strip personal OS home paths, email addresses, internal corporate domains,
    and API keys/tokens so no PII or company info ever enters `.scartissue/`.
    """
    if not text:
        return ""
    out = text
    # 1. Strip macOS / Linux / Windows user home directory paths
    out = re.sub(r"/Users/[^/\s\"':]+", "<HOME>", out)
    out = re.sub(r"/home/[^/\s\"':]+", "<HOME>", out)
    out = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"':]+", "<HOME>", out, flags=re.IGNORECASE)
    # 2. Strip email addresses
    out = re.sub(
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
        "<REDACTED_EMAIL>",
        out,
    )
    # 3. Strip common secret/token prefixes (sk-..., ghp_..., AKIA...)
    out = re.sub(
        r"\b(?:sk-[a-zA-Z0-9_-]{12,}|gh[pousr]_[a-zA-Z0-9]{12,}|AKIA[0-9A-Z]{16})\b",
        "<REDACTED_SECRET>",
        out,
    )
    # 4. Strip internal corporate hostnames (*-inc.com, *.internal, *.corp.*)
    out = re.sub(
        r"\b(?:https?://)?[a-zA-Z0-9.-]*(?:-inc\.com|\.corp\.[a-zA-Z0-9.-]+|\.internal)\b",
        "<REDACTED_CORP>",
        out,
        flags=re.IGNORECASE,
    )
    return out


def extract_forbidden_tokens(text: str) -> List[str]:
    """
    Extract code-like API calls, identifiers, and expressions from dead-end descriptions
    that could trigger the LLM Pink Elephant Effect if echoed verbatim.
    """
    if not text:
        return []
    patterns = [
        r"`([^`]{2,40})`",
        r"(\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)(?:\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))*)",
        r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\(?\)?)",
        r"\b(useEffect|setState|asyncio\.Lock|dist\.barrier|subprocess\.PIPE|proc\.wait)\b",
    ]
    found: List[str] = []
    for pat in patterns:
        for match in re.findall(pat, text):
            token = match.strip()
            if len(token) >= 3 and token not in found:
                found.append(token)
    return found[:8]


def sanitize_code_literals(text: str, use_cjk: bool | None = None) -> str:
    """
    Replace literal code tokens/snippets in dead-end descriptions with conceptual hazard
    labels to prevent autoregressive copy-attention (Pink Elephant Effect).
    """
    if use_cjk is None:
        use_cjk = _has_cjk(text)
    result = text
    # Strip backticks around code
    result = re.sub(r"`([^`]+)`", r"\1", result)
    # Apply known lexical de-anchoring replacements (case-insensitive)
    deanchor_map = LEXICAL_DEANCHOR_MAP if use_cjk else LEXICAL_DEANCHOR_MAP_EN
    for literal, concept in deanchor_map.items():
        pattern = re.compile(re.escape(literal), re.IGNORECASE)
        result = pattern.sub(concept, result)
    # Replace remaining dotted method calls like foo.bar() with conceptual token
    fallback_label = "该高危API调用" if use_cjk else "risky API call"
    result = re.sub(r"\b[a-zA-Z_]\w*\.[a-zA-Z_]\w*\([^)]*\)", fallback_label, result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _keyword_matches(kw: str, text_lower: str) -> bool:
    """
    Word-boundary-aware keyword matcher so short ASCII keywords like 'race', 'oom', 'wal'
    do not false-match substrings inside 'traceback', 'room', or 'invalid'.
    """
    if re.search(r"[\u4e00-\u9fff]", kw):
        return kw in text_lower
    if re.match(r"^[a-z0-9_]+$", kw):
        return bool(re.search(rf"\b{re.escape(kw)}\b", text_lower))
    return kw in text_lower


def _extract_structural_tokens(text_lower: str) -> List[str]:
    """
    Extract ASCII identifier words plus overlapping CJK character bigrams (and unigrams)
    so Chinese phrasing variations do not suffer from non-overlapping parity shift.
    """
    tokens: List[str] = re.findall(r"[a-z0-9_]+", text_lower)
    for cjk_run in re.findall(r"[\u4e00-\u9fff]+", text_lower):
        if len(cjk_run) == 1:
            tokens.append(cjk_run)
        else:
            for i in range(len(cjk_run) - 1):
                tokens.append(cjk_run[i : i + 2])
    return tokens


def encode_16d_scar_vector(
    target_file: str,
    target_symbol: str,
    attempted_change: str,
    error_root_cause: str,
    dead_end_approach: str,
) -> Tuple[List[float], FailureArchetype]:
    """
    Encode a dead-end event into a 16-dimensional L2-normalized latent vector:
    - Dimensions [0..7]: Semantic failure mechanism activation channels
    - Dimensions [8..15]: Signed locality-sensitive n-gram & symbol projection channels
    """
    combined = " ".join(
        [target_file, target_symbol, attempted_change, error_root_cause, dead_end_approach]
    ).lower()

    vec = [0.0] * 16
    best_archetype = FailureArchetype.GENERAL_REGRESSION
    best_score = 0.0

    # Channels 0..7: Failure mode semantics with word-boundary protection
    for idx, (archetype, keywords) in enumerate(CHANNEL_KEYWORDS):
        hits = sum(1.0 for kw in keywords if _keyword_matches(kw, combined))
        score = math.tanh(hits * 0.65)
        vec[idx] = score
        if score > best_score:
            best_score = score
            best_archetype = archetype

    # Channels 8..15: Signed locality-sensitive feature hashing (overlapping CJK bigrams + ASCII words)
    structural_source = f"{target_file.lower()}::{target_symbol.lower()}::{combined}"
    tokens = _extract_structural_tokens(structural_source)
    sym_file_low = f"{target_file.lower()} {target_symbol.lower()}"
    for tok in tokens:
        digest = hashlib.md5(tok.encode("utf-8")).digest()
        bucket = 8 + (digest[0] % 8)
        sign = 1.0 if (digest[1] & 1) == 0 else -1.0
        weight = 1.5 if tok in sym_file_low else 0.5
        vec[bucket] += sign * weight

    # Squash structural channels to [-1, 1]
    for i in range(8, 16):
        vec[i] = math.tanh(vec[i] * 0.35)

    # L2 normalize across all 16 dimensions
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-8:
        vec = [round(v / norm, 4) for v in vec]
    else:
        vec[0] = 1.0

    return vec, best_archetype


def cosine_similarity_16d(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two 16-dimensional latent scar vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _compact_phrase(
    text: str,
    max_len: int,
    sanitize_literals: bool = True,
    use_cjk: bool | None = None,
) -> str:
    """
    Clean and compact a phrase to at most `max_len` characters.
    - When `sanitize_literals=True` (for dead_end / cause), strips dangerous code literals.
    - When `sanitize_literals=False` (for safe_pivot), preserves positive API call names.
    - Removes spaces adjacent to CJK characters while preserving single spaces between English words.
    """
    if sanitize_literals:
        cleaned = sanitize_code_literals(text, use_cjk=use_cjk)
    else:
        cleaned = re.sub(r"`([^`]+)`", r"\1", text).strip()

    # Strip verbose exception class prefixes like 'sqlite3.OperationalError:' or 'RuntimeError:'
    cleaned = re.sub(r"^[A-Za-z0-9_.]*(?:Error|Exception|Fault|Timeout):\s*", "", cleaned)

    # If there is a concise parenthetical summary at the end like 'database is locked (并发写锁冲突)', prefer it if shorter
    paren_match = re.search(r"\(([\u4e00-\u9fffA-Za-z0-9_]{4,24})\)\s*$", cleaned)
    if paren_match and len(cleaned) > max_len:
        cleaned = paren_match.group(1)

    cleaned = re.sub(
        r"^(试图|尝试|通过|因为|由于|导致|触发|报错|禁止|严禁|请改用|必须用|改用|使用|直接|在)\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[。；;，,！!？?\n\r\t]+", " ", cleaned)
    # Remove spaces adjacent to CJK characters or operators, but preserve spaces between English words
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = re.sub(r"\s*([=<>!]+)\s*", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max(1, max_len - 1)].rstrip() + "…"


def compile_negative_constraint(
    dead_end_approach: str,
    error_root_cause: str,
    safe_pivot: str = "",
    max_chars: int = MAX_ALERT_CHARS,
) -> Tuple[str, str]:
    """
    Compile a dead-end record into:
    1. `alert_50c`: Strictly <= 50 character Contrastive Pivot redline alert
       that prevents the Pink Elephant Effect (de-anchored hazard + preserved positive pivot).
       Automatically adapts template to CJK (`⛔忌...→用...`) or English (`⛔NO ...->USE ...`).
    2. `raw_negative_alert`: Traditional negative warning (for ablation benchmarking).
    """
    use_cjk = _has_cjk(f"{dead_end_approach} {error_root_cause} {safe_pivot}")
    if use_cjk:
        raw_negative = f"警告：严禁使用 {dead_end_approach}，此前已触发 {error_root_cause} 并被回滚！"
    else:
        raw_negative = (
            f"WARNING: Do NOT use {dead_end_approach}; previously triggered {error_root_cause} and was reverted!"
        )

    if use_cjk:
        # Format with safe_pivot: "⛔忌{hazard}({cause})→用{pivot}"
        # Prefix/delimiters take 6 chars: "⛔忌" (2) + "(" (1) + ")→用" (3) = 6 chars -> 44 chars for content
        if safe_pivot and safe_pivot.strip():
            avail = max_chars - 6
            hazard_budget = max(8, int(avail * 0.34))
            cause_budget = max(6, int(avail * 0.25))

            hazard_str = _compact_phrase(
                dead_end_approach, hazard_budget, sanitize_literals=True, use_cjk=True
            )
            cause_str = _compact_phrase(
                error_root_cause, cause_budget, sanitize_literals=True, use_cjk=True
            )
            pivot_budget = max(8, avail - len(hazard_str) - len(cause_str))
            pivot_str = _compact_phrase(
                safe_pivot, pivot_budget, sanitize_literals=False, use_cjk=True
            )
            alert = f"⛔忌{hazard_str}({cause_str})→用{pivot_str}"
        else:
            suffix = "；须换安全方案"
            # Fixed chars: "⛔忌" (2) + "致" (1) + suffix (7) = 10 chars
            avail = max(4, max_chars - (3 + len(suffix)))
            hazard_budget = max(2, int(avail * 0.55))
            cause_budget = max(2, avail - hazard_budget)
            hazard_str = _compact_phrase(
                dead_end_approach, hazard_budget, sanitize_literals=True, use_cjk=True
            )
            cause_str = _compact_phrase(
                error_root_cause, cause_budget, sanitize_literals=True, use_cjk=True
            )
            alert = f"⛔忌{hazard_str}致{cause_str}{suffix}"
    else:
        # English template: "⛔NO {hazard}({cause})->USE {pivot}"
        # Prefix/delimiters take 11 chars: "⛔NO " (4) + "(" (1) + ")->USE " (6) = 11 chars -> 39 chars for content
        if safe_pivot and safe_pivot.strip():
            avail = max_chars - 11
            hazard_budget = max(8, int(avail * 0.36))
            cause_budget = max(6, int(avail * 0.26))

            hazard_str = _compact_phrase(
                dead_end_approach, hazard_budget, sanitize_literals=True, use_cjk=False
            )
            cause_str = _compact_phrase(
                error_root_cause, cause_budget, sanitize_literals=True, use_cjk=False
            )
            pivot_budget = max(8, avail - len(hazard_str) - len(cause_str))
            pivot_str = _compact_phrase(
                safe_pivot, pivot_budget, sanitize_literals=False, use_cjk=False
            )
            alert = f"⛔NO {hazard_str}({cause_str})->USE {pivot_str}"
        else:
            suffix = ")->USE safe alt"
            # Fixed chars: "⛔NO " (4) + "(" (1) + suffix (15) = 20 chars
            avail = max(4, max_chars - (5 + len(suffix)))
            hazard_budget = max(2, int(avail * 0.55))
            cause_budget = max(2, avail - hazard_budget)
            hazard_str = _compact_phrase(
                dead_end_approach, hazard_budget, sanitize_literals=True, use_cjk=False
            )
            cause_str = _compact_phrase(
                error_root_cause, cause_budget, sanitize_literals=True, use_cjk=False
            )
            alert = f"⛔NO {hazard_str}({cause_str}{suffix}"

    if len(alert) > max_chars:
        alert = alert[: max_chars - 1] + "…"

    return alert, raw_negative


def compute_pink_elephant_risk(alert_text: str, forbidden_tokens: List[str]) -> float:
    """
    Evaluate the risk [0.0 .. 1.0] that an alert string triggers the LLM Pink Elephant Effect:
    - Penalizes literal leakage of forbidden code identifiers/snippets in the negative hazard clause
    - Penalizes pure negation without a positive replacement branch ('→用' / '->USE ' / '改用')
    """
    if not alert_text:
        return 0.0
    negative_clause = re.split(r"→用|->USE\s+", alert_text, flags=re.IGNORECASE)[0].lower()
    leaked = sum(1 for tok in forbidden_tokens if tok.lower() in negative_clause)
    leak_ratio = min(1.0, leaked / max(1, len(forbidden_tokens))) if forbidden_tokens else 0.0
    has_positive_pivot = (
        ("→用" in alert_text)
        or ("->use " in alert_text.lower())
        or ("改用" in alert_text)
        or ("须换" in alert_text)
    )
    pivot_penalty = 0.0 if has_positive_pivot else 0.45
    return round(min(1.0, leak_ratio * 0.55 + pivot_penalty), 3)
