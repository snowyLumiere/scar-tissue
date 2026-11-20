"""Unit tests for 16-dim Memorilla-inspired Scar Encoder and <=50-char Anti-Pink-Elephant Compiler."""

import math
from scar_tissue.encoder import (
    MAX_ALERT_CHARS,
    _compact_phrase,
    _keyword_matches,
    compile_negative_constraint,
    compute_pink_elephant_risk,
    cosine_similarity_16d,
    encode_16d_scar_vector,
    extract_forbidden_tokens,
    scrub_sensitive_info,
)
from scar_tissue.models import FailureArchetype


def test_16d_vector_normalization_and_archetype():
    vec, arch = encode_16d_scar_vector(
        target_file="trainer/ddp_runner.py",
        target_symbol="sync_shard_metrics",
        attempted_change="Added dist.barrier() inside rank 0 check",
        error_root_cause="NCCL deadlock across ranks",
        dead_end_approach="Conditional dist.barrier()",
    )
    assert len(vec) == 16
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-3
    assert arch == FailureArchetype.CONCURRENCY_LOCK


def test_word_boundary_prevents_substring_false_matches():
    # 'race' must not match 'traceback'; 'oom' must not match 'room'; 'wal' must not match 'invalid'
    assert _keyword_matches("race", "standard python traceback line") is False
    assert _keyword_matches("race", "thread race condition detected") is True
    assert _keyword_matches("oom", "classroom layout") is False
    assert _keyword_matches("oom", "cuda oom error") is True
    assert _keyword_matches("wal", "invalid argument walk") is False
    assert _keyword_matches("wal", "enable sqlite wal mode") is True


def test_16d_similarity_ranking_and_cjk_bigram_stability():
    vec_lock_a, _ = encode_16d_scar_vector(
        "storage/db.py", "write_tx", "在协程中使用全局并发死锁", "数据库并发写锁冲突", "加锁死路"
    )
    vec_lock_b, _ = encode_16d_scar_vector(
        "storage/db.py", "write_tx", "协程中使用全局并发死锁", "并发写锁冲突", "加锁死路"
    )
    vec_unrelated, _ = encode_16d_scar_vector(
        "ui/theme.tsx", "Button", "change border radius css", "visual alignment", "padding"
    )
    sim_related = cosine_similarity_16d(vec_lock_a, vec_lock_b)
    sim_unrelated = cosine_similarity_16d(vec_lock_a, vec_unrelated)
    assert sim_related > 0.85
    assert sim_related > sim_unrelated + 0.35


def test_strict_50_char_alert_ceiling():
    huge_dead_end = "在高频异步循环中反复调用 `asyncio.Lock()` 与 `dist.barrier()` " * 20
    huge_cause = "触发跨进程 EventLoop 死锁与显存 OOM 崩溃导致全部训练节点挂起 " * 20
    huge_pivot = "改用无锁环形队列和单写协程异步落盘并启用预分配缓冲区 " * 20

    alert_50c, raw_neg = compile_negative_constraint(
        dead_end_approach=huge_dead_end,
        error_root_cause=huge_cause,
        safe_pivot=huge_pivot,
    )
    assert len(alert_50c) <= MAX_ALERT_CHARS
    assert alert_50c.startswith("⛔忌")
    assert "→用" in alert_50c
    assert len(raw_neg) > 500


def test_anti_pink_elephant_deanchoring_preserves_safe_pivot_api():
    dead_end = "挂载 `subprocess.PIPE` 后调用 `proc.wait()`"
    cause = "64KB 管道写满死锁"
    pivot = "改用 `proc.communicate()` 流式消费"

    forbidden = extract_forbidden_tokens(dead_end)
    assert "proc.wait()" in forbidden

    alert_50c, raw_neg = compile_negative_constraint(dead_end, cause, pivot)
    assert len(alert_50c) <= MAX_ALERT_CHARS
    # Forbidden literal code token must be stripped from hazard part
    assert "proc.wait" not in alert_50c
    assert "subprocess.PIPE" not in alert_50c
    # Positive API in safe_pivot MUST be preserved (not turned into '该高危API调用')!
    assert "该高危API调用" not in alert_50c
    assert "proc.communicate()" in alert_50c

    risk_pivot = compute_pink_elephant_risk(alert_50c, forbidden)
    risk_raw = compute_pink_elephant_risk(raw_neg, forbidden)
    assert risk_pivot == 0.0
    assert risk_raw >= 0.8


def test_cjk_english_mixed_spacing_preserved():
    compact = _compact_phrase("启用 SDPA is causal flag 免实例化", max_len=40, sanitize_literals=False)
    assert "SDPA is causal flag" in compact
    assert "SDPAiscausalflag" not in compact


def test_pure_english_alert_adaptation_and_deanchoring():
    dead_end = "rank 0 conditional dist.barrier()"
    cause = "NCCL deadlock across ranks"
    pivot = "all_reduce across all ranks"

    forbidden = extract_forbidden_tokens(dead_end)
    assert "dist.barrier()" in forbidden

    alert_50c, raw_neg = compile_negative_constraint(dead_end, cause, pivot)
    assert len(alert_50c) <= MAX_ALERT_CHARS
    assert alert_50c.startswith("⛔NO ")
    assert "->USE " in alert_50c
    assert "dist.barrier" not in alert_50c
    assert "all_reduce" in alert_50c
    # Ensure no Chinese characters leak into a pure-English alert
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in alert_50c)
    assert compute_pink_elephant_risk(alert_50c, forbidden) == 0.0
    assert compute_pink_elephant_risk(raw_neg, forbidden) >= 0.8


def test_privacy_scrubber_strips_personal_and_company_info():
    raw_trace = (
        'File "/Users/alice_dev/company-repo/src/db.py", line 42, in connect\n'
        r'WinFile "C:\Users\bob_win\corp\app.py", line 7' + "\n"
        "Contact: alice.dev@example-inc.com or https://gitlab.example-inc.com/proj\n"
        "Token: sk-1234567890abcdef12345678"
    )
    scrubbed = scrub_sensitive_info(raw_trace)
    assert "/Users/alice_dev" not in scrubbed
    assert "<HOME>/company-repo/src/db.py" in scrubbed
    assert "bob_win" not in scrubbed
    assert r"<HOME>\corp\app.py" in scrubbed
    assert "alice.dev@example-inc.com" not in scrubbed
    assert "<REDACTED_EMAIL>" in scrubbed
    assert "example-inc.com" not in scrubbed
    assert "<REDACTED_CORP>" in scrubbed
    assert "sk-1234567890" not in scrubbed
    assert "<REDACTED_SECRET>" in scrubbed


def test_no_pivot_alerts_preserve_generic_suffix_within_50_chars():
    cjk_alert, _ = compile_negative_constraint(
        "在高频异步循环中调用极其漫长的高危阻塞实现方式名称测试甲乙丙丁",
        "触发非常严重的超时与错误根因描述测试甲乙丙丁戊己",
        "",
    )
    assert len(cjk_alert) <= MAX_ALERT_CHARS
    assert cjk_alert.endswith("；须换安全方案")
    assert "异步无锁" not in cjk_alert

    en_alert, _ = compile_negative_constraint(
        "extremely verbose dead end pattern that exceeds normal alert limits",
        "very long root cause exception explanation string that would overflow",
        "",
    )
    assert len(en_alert) <= MAX_ALERT_CHARS
    assert en_alert.endswith(")->USE safe alt")


