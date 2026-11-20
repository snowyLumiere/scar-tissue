# ScarTissue：别再往 CLAUDE.md 里塞垃圾了——只记“死路”的 Anti-Memory MCP 插件

> **从 Stanford `Memorilla` (arXiv:2610.21375) 的 16-Token 隐空间记忆压缩与 LoreKit HN 记忆碎片化大讨论出发，将“卷通用闲聊记忆准确率”重构为“多 Agent 研发项目的隐式避坑疫苗（Anti-Memory / Scar Tissue）”。**

---

## 一、 为什么需要 ScarTissue？（痛点与两级跳推演）

1. **文本文档记忆（`CLAUDE.md` / `MEMORY.md`）的工程死穴**：
   - 超过 200 行即遭截断或引发严重的指令注意力稀释（Instruction Dilution）。
   - 混杂陈旧笔记会跨会话污染上下文；人工手写维护成本极高。
2. **闭源 Coding Agent 的最大真实痛点——“在同一个坑里反复横跳”**：
   - 高强度使用 Claude Code / Cursor / Codex 的开发者最痛的不是“AI 记不住缩进风格”，而是**昨天刚因为隐蔽的 NCCL 跨卡死锁、SQLite 并发锁死或 32K 序列掩码 OOM 撤回（`git revert`）了一种改法，今天开新 Session，AI 又自信满满地把同样的死路重写了一遍**。
3. **从 Stanford `Memorilla` 到本地极速“记忆守门员（Memory Gatekeeper）”**：
   - `Memorilla` 证明了将长历史编码为 **16 个紧凑隐空间向量** 可节省 108—2,000 倍 Token 并减少 18.6%—24.2% 的无效原地打转（Wandering），但原生 Soft Prompt 绑定特定开源权重（Qwen3-8B）且人类不可读不可删改。
   - **ScarTissue 的工程降维与破局**：
     - **双触发器零人工采集**：通过极简 Git Hook（捕获 `git revert` / `git reset`）与单测连败追踪器（捕获 **连续 3 次跑挂单测后终于修好** 的 Crash-Loop Recovery），自动提炼 `{当时试图做的改动 + 触发的报错根因 + 最终证明不可行的死路 + 验证可行的替代正路}`。
     - **16 维轻量隐空间特征守门员 + 人类可审计双写存储**：将每条死路编码为 16 维归一化语义/结构向量（8 维故障机理通道 + 8 维局部敏感结构哈希通道），同时落地至 `.scartissue/scars.db`（SQLite WAL）与人类可直接增删的 `.scartissue/scars.jsonl`。
     - **防“粉红大象效应（Anti-Pink-Elephant）”的 ≤50 字中英自适应红线警告编译器**：在下次 Agent 准备修改同一模块/符号时，不塞入几万字历史，而是精准弹出一条 **严格 ≤ 50 字** 的对比式正向转向警告（中文：`⛔忌[去字面化死路]([根因])→用[安全正路]`；英文：`⛔NO [de-anchored hazard]([cause])->USE [safe pivot]`），剥离危险 API 原文字面量，彻底消除 LLM 因负向提示注意力聚焦反而照抄错误代码的“粉红大象效应”。

---

## 二、 架构全景与目录结构

```text
scar-tissue/
├── .github/workflows/ci.yml            # GitHub Actions CI (Python 3.10–3.12, wheel & npm 只读冒烟测试)
├── .gitignore                          # 过滤构建产物、缓存与本地 SQLite/连败状态
├── .npmignore                          # npm 打包过滤配置
├── README.md                           # 项目文档与一键挂载指南
├── LICENSE                             # Apache-2.0 完整开源许可证
├── pyproject.toml                      # Python 包与 CLI 入口配置 (scar-tissue / scar-tissue-mcp)
├── package.json                        # 支持 npx scar-tissue-mcp / scar-tissue 一键启动
├── bin/
│   └── scar-tissue.js                  # npx 跨平台轻量包装器 (支持 Python>=3.10 / uv --no-project 自动探测)
├── scar_tissue/
│   ├── __init__.py
│   ├── __main__.py                     # 支持 `python -m scar_tissue` 直接执行
│   ├── models.py                       # 核心数据契约：ScarRecord, ScarMatch, ScarCheckResponse
│   ├── encoder.py                      # 16 维 Memorilla 风格特征守门员 + 防粉红大象 <=50 字编译器
│   ├── store.py                        # SQLite (WAL) + JSONL 双向原子同步存储与混合检索器
│   ├── hooks.py                        # Git Revert/Reset 自动提取器 + 连续 3 次单测崩溃恢复捕获器
│   ├── server.py                       # ~150 行极简 MCP Server 入口 (record_dead_end & check_scar_tissue)
│   └── cli.py                          # 命令行工具：MCP 服务、Git Hook 挂载、真实测试运行与终端检索
├── benchmarks/
│   ├── evaluate_real_oss_commits.py    # 5 个真实开源仓库 (MCP SDK/Servers, Pydantic-AI) Commit 实测脚本
│   └── evaluate_5_scenarios.py         # 5 大经典高并发/OOM/死锁工程踩坑沙箱端到端复现脚本
├── reports/
│   └── SCAR_TISSUE_VALIDATION_REPORT.md # 10 大案例实测报告与防粉红大象消融对比
└── tests/                              # 单元与端到端集成测试套件
    ├── test_encoder_and_limit.py
    ├── test_store_and_gatekeeper.py
    ├── test_git_and_pytest_hooks.py
    └── test_mcp_server.py
```

---

## 三、 快速开始与一键挂载（Claude Code / Cursor / Codex）

### 0. 环境前置条件（Prerequisites）

- **Python CLI / PyPI 模式**：需要 **Python >= 3.10**（运行时依赖 `mcp>=1.0.0,<2.0.0` 与 `pydantic>=2.0.0`）。
- **`npx` 零配置模式**：需要 **Node.js >= 18**，以及已安装 `scar-tissue` 依赖的 Python >= 3.10 环境 **或** 已安装 [`uv`](https://docs.astral.sh/uv/)（包装器会在无环境时自动通过只读缓存安全的 `uv run --no-project` 拉起）。

### 1. 安装与在 Claude Code / Cursor 中挂载 MCP Server

#### 方式 A：通过 PyPI 安装（推荐）

```bash
pip install scar-tissue
scar-tissue --version
```

#### 方式 B：从 GitHub 克隆本地开发安装

```bash
git clone https://github.com/SnowyLumiere/scar-tissue.git
cd scar-tissue
pip install -e ".[dev]"
scar-tissue --version
```

安装完成后，在项目的 `.mcp.json` 或 Cursor MCP 配置中添加：

```json
{
  "mcpServers": {
    "scar-tissue": {
      "command": "scar-tissue-mcp",
      "args": []
    }
  }
}
```

#### 方式 C：通过 `npx` 启动（发布至 npm 后支持一键零配置挂载）

```json
{
  "mcpServers": {
    "scar-tissue": {
      "command": "npx",
      "args": ["-y", "scar-tissue-mcp"]
    }
  }
}
```

> 注：MCP Server 启动后会自动向上探测当前工作区所属的 Git 仓库根目录（`<git-root>/.scartissue`），也可通过环境变量 `SCAR_TISSUE_DIR` 在服务端显式指定存储路径。

### 2. 一键挂载 Git 自动避坑钩子 & 扫描历史死路

```bash
# 在目标仓库挂载 post-commit / post-rewrite / post-checkout 自动捕获钩子
scar-tissue install-hooks --repo /path/to/your/repo

# 冷启动：扫描仓库最近 50 次提交中的 Revert / Rollback 自动建库
scar-tissue capture-git --repo /path/to/your/repo --scan 50

# 包装执行单测命令：当同一模块连续 3 次单测失败并在第 4 次修复通过时，自动提炼死路
scar-tissue test -- pytest -q
```

### 3. 核心 MCP 工具契约

| 工具名 | 触发时机 | 输入参数 | 返回效果 |
| :--- | :--- | :--- | :--- |
| `record_dead_end` | `git revert/reset`、单测连续 3 次失败后修好、或 Agent 踩坑回滚时 | `target_file`, `attempted_change`, `error_root_cause`, `dead_end_approach`, `safe_pivot` | 自动生成 16 维隐空间指纹与严格 `<=50` 字的防粉红大象警告并持久化 |
| `check_scar_tissue` | Agent 准备修改任何文件或模块前强制调用 | `target_file`, `planned_change`, `target_symbol` | 无关文件返回空（零上下文污染）；命中雷区时精准弹出 `<=50` 字红线警报 |

---

## 四、 真实开源仓库 Commit 实测验证（零个人/公司信息，100% 公开可查验）

运行端到端基准测试（含 5 个公开开源仓库真实 Commit SHA 实测 + 5 个经典工程沙箱压测）：

```bash
PYTHONPATH=. python3 benchmarks/evaluate_real_oss_commits.py
PYTHONPATH=. python3 benchmarks/evaluate_5_scenarios.py
pytest tests/ -v
```

### A. 5 个来自官方 MCP / Pydantic-AI 仓库的真实公开 Git Revert / 死锁修复案例

| 案例编号 | 开源仓库与真实 Commit SHA | 目标模块与符号 | 弹出的 ≤50 字高优红线警告 (`alert_50c`) | 字数 | 压缩比 | 粉红大象风险 (Raw $\rightarrow$ Pivot) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OSS-1** | `modelcontextprotocol/python-sdk` ([`edf0950a`](https://github.com/modelcontextprotocol/python-sdk/commit/edf0950a6a8afa48ae2dd99f02d56d383c5904bb)) | `src/mcp/server/fastmcp/server.py` (`FastMCP`) | `⛔忌构造器直接暴露挂载参数与r…(与Starlette…)→用Starlette Mount('/…` | 50 字 | 84.2x | `1.00 -> 0.00` |
| **OSS-2** | `modelcontextprotocol/servers` ([`0b36cb4b`](https://github.com/modelcontextprotocol/servers/commit/0b36cb4b54852a06430d5d678a6095f790485425)) | `src/filesystem/index.ts` (`expandHome`) | `⛔忌启动时隐式调用硬编码扫描根…(宿主机无/proje…)→用强制校验process.argv.s…` | 50 字 | 77.8x | `1.00 -> 0.00` |
| **OSS-3** | `modelcontextprotocol/servers` ([`d73f99ef`](https://github.com/modelcontextprotocol/servers/commit/d73f99efbfd40c3aa1b61e88728b3d49fb52608f)) | `src/memory/index.ts` (`KnowledgeGraphManager`) | `⛔忌createEntitie…(单轮多工具并发调用时…)→用this.withLock()单队列…` | 50 字 | 108.4x | `1.00 -> 0.00` |
| **OSS-4** | `modelcontextprotocol/python-sdk` ([`528e3665`](https://github.com/modelcontextprotocol/python-sdk/commit/528e36655819a686d5b07e23d70748fcecebfc0a)) | `src/mcp/server/streamable_http.py` (`StreamableHTTPServerTransport`) | `⛔忌is_json_respo…(纯JSON响应无SS…)→用标记can_send_request…` | 50 字 | 123.0x | `0.45 -> 0.00` |
| **OSS-5** | `pydantic/pydantic-ai` ([`52357f8f`](https://github.com/pydantic/pydantic-ai/commit/52357f8faa9ca075d397fcda841763a4907dc033)) | `.../shell/_jobs.py` (`stop_watcher`) | `⛔忌写入停止标记文件失败时仍无…(任务目录被删或只读时…)→用写watch.stop失败分支直接k…` | 50 字 | 96.4x | `0.73 -> 0.00` |

### B. 5 个经典高强度研发“AI 反复横跳”基准场景压测

| 场景编号 | 目标模块与符号 | 捕获来源 | 16 维守门员匹配度 | 弹出的 ≤50 字高优红线警告 (`alert_50c`) | 字数 | 单次压缩比 | 粉红大象风险 (Raw -> Pivot) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | `trainer/ddp_runner.py` (`sync_shard_metrics`) | `git_revert` | `0.9470` | `⛔忌rank==0分支调用跨卡…(非0卡未进分支致NC…)→用全卡无条件all_reduce归约` | 48 字 | 13.9x | `1.00 -> 0.00` |
| **S2** | `storage/async_sqlite.py` (`write_audit_event`) | `pytest_3_strike` | `0.9665` | `⛔忌用协程全局锁包装跨线程裸共…(线程池并发写冲突)→用开启WAL日志并由单写协程队列串行落盘` | 47 字 | 58.0x | `1.00 -> 0.00` |
| **S3** | `web/components/FilterPanel.tsx` (`FilterPanel`) | `git_revert` | `0.9314` | `⛔忌副作用钩子中双向监听并调用…(路由对象引用更新致无…)→用仅在表单提交事件回调中单向写入URL` | 49 字 | 35.3x | `1.00 -> 0.00` |
| **S4** | `inference/attention_kv.py` (`apply_causal_mask`) | `pytest_3_strike` | `0.9033` | `⛔忌对32K序列调用扩维广播实…(32K序列实例化16…)→用启用SDPA内置is_causal标…` | 50 字 | 67.5x | `1.00 -> 0.00` |
| **S5** | `mcp_bridge/stdio_runner.py` (`run_tool_subprocess`) | `git_revert` | `0.7844` | `⛔忌挂载未消费管道缓冲后先调阻…(输出超64KB写满O…)→用communicate或异步流式消费…` | 50 字 | 30.1x | `1.00 -> 0.00` |

- **历史雷区拦截召回率**：**10 / 10 (100%)**
- **跨模块无关文件误报率**：**0 / 3 (0% 误报，零日常窗口污染)**
- **单次命中平均压缩比**：真实开源仓库案例（表 A）**98.0x**；合成 5 场景沙箱（表 B）单次平均 **41.0x**（模拟 8 轮会话累计对比全量历史 Token 节省估算约 **1,640.3x**）
- **粉红大象效应（字面禁用标识符泄漏风险）**：从传统反面警告的 **`1.00` 降至 `0.00`**
