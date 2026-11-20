#!/usr/bin/env node
/**
 * Cross-platform wrapper for ScarTissue MCP Server & CLI (`npx scar-tissue-mcp`).
 * - Routes `--help` / `-h` / `--version` / `-V` and CLI subcommands to `scar_tissue.cli`
 * - Routes no-arg stdio execution to `scar_tissue.server`
 * - Automatically falls back to `uv run --no-project` (cache-only, read-only safe)
 *   if ambient Python lacks Python>=3.10, `mcp<2` (`FastMCP`), or `pydantic`.
 */
const { spawn, spawnSync } = require("child_process");
const path = require("path");

const PY_DEP_PROBE =
  "import sys; assert sys.version_info >= (3, 10); import pydantic; from mcp.server.fastmcp import FastMCP";

function resolveRunner(env) {
  const candidates = process.env.PYTHON_BIN
    ? [process.env.PYTHON_BIN]
    : ["python3", "python"];

  for (const bin of candidates) {
    const check = spawnSync(bin, ["-c", PY_DEP_PROBE], {
      stdio: "ignore",
      env,
    });
    if (check.status === 0) {
      return { cmd: bin, prefixArgs: [] };
    }
  }

  // Fallback to `uv run --no-project` if `uv` is available on PATH.
  // Never mutates the npm package directory (safe on read-only global installs & Windows).
  const uvCheck = spawnSync("uv", ["--version"], { stdio: "ignore", env });
  if (uvCheck.status === 0) {
    return {
      cmd: "uv",
      prefixArgs: [
        "run",
        "--no-project",
        "--python",
        ">=3.10",
        "--with",
        "mcp>=1.0.0,<2.0.0",
        "--with",
        "pydantic>=2.0.0",
        "python",
      ],
    };
  }

  // Fallback to first available python binary so user sees a clear diagnostic
  for (const bin of candidates) {
    const basic = spawnSync(bin, ["--version"], { stdio: "ignore", env });
    if (basic.status === 0) {
      return { cmd: bin, prefixArgs: [] };
    }
  }
  return { cmd: process.platform === "win32" ? "python" : "python3", prefixArgs: [] };
}

const args = process.argv.slice(2);
const repoRoot = path.resolve(__dirname, "..");
const env = Object.assign({}, process.env, {
  PYTHONPATH: [repoRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
});

const isHelpOrVersion =
  args.includes("--help") ||
  args.includes("-h") ||
  args.includes("--version") ||
  args.includes("-V");
const isCliSubcommand =
  args.length > 0 && (!args[0].startsWith("-") || isHelpOrVersion);
const moduleName = isCliSubcommand ? "scar_tissue.cli" : "scar_tissue.server";

const runner = resolveRunner(env);
const child = spawn(
  runner.cmd,
  [...runner.prefixArgs, "-m", moduleName, ...args],
  {
    stdio: "inherit",
    env,
  }
);

child.on("error", (err) => {
  console.error(
    `[scar-tissue] Failed to launch '${runner.cmd}': ${err.message}.\n` +
      "Install Python >= 3.10 with `pip install scar-tissue` or install `uv` (https://docs.astral.sh/uv/)."
  );
  process.exit(err && err.code === "ENOENT" ? 127 : 1);
});

child.on("exit", (code, signal) => {
  if (code !== null) {
    process.exit(code);
  }
  if (signal) {
    try {
      process.kill(process.pid, signal);
    } catch (_) {
      process.exit(1);
    }
    return;
  }
  process.exit(1);
});
