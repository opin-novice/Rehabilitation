#!/usr/bin/env python3
"""
Pre-run verification for ML experiments: syntax check + CUDA smoke test +
deterministic run metadata logged to RESEARCH_LOG.md.

Wired as a PreToolUse hook (see .claude/settings.json), or run manually via
`/dry-run` or `python scripts/dry_run_check.py`. Assumes it's run from the
project root (same cwd Claude Code hooks run in by default).

The hook is NOT scoped to `python ` commands, despite what the settings file
reads like: a PreToolUse `matcher` matches the tool *name*, so "Bash" matches
every Bash call, and the `"if": "Bash(python *)"` key beside it is not part of
the hook schema and is ignored. Scoping is therefore this script's own job --
see the early exit in main() for commands that run no Python file.

As a hook it gets the tool call as JSON on stdin and lints the script that
command actually invokes. Entrypoints in this repo live under src/, so the
old "look for train.py in the project root" scan never matched anything and
the syntax half was a silent no-op. That scan survives only as the fallback
for the manual and `/dry-run` paths, where there is no stdin payload.

Exit codes matter here: Claude Code only treats exit code 2 as a blocking
failure for PreToolUse hooks. Exit code 1 -- the normal Unix failure code --
is a NON-blocking error, and the tool call proceeds anyway. So this script
exits 0 on success and 2 on failure, never 1.
"""

import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Installed at the user level (~/.claude/scripts/) this fires before every
# python command in every repo, so it has to stay quiet where it doesn't
# belong. RESEARCH_LOG.md doubles as the explicit opt-in: `touch` it in a
# repo to turn the check on there regardless of the inferred markers.
ML_MARKER_FILES = ("RESEARCH_LOG.md",)
ML_MARKER_GLOBS = ("train*.py", "src/train*.py", "src/models*.py")
ML_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+"
    r"(torch|torchcde|torchdiffeq|e3nn|jax|flax|tensorflow|keras|lightning)\b",
    re.MULTILINE,
)

# Two tiers, because a false block is worse than a missed warning -- it
# teaches you to switch the gate off.
#
# Blocking: syntax errors (E9), always-true asserts and literal `is`
# comparisons (F63), misplaced statements like `break` outside a loop (F7).
# These have no false positives. Deliberately narrower than ruff's default
# set: F401 and friends would block a training run over an unused import.
#
# Advisory: undefined names (F82) and redefinitions (F811). F821 cannot tell
# a genuine NameError from a loop-carried binding -- `cur_feature` in
# src/models_curvenet.py is assigned at the end of each iteration and read
# under `if step != 0` on the next one, which is correct code that ruff
# flags. So these are reported, never enforced.
RUFF_BLOCKING = "E9,F63,F7"
RUFF_ADVISORY = "F82,F811"

PY_EXE = {"python", "python3", "py", "python.exe", "python3.exe", "py.exe"}
SHELL_SEPARATORS = {"&&", "||", ";", "|", "&"}

# A CUDA context alone costs a few hundred MB. Below this, the smoke test
# would OOM for reasons that have nothing to do with the code being checked.
MIN_FREE_GPU_BYTES = 1024**3


def run(cmd: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return ""


def command_from_stdin() -> str:
    """The Bash command being gated, or "" when run manually."""
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return ""
    command = tool_input.get("command")
    return command if isinstance(command, str) else ""


def scripts_from_command(command: str) -> list[str]:
    """The .py files `command` would actually execute.

    Handles chains (`cd d:/repo && python src/train.py`), skips the `-m` and
    `-c` forms that have no file to lint, and drops paths that don't exist.
    Split with posix=False so Windows backslash paths survive it.
    """
    if not command:
        return []
    try:
        tokens = [t.strip("\"'") for t in shlex.split(command, posix=False)]
    except ValueError:
        return []

    scripts: list[str] = []
    i = 0
    while i < len(tokens):
        if Path(tokens[i]).name.lower() not in PY_EXE:
            i += 1
            continue
        i += 1
        # Walk this invocation's arguments until the next shell separator,
        # stepping over interpreter flags like -u or -W ignore.
        while i < len(tokens) and tokens[i] not in SHELL_SEPARATORS:
            token = tokens[i]
            if token in ("-m", "-c"):
                break
            if token.endswith(".py"):
                if Path(token).is_file():
                    scripts.append(token)
                break
            i += 1
    return scripts


def ruff(select: str, script_path: str) -> tuple[bool, str]:
    """Run ruff for one rule set. Raises FileNotFoundError if ruff is absent."""
    result = subprocess.run(
        ["ruff", "check", "--isolated", "--output-format", "concise",
         "--select", select, script_path],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def is_ml_project(scripts: list[str]) -> bool:
    """Cheapest markers first -- this runs ahead of every python command."""
    if any(Path(f).exists() for f in ML_MARKER_FILES):
        return True
    if any(next(Path(".").glob(g), None) is not None for g in ML_MARKER_GLOBS):
        return True
    for script in scripts:
        try:
            source = Path(script).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if ML_IMPORT.search(source):
            return True
    return False


def project_copy_takes_precedence() -> bool:
    """True when a repo pins its own copy and we are the user-level install.

    Both installs are live in a repo that has one, and both would fire on the
    same command. The repo's copy is the authoritative one -- it is the
    version that got committed alongside the code it gates -- so the global
    install steps aside and the check runs once.
    """
    local = Path("scripts/dry_run_check.py")
    if not local.is_file():
        return False
    try:
        return local.resolve() != Path(__file__).resolve()
    except OSError:
        return False


def check_syntax(script_path: str) -> tuple[bool, str]:
    """Prefer ruff; fall back to py_compile if it isn't installed."""
    try:
        return ruff(RUFF_BLOCKING, script_path)
    except FileNotFoundError:
        import py_compile
        try:
            py_compile.compile(script_path, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)


def advisory_lint(script_path: str) -> str:
    """Findings worth seeing but not worth blocking on. "" if clean."""
    try:
        ok, msg = ruff(RUFF_ADVISORY, script_path)
    except FileNotFoundError:
        return ""
    return "" if ok else msg


def cuda_smoke_test() -> tuple[bool, str]:
    """One forward+backward pass on a batch of 4.

    Anything that looks like GPU contention -- too little free memory, or an
    OOM on a 4x64 Linear -- is reported as a skip, not a failure. A smoke test
    this small can only run out of memory because something else owns the card
    (an overnight run, say), and blocking an unrelated command over that is a
    false positive, not a caught bug.
    """
    try:
        import torch
    except ImportError:
        return True, "torch not installed - skipped"

    if not torch.cuda.is_available():
        return True, "CUDA not available - skipped GPU checks"

    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        free = None
    if free is not None and free < MIN_FREE_GPU_BYTES:
        return True, (
            f"only {free / 1024**2:.0f} MiB free - skipped, "
            "another job likely owns the GPU"
        )

    try:
        model = torch.nn.Linear(64, 10).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 64).cuda()
        y = torch.randint(0, 10, (4,)).cuda()

        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()

        if torch.isnan(loss) or torch.isinf(loss):
            return False, f"loss is NaN/Inf: {loss.item()}"

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        if not has_grad:
            return False, "no non-zero gradients after backward()"

        torch.cuda.empty_cache()
        return True, ""
    except torch.cuda.OutOfMemoryError:
        return True, "OOM on a 4x64 Linear - skipped, GPU is busy"
    except Exception as e:
        return False, f"smoke test raised: {e}"


def log_metadata(status: str, checked: str) -> str:
    """Deterministic bookkeeping - no LLM subagent needed for this part."""
    git_hash = run(["git", "rev-parse", "--short", "HEAD"]) or "no commit yet"
    git_branch = run(["git", "branch", "--show-current"]) or "unknown"
    git_diff_stat = run(["git", "diff", "--stat"]) or "none"
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) or "none detected"

    snapshot = run(["pip", "freeze"])
    if snapshot:
        Path("requirements_snapshot.txt").write_text(snapshot)

    entry = (
        f"\n## Pre-run check ({status}) \u2014 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Commit: {git_hash} (branch: {git_branch})\n"
        f"- Syntax-checked: {checked}\n"
        f"- Uncommitted changes: {git_diff_stat}\n"
        f"- GPU: {gpu}\n"
    )
    try:
        with open("RESEARCH_LOG.md", "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        pass
    return entry


def main() -> None:
    command = command_from_stdin()  # empty on the manual / `/dry-run` path
    if command and project_copy_takes_precedence():
        sys.exit(0)

    scripts = scripts_from_command(command)
    if command and not is_ml_project(scripts):
        # Someone else's repo, fired by the global hook. Say nothing.
        sys.exit(0)

    if command and not scripts:
        # Fired on a command that runs no Python file -- git, ls, pip, cat.
        # The hook sees every Bash call (see module docstring), and
        # is_ml_project() is unconditionally true in any repo that has a
        # RESEARCH_LOG.md, so without this guard every shell command pays for
        # a CUDA smoke test and a `pip freeze`, and banks a log entry reading
        # "Syntax-checked: nothing". There is nothing here to gate.
        sys.exit(0)

    if not scripts:
        # Manual / `/dry-run` path: no stdin payload to read a command from.
        fallback = next(
            (c for c in ("train.py", "main.py", "run.py") if Path(c).exists()), None
        )
        scripts = [fallback] if fallback else []

    failures, advisories = [], []
    for script in scripts:
        ok, msg = check_syntax(script)
        if not ok:
            # A hard error re-surfaces in the advisory pass too; fix it first.
            failures.append((script, msg))
            continue
        note = advisory_lint(script)
        if note:
            advisories.append(note)

    cuda_ok, cuda_msg = cuda_smoke_test()
    passed = not failures and cuda_ok

    checked = ", ".join(scripts) if scripts else "nothing (no script in command)"
    log_entry = log_metadata("PASS" if passed else "FAIL", checked)

    if passed:
        print(f"dry-run check passed - syntax: {checked}")
        if cuda_msg:
            print(f"  cuda: {cuda_msg}")
        for note in advisories:
            print(f"  advisory (not blocking):\n{note}")
        print(log_entry)
        sys.exit(0)

    print("dry-run check FAILED", file=sys.stderr)
    for script, msg in failures:
        print(f"  syntax/lint ({script}): {msg}", file=sys.stderr)
    if not cuda_ok:
        print(f"  cuda smoke test: {cuda_msg}", file=sys.stderr)
    for note in advisories:
        print(f"  advisory (not blocking):\n{note}", file=sys.stderr)
    sys.exit(2)  # 2, not 1 -- see module docstring


if __name__ == "__main__":
    main()
