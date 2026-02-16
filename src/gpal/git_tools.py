"""
Read-only git tool for codebase exploration.

Provides status, diff, log, and show subcommands with defense-in-depth
input validation. Zero third-party dependencies (stdlib only).

Exports:
- git() — plain function for gpal's Gemini automatic_function_calling
- GIT_TOOL_SCHEMA — JSON schema dict for cpal's CLAUDE_TOOLS list
- execute_git() — dispatch handler for cpal's execute_tool
"""

import re
import subprocess
from pathlib import Path

MAX_GIT_OUTPUT = 100_000  # 100KB output cap
MAX_LOG_COUNT = 100
GIT_TIMEOUT = 30  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

_REF_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/~^@{}\-]*$")
_REF_BLOCKLIST = ("--exec", "--upload-pack", "--receive-pack", "..", "$(", "`", "${", "|", ";", "\n", "\r", "\0")


def _validate_ref(ref: str) -> str | None:
    """Validate a git ref. Returns error message or None if valid."""
    if not ref:
        return "Error: ref must not be empty."
    if len(ref) > 256:
        return f"Error: ref too long ({len(ref)} chars, max 256)."
    if ref.startswith("-"):
        return f"Error: ref must not start with '-': {ref!r}"
    for blocked in _REF_BLOCKLIST:
        if blocked in ref:
            return f"Error: ref contains blocked sequence {blocked!r}: {ref!r}"
    if not _REF_PATTERN.match(ref):
        return f"Error: ref contains invalid characters: {ref!r}"
    return None


def _validate_path(path: str, root: Path) -> str | None:
    """Validate a file path is within the git root. Returns error message or None."""
    if not path:
        return "Error: path must not be empty."
    if path.startswith("-"):
        return f"Error: path must not start with '-': {path!r}"
    try:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            return f"Error: path escapes project root: {path!r}"
    except Exception as e:
        return f"Error: invalid path {path!r}: {e}"
    return None


def _get_git_root(cwd: Path | None = None) -> Path | str:
    """Get the git repository root. Returns Path or error string."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
            cwd=cwd,
        )
        if result.returncode != 0:
            return f"Error: not a git repository: {result.stderr.strip()}"
        return Path(result.stdout.strip())
    except FileNotFoundError:
        return "Error: git is not installed."
    except subprocess.TimeoutExpired:
        return "Error: git command timed out."


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout or an error string."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=GIT_TIMEOUT,
            cwd=cwd,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Some commands (like log on empty repo) return non-zero with useful stderr
            return f"Error: git returned exit code {result.returncode}: {stderr}"
        output = result.stdout
        if len(output) > MAX_GIT_OUTPUT:
            output = output[:MAX_GIT_OUTPUT] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: git command timed out."
    except Exception as e:
        return f"Error running git: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Main tool function
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SUBCOMMANDS = frozenset({"status", "diff", "log", "show"})

_LOG_FORMAT = "%h %aI %an %s"  # short hash, ISO date, author, subject
_SHOW_FORMAT = "commit %H%nAuthor: %an <%ae>%nDate:   %aI%n%n%w(0,4,4)%B"


def git(
    subcommand: str,
    ref: str | None = None,
    ref2: str | None = None,
    path: str | None = None,
    max_count: int = 20,
) -> str:
    """Read-only git operations for codebase exploration.

    Args:
        subcommand: One of "status", "diff", "log", "show".
        ref: Git ref (branch, tag, commit SHA). Default HEAD for log/show, omitted for diff.
        ref2: Second ref for diff ranges (e.g. diff ref ref2).
        path: Limit to a specific file path.
        max_count: Max commits for log (capped at 100). Default 20.
    """
    # Layer 1: Subcommand whitelist
    if subcommand not in _VALID_SUBCOMMANDS:
        return f"Error: invalid subcommand {subcommand!r}. Must be one of: {', '.join(sorted(_VALID_SUBCOMMANDS))}"

    # Get git root
    root = _get_git_root()
    if isinstance(root, str):
        return root  # error message

    # Default ref for subcommands that need one
    if ref is None and subcommand in ("log", "show"):
        ref = "HEAD"

    # Layer 2: Validate ref
    if ref is not None:
        err = _validate_ref(ref)
        if err:
            return err

    if ref2 is not None:
        err = _validate_ref(ref2)
        if err:
            return err

    # Layer 3: Validate path
    if path is not None:
        err = _validate_path(path, root)
        if err:
            return err

    # Dispatch
    if subcommand == "status":
        return _run_git(["git", "status", "--porcelain=v2", "--branch"], root)

    elif subcommand == "diff":
        args = ["git", "diff"]
        if ref is not None:
            args.append(ref)
        if ref2 is not None:
            args.append(ref2)
        if path is not None:
            args.extend(["--", path])
        return _run_git(args, root)

    elif subcommand == "log":
        count = min(max(1, max_count), MAX_LOG_COUNT)
        args = ["git", "log", f"--max-count={count}", f"--format={_LOG_FORMAT}", ref]
        if path is not None:
            args.extend(["--", path])
        return _run_git(args, root)

    elif subcommand == "show":
        args = ["git", "show", f"--format={_SHOW_FORMAT}", ref]
        if path is not None:
            args.extend(["--", path])
        return _run_git(args, root)

    # Unreachable due to whitelist check, but just in case
    return f"Error: unhandled subcommand {subcommand!r}"


# ─────────────────────────────────────────────────────────────────────────────
# cpal integration
# ─────────────────────────────────────────────────────────────────────────────

GIT_TOOL_SCHEMA = {
    "name": "git",
    "description": (
        "Read-only git operations for codebase exploration. "
        "Subcommands: status, diff, log, show."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subcommand": {
                "type": "string",
                "enum": ["status", "diff", "log", "show"],
                "description": "Git operation to perform.",
            },
            "ref": {
                "type": "string",
                "description": "Git ref (branch, tag, commit SHA). Default HEAD for log/show, omitted for diff (working tree vs index).",
            },
            "ref2": {
                "type": "string",
                "description": "Second ref for diff ranges.",
            },
            "path": {
                "type": "string",
                "description": "Limit to a specific file path.",
            },
            "max_count": {
                "type": "integer",
                "default": 20,
                "description": "Max commits for log (capped at 100).",
            },
        },
        "required": ["subcommand"],
    },
}


def execute_git(input_data: dict) -> str:
    """Dispatch handler for cpal's execute_tool."""
    return git(
        subcommand=input_data.get("subcommand", ""),
        ref=input_data.get("ref"),
        ref2=input_data.get("ref2"),
        path=input_data.get("path"),
        max_count=input_data.get("max_count", 20),
    )
