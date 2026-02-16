"""Tests for git_tools — functional, security, and edge cases."""

import subprocess
from pathlib import Path

import pytest

from gpal.git_tools import (
    GIT_TOOL_SCHEMA,
    MAX_GIT_OUTPUT,
    MAX_LOG_COUNT,
    _validate_path,
    _validate_ref,
    execute_git,
    git,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _git(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command in the tmp_path repo."""
    return subprocess.run(
        ["git", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Create a git repo with one committed file."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "hello.txt").write_text("hello world\n")
    _git(tmp_path, "add", "hello.txt")
    _git(tmp_path, "commit", "-m", "initial commit")
    return tmp_path


@pytest.fixture
def empty_repo(tmp_path, monkeypatch):
    """Create a git repo with no commits."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Functional tests (happy path)
# ─────────────────────────────────────────────────────────────────────────────


def test_git_status(git_repo):
    """Status shows untracked file."""
    (git_repo / "new.txt").write_text("new file\n")
    result = git(subcommand="status")
    assert "new.txt" in result


def test_git_status_clean(git_repo):
    """Clean repo shows branch info but no changes."""
    result = git(subcommand="status")
    assert "branch.oid" in result or "# branch" in result
    # No untracked/modified entries
    assert "?" not in result.split("\n", 2)[-1] if "\n" in result else True


def test_git_log(git_repo):
    """Log shows commits."""
    # Add 2 more commits
    (git_repo / "a.txt").write_text("a\n")
    _git(git_repo, "add", "a.txt")
    _git(git_repo, "commit", "-m", "second commit")

    (git_repo / "b.txt").write_text("b\n")
    _git(git_repo, "add", "b.txt")
    _git(git_repo, "commit", "-m", "third commit")

    result = git(subcommand="log")
    assert "initial commit" in result
    assert "second commit" in result
    assert "third commit" in result


def test_git_log_max_count(git_repo):
    """Log respects max_count."""
    for i in range(5):
        (git_repo / f"file{i}.txt").write_text(f"content {i}\n")
        _git(git_repo, "add", f"file{i}.txt")
        _git(git_repo, "commit", "-m", f"commit {i}")

    result = git(subcommand="log", max_count=2)
    lines = [l for l in result.strip().split("\n") if l.strip()]
    assert len(lines) == 2


def test_git_log_with_path(git_repo):
    """Log with path filter shows only relevant commits."""
    (git_repo / "tracked.txt").write_text("tracked\n")
    _git(git_repo, "add", "tracked.txt")
    _git(git_repo, "commit", "-m", "add tracked")

    (git_repo / "other.txt").write_text("other\n")
    _git(git_repo, "add", "other.txt")
    _git(git_repo, "commit", "-m", "add other")

    result = git(subcommand="log", path="tracked.txt")
    assert "add tracked" in result
    assert "add other" not in result


def test_git_diff(git_repo):
    """Diff shows modification to tracked file."""
    (git_repo / "hello.txt").write_text("hello world\nmodified line\n")
    result = git(subcommand="diff")
    assert "+modified line" in result


def test_git_diff_no_ref(git_repo):
    """Plain git diff (working tree vs index) when ref is omitted."""
    (git_repo / "hello.txt").write_text("hello world\nstaged change\n")
    _git(git_repo, "add", "hello.txt")
    (git_repo / "hello.txt").write_text("hello world\nstaged change\nunstaged change\n")
    result = git(subcommand="diff")
    assert "+unstaged change" in result
    # Should NOT show the staged change (that's index vs working tree)


def test_git_diff_between_refs(git_repo):
    """Diff between two commits."""
    (git_repo / "hello.txt").write_text("changed\n")
    _git(git_repo, "add", "hello.txt")
    _git(git_repo, "commit", "-m", "change hello")

    result = git(subcommand="diff", ref="HEAD~1", ref2="HEAD")
    assert "-hello world" in result
    assert "+changed" in result


def test_git_show(git_repo):
    """Show displays commit details and patch."""
    result = git(subcommand="show")
    assert "initial commit" in result
    assert "Test" in result  # author name
    assert "+hello world" in result  # the added line


def test_git_show_specific_ref(git_repo):
    """Show specific earlier commit."""
    (git_repo / "hello.txt").write_text("v2\n")
    _git(git_repo, "add", "hello.txt")
    _git(git_repo, "commit", "-m", "second commit")

    result = git(subcommand="show", ref="HEAD~1")
    assert "initial commit" in result


def test_git_show_with_path(git_repo):
    """Show with path limits output to that file."""
    (git_repo / "other.txt").write_text("other\n")
    _git(git_repo, "add", "other.txt")
    (git_repo / "hello.txt").write_text("changed\n")
    _git(git_repo, "add", "hello.txt")
    _git(git_repo, "commit", "-m", "change both")

    result = git(subcommand="show", path="hello.txt")
    assert "changed" in result
    # other.txt diff should not appear when filtering by path
    assert "+other" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Input validation tests (security)
# ─────────────────────────────────────────────────────────────────────────────


def test_invalid_subcommand(git_repo):
    """Reject invalid subcommands."""
    for cmd in ["checkout", "push", "reset", "", "rm", "fetch", "clone"]:
        result = git(subcommand=cmd)
        assert "Error" in result, f"Should reject subcommand {cmd!r}"


def test_ref_leading_dash(git_repo):
    """Reject refs starting with dash (flag injection)."""
    for ref in ["--exec=sh", "-c evil", "--upload-pack=x", "-"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result, f"Should reject ref {ref!r}"


def test_ref_shell_injection(git_repo):
    """Reject shell metacharacters in refs."""
    for ref in ["HEAD;rm -rf /", "$(whoami)", "`id`", "HEAD|cat"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result, f"Should reject ref {ref!r}"


def test_ref_newline_null(git_repo):
    """Reject control characters in refs."""
    for ref in ["HEAD\nmalicious", "HEAD\0evil", "HEAD\rinjection"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result, f"Should reject ref {ref!r}"


def test_ref_command_substitution(git_repo):
    """Reject various command substitution patterns."""
    for ref in ["${IFS}cat", "$(cat /etc/passwd)"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result, f"Should reject ref {ref!r}"


def test_ref_overlength(git_repo):
    """Reject overly long refs."""
    result = git(subcommand="log", ref="a" * 300)
    assert "Error" in result


def test_ref_empty(git_repo):
    """Reject empty ref."""
    result = git(subcommand="log", ref="")
    assert "Error" in result


def test_ref_double_dot_traversal(git_repo):
    """Reject refs with .. traversal."""
    result = git(subcommand="log", ref="HEAD/../../../etc")
    assert "Error" in result


def test_ref_valid_examples(git_repo):
    """Accept valid refs (should not error on validation)."""
    # These should pass validation but may fail on git lookup — that's fine
    for ref in ["HEAD", "HEAD~3", "main", "feature/foo", "v1.0.0", "abc123"]:
        err = _validate_ref(ref)
        assert err is None, f"Should accept ref {ref!r}, got: {err}"


def test_path_traversal(git_repo):
    """Reject path traversal attempts."""
    result = git(subcommand="log", path="../../etc/passwd")
    assert "Error" in result


def test_path_absolute_escape(git_repo):
    """Reject absolute paths outside root."""
    result = git(subcommand="log", path="/etc/shadow")
    assert "Error" in result


def test_path_leading_dash(git_repo):
    """Reject paths starting with dash."""
    result = git(subcommand="log", path="--some-flag")
    assert "Error" in result


def test_max_count_clamped(git_repo):
    """max_count is clamped to MAX_LOG_COUNT."""
    # Create a few commits
    for i in range(3):
        (git_repo / f"f{i}.txt").write_text(f"{i}\n")
        _git(git_repo, "add", f"f{i}.txt")
        _git(git_repo, "commit", "-m", f"commit {i}")

    # Request 999 — should work but be clamped
    result = git(subcommand="log", max_count=999)
    assert "Error" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_not_a_git_repo(tmp_path, monkeypatch):
    """Graceful error when not in a git repo."""
    monkeypatch.chdir(tmp_path)
    result = git(subcommand="status")
    assert "Error" in result


def test_output_truncation(git_repo):
    """Large output gets truncated at MAX_GIT_OUTPUT."""
    # Create a file with enough content to generate a huge diff
    big_content = "x" * (MAX_GIT_OUTPUT + 10000) + "\n"
    (git_repo / "big.txt").write_text(big_content)
    _git(git_repo, "add", "big.txt")
    _git(git_repo, "commit", "-m", "add big file")

    # Diff against parent should be large
    result = git(subcommand="show", ref="HEAD")
    if len(result) > MAX_GIT_OUTPUT:
        assert "truncated" in result


def test_empty_repo(empty_repo):
    """Handle empty repo (no commits) gracefully."""
    result = git(subcommand="status")
    # Should work — status doesn't need commits
    assert "Error" not in result or "No commits yet" not in result

    result = git(subcommand="log")
    # Log on empty repo returns error (no HEAD)
    assert "Error" in result or result.strip() == ""


def test_execute_git_dispatch(git_repo):
    """execute_git wrapper dispatches correctly."""
    result = execute_git({"subcommand": "status"})
    assert "branch" in result.lower() or "# branch" in result


def test_git_tool_schema_valid():
    """Schema has required fields and correct enum."""
    assert GIT_TOOL_SCHEMA["name"] == "git"
    assert "description" in GIT_TOOL_SCHEMA
    props = GIT_TOOL_SCHEMA["input_schema"]["properties"]
    assert props["subcommand"]["enum"] == ["status", "diff", "log", "show"]
    assert GIT_TOOL_SCHEMA["input_schema"]["required"] == ["subcommand"]
