"""
In-process integration tests for gpal MCP server.

Uses FastMCP's Client class for zero-network testing of tool registration,
metadata, path validation, and annotation correctness. Live API tests are
gated behind GEMINI_API_KEY.
"""

import inspect
import os
import pytest
from pathlib import Path

from fastmcp import Client
from gpal.server import mcp, _validate_output_path


# ─────────────────────────────────────────────────────────────────────────────
# A. Tool Registration & Metadata
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TOOLS = {
    "consult_gemini_flash",
    "consult_gemini_pro",
    "gemini_search",
    "gemini_code_exec",
    "upload_file",
    "create_context_cache",
    "delete_context_cache",
    "semantic_search",
    "rebuild_index",
    "list_models",
    "generate_image",
    "generate_speech",
}


@pytest.mark.asyncio
async def test_all_tools_registered():
    """Every expected tool is exposed via the MCP server."""
    async with Client(mcp) as c:
        tools = await c.list_tools()
        names = {t.name for t in tools}
        missing = EXPECTED_TOOLS - names
        assert not missing, f"Missing tools: {missing}"


@pytest.mark.asyncio
async def test_tool_descriptions_non_empty():
    """Every tool has a non-empty description."""
    async with Client(mcp) as c:
        tools = await c.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"


@pytest.mark.asyncio
async def test_tool_input_schemas():
    """Key tools have expected parameters in their input schemas."""
    async with Client(mcp) as c:
        tools = await c.list_tools()
        by_name = {t.name: t for t in tools}

        # consult tools should have 'query' required
        for name in ("consult_gemini_flash", "consult_gemini_pro"):
            schema = by_name[name].inputSchema
            assert "query" in schema.get("properties", {}), f"{name} missing 'query' param"
            assert "query" in schema.get("required", []), f"{name}: 'query' not required"

        # generate_image should have prompt + output_path
        schema = by_name["generate_image"].inputSchema
        assert "prompt" in schema.get("properties", {})
        assert "output_path" in schema.get("properties", {})


# ─────────────────────────────────────────────────────────────────────────────
# B. Tool Timeouts
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TIMEOUTS = {
    "consult_gemini_flash": 60,
    "consult_gemini_pro": 600,
    "rebuild_index": 300,
}


@pytest.mark.asyncio
async def test_tool_timeouts():
    """Verify timeout values on tools that have them."""
    all_tools = await mcp._local_provider.list_tools()
    by_name = {t.name: t for t in all_tools}

    for tool_name, expected in EXPECTED_TIMEOUTS.items():
        tool = by_name[tool_name]
        assert tool.timeout == expected, (
            f"{tool_name}: expected timeout={expected}, got {tool.timeout}"
        )


@pytest.mark.asyncio
async def test_tools_without_timeout():
    """Tools not in the timeout table should have no timeout set."""
    all_tools = await mcp._local_provider.list_tools()
    for tool in all_tools:
        if tool.name not in EXPECTED_TIMEOUTS:
            assert tool.timeout is None, (
                f"{tool.name}: unexpected timeout={tool.timeout}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# C. Path Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_output_path_within_cwd(tmp_path, monkeypatch):
    """Paths under cwd are accepted."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path(str(tmp_path / "output" / "file.png"))
    assert result is None
    assert (tmp_path / "output").is_dir()


def test_validate_output_path_rejects_traversal(tmp_path, monkeypatch):
    """Path traversal outside cwd is rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path("/tmp/evil.png")
    assert result is not None
    assert "outside" in result.lower() or "denied" in result.lower()


def test_validate_output_path_rejects_etc(tmp_path, monkeypatch):
    """Absolute paths to system dirs are rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path("/etc/passwd")
    assert result is not None


def test_validate_output_path_relative_traversal(tmp_path, monkeypatch):
    """../../../etc/passwd style traversal is caught."""
    monkeypatch.chdir(tmp_path)
    result = _validate_output_path("../../../etc/passwd")
    assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# D. Annotation Regression (guards against `from __future__ import annotations`)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_future_annotations_import():
    """server.py must not use `from __future__ import annotations`.

    PEP 563 deferred evaluation breaks Gemini's automatic function calling
    because type annotations become strings instead of real types.
    """
    import gpal.server as mod
    source = inspect.getsource(mod)
    assert "from __future__ import annotations" not in source


def test_tool_functions_have_real_annotations():
    """Gemini AFC tool functions must have concrete type annotations, not strings."""
    from gpal.server import list_directory, read_file, search_project

    for fn in (list_directory, read_file, search_project):
        hints = fn.__annotations__
        for param, annotation in hints.items():
            assert not isinstance(annotation, str), (
                f"{fn.__name__}.{param} has string annotation '{annotation}' — "
                "probably caused by `from __future__ import annotations`"
            )


# ─────────────────────────────────────────────────────────────────────────────
# E. ToolResult Structure (requires API key)
# ─────────────────────────────────────────────────────────────────────────────

HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_API_KEY, reason="no GEMINI_API_KEY")
async def test_flash_returns_tool_result():
    """consult_gemini_flash returns structured ToolResult with meta."""
    async with Client(mcp) as c:
        result = await c.call_tool(
            "consult_gemini_flash",
            {"query": "What is 2+2? Reply with just the number."},
        )
        assert not result.is_error
        # structured_content should have result + model
        assert result.structured_content is not None
        assert "result" in result.structured_content
        assert "model" in result.structured_content
        # meta should have model, session_id, duration_ms
        assert result.meta is not None
        assert "model" in result.meta
        assert "session_id" in result.meta
        assert "duration_ms" in result.meta
        assert result.meta["duration_ms"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# F. Gemini Autonomous Tools (requires API key)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_API_KEY, reason="no GEMINI_API_KEY")
async def test_flash_can_read_file():
    """Flash can use read_file to read a known file autonomously."""
    async with Client(mcp) as c:
        result = await c.call_tool(
            "consult_gemini_flash",
            {"query": "Use read_file to read pyproject.toml and tell me the project name."},
        )
        assert not result.is_error
        # The response should mention "gpal" somewhere
        text = str(result.content)
        assert "gpal" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# G. Resources
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_info_resource():
    """gpal://info returns valid JSON with models and limits."""
    async with Client(mcp) as c:
        resources = await c.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "gpal://info" in uris

        import json
        result = await c.read_resource("gpal://info")
        # read_resource returns a list of ResourceContents
        text = result[0].text if isinstance(result, list) else str(result)
        data = json.loads(text)
        assert "models" in data
        assert "limits" in data
        assert "flash" in data["models"]
