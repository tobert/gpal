"""
In-process integration tests for gpal MCP server.

Uses FastMCP's Client class for zero-network testing of tool registration,
metadata, path validation, and annotation correctness. Live API tests are
gated behind GEMINI_API_KEY.
"""

import inspect
import os
import time
import pytest
from pathlib import Path

from fastmcp import Client
from unittest.mock import patch
from gpal.server import (
    mcp, _validate_output_path,
    record_tokens, tokens_in_window, token_stats, GeminiResponse,
    _sync_throttle, _async_throttle, _KNOWN_MODELS, MODEL_SEARCH,
    RATE_LIMITS_TPM, _afc_local, _send_with_retry, _EXECUTOR,
)


# ─────────────────────────────────────────────────────────────────────────────
# A. Tool Registration & Metadata
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TOOLS = {
    "consult_gemini",
    "consult_gemini_oneshot",
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

        # consult_gemini should have 'query' required and 'model' optional
        schema = by_name["consult_gemini"].inputSchema
        assert "query" in schema.get("properties", {}), "consult_gemini missing 'query' param"
        assert "query" in schema.get("required", []), "consult_gemini: 'query' not required"
        assert "model" in schema.get("properties", {}), "consult_gemini missing 'model' param"

        # consult_gemini_oneshot should have 'query' required
        schema = by_name["consult_gemini_oneshot"].inputSchema
        assert "query" in schema.get("properties", {})
        assert "query" in schema.get("required", [])

        # generate_image should have prompt + output_path
        schema = by_name["generate_image"].inputSchema
        assert "prompt" in schema.get("properties", {})
        assert "output_path" in schema.get("properties", {})


# ─────────────────────────────────────────────────────────────────────────────
# B. Tool Timeouts
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TIMEOUTS = {
    "consult_gemini": 660,
    "consult_gemini_oneshot": 120,
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
# E. Token Tracking
# ─────────────────────────────────────────────────────────────────────────────


def test_record_and_query_tokens():
    """Token tracking records and queries correctly."""
    from gpal.server import MODEL_FLASH
    record_tokens(MODEL_FLASH, 100)
    record_tokens(MODEL_FLASH, 200)
    assert tokens_in_window(MODEL_FLASH) >= 300


def test_tokens_in_window_expires():
    """Tokens older than the window are not counted."""
    from gpal.server import MODEL_PRO
    import gpal.server as srv
    with srv._token_lock:
        srv._token_windows[MODEL_PRO] = [(time.monotonic() - 120, 9999)]
    # Should be zero — 120s ago is outside the 60s window
    assert tokens_in_window(MODEL_PRO) == 0


def test_token_stats_returns_active():
    """token_stats() returns models with recent usage."""
    from gpal.server import MODEL_FLASH
    record_tokens(MODEL_FLASH, 500)
    stats = token_stats()
    assert MODEL_FLASH in stats
    assert stats[MODEL_FLASH]["tokens_last_60s"] >= 500


def test_record_tokens_rejects_unknown_model():
    """Unknown model strings are silently ignored (DoS prevention)."""
    record_tokens("totally-fake-model", 9999)
    assert tokens_in_window("totally-fake-model") == 0


def test_gemini_response_dataclass():
    """GeminiResponse holds text and token counts."""
    r = GeminiResponse("hello", 10, 20, 30)
    assert r.text == "hello"
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 20
    assert r.total_tokens == 30


# ─────────────────────────────────────────────────────────────────────────────
# F. ToolResult Structure (requires API key)
# ─────────────────────────────────────────────────────────────────────────────

HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_API_KEY, reason="no GEMINI_API_KEY")
async def test_consult_returns_tool_result():
    """consult_gemini returns structured ToolResult with meta including tokens."""
    async with Client(mcp) as c:
        result = await c.call_tool(
            "consult_gemini",
            {"query": "What is 2+2? Reply with just the number.", "model": "flash"},
        )
        assert not result.is_error
        assert result.structured_content is not None
        assert "result" in result.structured_content
        assert "model" in result.structured_content
        assert result.meta is not None
        assert "model" in result.meta
        assert "session_id" in result.meta
        assert "duration_ms" in result.meta
        assert result.meta["duration_ms"] > 0
        # Token counts should be present
        assert "total_tokens" in result.meta


# ─────────────────────────────────────────────────────────────────────────────
# G. Gemini Autonomous Tools (requires API key)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_API_KEY, reason="no GEMINI_API_KEY")
async def test_flash_can_read_file():
    """Flash can use read_file to read a known file autonomously."""
    async with Client(mcp) as c:
        result = await c.call_tool(
            "consult_gemini",
            {
                "query": "Use read_file to read pyproject.toml and tell me the project name.",
                "model": "flash",
            },
        )
        assert not result.is_error
        text = str(result.content)
        assert "gpal" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# H. Resources
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_info_resource():
    """gpal://info returns valid JSON with models, limits, and token_usage."""
    async with Client(mcp) as c:
        resources = await c.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "gpal://info" in uris

        import json
        result = await c.read_resource("gpal://info")
        text = result[0].text if isinstance(result, list) else str(result)
        data = json.loads(text)
        assert "models" in data
        assert "limits" in data
        assert "flash" in data["models"]
        assert "token_usage" in data


# ─────────────────────────────────────────────────────────────────────────────
# I. Throttle Helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_model_search_in_known_models():
    """MODEL_SEARCH is tracked (present in _KNOWN_MODELS and RATE_LIMITS_TPM)."""
    assert MODEL_SEARCH in _KNOWN_MODELS
    assert MODEL_SEARCH in RATE_LIMITS_TPM


def test_sync_throttle_no_block_under_limit():
    """_sync_throttle returns immediately when usage is under 90%."""
    from gpal.server import MODEL_FLASH
    # tokens_in_window returns current usage; under limit should not block
    with patch("gpal.server.tokens_in_window", return_value=0):
        _sync_throttle(MODEL_FLASH)  # Should return immediately


def test_sync_throttle_blocks_then_passes():
    """_sync_throttle loops when over limit, then exits when under."""
    from gpal.server import MODEL_FLASH
    # First call: over limit, second call: under limit
    with patch("gpal.server.tokens_in_window", side_effect=[4_000_000, 0]):
        with patch("gpal.server.time.sleep") as mock_sleep:
            _sync_throttle(MODEL_FLASH)
            assert mock_sleep.call_count == 1
            # Jitter should be between 1.0 and 5.0
            slept = mock_sleep.call_args[0][0]
            assert 1.0 <= slept <= 5.0


def test_sync_throttle_no_limit_model():
    """_sync_throttle is a no-op for models not in RATE_LIMITS_TPM."""
    _sync_throttle("some-unknown-model")  # Should return immediately


@pytest.mark.asyncio
async def test_async_throttle_no_block_under_limit():
    """_async_throttle returns immediately when under limit."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server.tokens_in_window", return_value=0):
        await _async_throttle(MODEL_FLASH)


@pytest.mark.asyncio
async def test_async_throttle_blocks_then_passes():
    """_async_throttle loops when over limit, then exits."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server.tokens_in_window", side_effect=[4_000_000, 0]):
        with patch("gpal.server.asyncio.sleep") as mock_sleep:
            await _async_throttle(MODEL_FLASH)
            assert mock_sleep.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# J. AFC Context Flag
# ─────────────────────────────────────────────────────────────────────────────


def test_afc_flag_set_during_send(tmp_path):
    """_afc_local.in_afc is True during send_message and False after."""
    observed = []

    class FakeSession:
        def send_message(self, parts, config=None):
            observed.append(getattr(_afc_local, "in_afc", False))

            class FakeResponse:
                text = "ok"
                candidates = []
                usage_metadata = None
            return FakeResponse()

    from google.genai import types
    result = _send_with_retry(FakeSession(), [], types.GenerateContentConfig())
    assert observed == [True], f"in_afc was {observed} during send_message"
    assert getattr(_afc_local, "in_afc", False) is False


def test_afc_flag_cleared_on_exception():
    """_afc_local.in_afc is cleared even when send_message raises."""
    class FakeSession:
        def send_message(self, parts, config=None):
            raise RuntimeError("boom")

    from google.genai import types
    try:
        _send_with_retry(FakeSession(), [], types.GenerateContentConfig())
    except RuntimeError:
        pass
    assert getattr(_afc_local, "in_afc", False) is False


def test_executor_has_gpal_prefix():
    """_EXECUTOR threads use 'gpal' prefix."""
    assert _EXECUTOR._thread_name_prefix == "gpal"
    assert _EXECUTOR._max_workers == 16
