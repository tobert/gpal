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
    mcp, _validate_input_path, _validate_output_path,
    record_tokens, tokens_in_window, token_stats, GeminiResponse,
    _sync_throttle, _async_throttle, _throttle_delay, _KNOWN_MODELS,
    MODEL_SEARCH, MODEL_LITE, RATE_LIMITS_TPM, _afc_local,
    _send_with_retry, _EXECUTOR,
    _extract_retry_delay, _is_retriable_genai_error,
    search_project, _sanitize_history,
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
    "create_file_store",
    "list_file_stores",
    "delete_file_store",
    "upload_to_file_store",
    "list_models",
    "generate_image",
    "generate_speech",
    "create_batch",
    "get_batch",
    "list_batches",
    "get_batch_results",
    "cancel_batch",
    "delete_batch",
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

        # consult_gemini should have 'query' required and 'model', 'thinking' optional
        schema = by_name["consult_gemini"].inputSchema
        assert "query" in schema.get("properties", {}), "consult_gemini missing 'query' param"
        assert "query" in schema.get("required", []), "consult_gemini: 'query' not required"
        assert "model" in schema.get("properties", {}), "consult_gemini missing 'model' param"
        assert "thinking" in schema.get("properties", {}), "consult_gemini missing 'thinking' param"

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
    "consult_gemini_oneshot": 600,
    "create_batch": 120,
    "get_batch": 30,
    "list_batches": 30,
    "get_batch_results": 60,
    "cancel_batch": 30,
    "delete_batch": 30,
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
        assert "lite" in data["models"]
        assert "flash" in data["models"]
        assert "token_usage" in data


# ─────────────────────────────────────────────────────────────────────────────
# I. Throttle Helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_model_search_in_known_models():
    """MODEL_SEARCH is tracked (present in _KNOWN_MODELS and RATE_LIMITS_TPM)."""
    assert MODEL_SEARCH in _KNOWN_MODELS
    assert MODEL_SEARCH in RATE_LIMITS_TPM


def test_model_lite_in_known_models():
    """MODEL_LITE is tracked (present in _KNOWN_MODELS and RATE_LIMITS_TPM)."""
    assert MODEL_LITE in _KNOWN_MODELS
    assert MODEL_LITE in RATE_LIMITS_TPM


def test_throttle_delay_under_limit():
    """_throttle_delay returns 0 when usage is under 90%."""
    from gpal.server import MODEL_FLASH
    assert _throttle_delay(MODEL_FLASH) == 0.0


def test_throttle_delay_over_limit():
    """_throttle_delay calculates exact sleep time from token window."""
    from gpal.server import MODEL_FLASH, _token_windows, _token_lock
    limit = RATE_LIMITS_TPM[MODEL_FLASH]
    now = time.monotonic()
    # Record tokens that will expire at now + 60 - make it over 90%
    with _token_lock:
        _token_windows[MODEL_FLASH] = [(now, limit)]  # 100% usage
    try:
        delay = _throttle_delay(MODEL_FLASH)
        # Should be ~60s (tokens recorded at 'now', expire at now+60)
        assert 55.0 < delay <= 61.0
    finally:
        with _token_lock:
            _token_windows[MODEL_FLASH] = []


def test_throttle_delay_unknown_model():
    """_throttle_delay returns 0 for models not in RATE_LIMITS_TPM."""
    assert _throttle_delay("some-unknown-model") == 0.0


def test_sync_throttle_sleeps_exact():
    """_sync_throttle sleeps for the delay from _throttle_delay."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=5.0):
        with patch("gpal.server.time.sleep") as mock_sleep:
            _sync_throttle(MODEL_FLASH)
            mock_sleep.assert_called_once_with(5.0)


def test_sync_throttle_no_sleep_when_clear():
    """_sync_throttle doesn't sleep when _throttle_delay returns 0."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=0.0):
        with patch("gpal.server.time.sleep") as mock_sleep:
            _sync_throttle(MODEL_FLASH)
            mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_async_throttle_sleeps_exact():
    """_async_throttle sleeps for the delay from _throttle_delay."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=5.0):
        with patch("gpal.server.asyncio.sleep") as mock_sleep:
            await _async_throttle(MODEL_FLASH)
            mock_sleep.assert_called_once_with(5.0)


@pytest.mark.asyncio
async def test_async_throttle_no_sleep_when_clear():
    """_async_throttle doesn't sleep when _throttle_delay returns 0."""
    from gpal.server import MODEL_FLASH
    with patch("gpal.server._throttle_delay", return_value=0.0):
        with patch("gpal.server.asyncio.sleep") as mock_sleep:
            await _async_throttle(MODEL_FLASH)
            mock_sleep.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# J. History Sanitization
# ─────────────────────────────────────────────────────────────────────────────


class _FakeContent:
    """Minimal stand-in for a Gemini history entry."""
    def __init__(self, role, parts=None):
        self.role = role
        self.parts = parts or []


class _FakePart:
    """Minimal stand-in for a Gemini content part."""
    def __init__(self, function_call=None):
        self.function_call = function_call


def test_sanitize_history_clean():
    """No-op when history is already valid (ends with model, no orphans)."""
    history = [_FakeContent("user"), _FakeContent("model")]
    assert _sanitize_history(history) is False
    assert len(history) == 2


def test_sanitize_history_trailing_user():
    """Strips trailing user turn."""
    history = [_FakeContent("user"), _FakeContent("model"), _FakeContent("user")]
    assert _sanitize_history(history) is True
    assert len(history) == 2
    assert history[-1].role == "model"


def test_sanitize_history_orphaned_function_call():
    """Strips model response with function_call that has no function_response."""
    fc_part = _FakePart(function_call={"name": "read_file", "args": {}})
    history = [
        _FakeContent("user"),
        _FakeContent("model"),
        _FakeContent("user"),
        _FakeContent("model", parts=[fc_part]),  # orphaned function_call
    ]
    assert _sanitize_history(history) is True
    # Should drop the orphaned model turn and its preceding user turn
    assert len(history) == 2
    assert history[-1].role == "model"


def test_sanitize_history_empty():
    """No-op on empty history."""
    history = []
    assert _sanitize_history(history) is False
    assert len(history) == 0


def test_sanitize_history_user_then_orphaned_fc():
    """Handles trailing user + orphaned function_call in sequence."""
    fc_part = _FakePart(function_call={"name": "search", "args": {}})
    history = [
        _FakeContent("user"),
        _FakeContent("model"),
        _FakeContent("user"),
        _FakeContent("model", parts=[fc_part]),
        _FakeContent("user"),  # trailing user turn
    ]
    assert _sanitize_history(history) is True
    # Strips trailing user, then finds orphaned fc, strips that + its user
    assert len(history) == 2


# ─────────────────────────────────────────────────────────────────────────────
# K. AFC Context Flag
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


# ─────────────────────────────────────────────────────────────────────────────
# I. Retry Delay Extraction
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_retry_delay_from_retry_info():
    """Extracts retryDelay from google.rpc.RetryInfo in error details."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {
        "error": {
            "code": 429,
            "message": "Rate limited",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18.5s"},
            ],
        }
    })
    delay = _extract_retry_delay(exc)
    assert delay == pytest.approx(18.5)


def test_extract_retry_delay_integer_seconds():
    """Handles integer-style retryDelay like '18s'."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {
        "error": {
            "code": 429,
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18s"},
            ],
        }
    })
    assert _extract_retry_delay(exc) == 18.0


def test_extract_retry_delay_no_retry_info():
    """Returns None when no RetryInfo is present."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {"error": {"code": 429, "details": []}})
    assert _extract_retry_delay(exc) is None


def test_extract_retry_delay_not_api_error():
    """Returns None for non-APIError exceptions."""
    assert _extract_retry_delay(RuntimeError("boom")) is None


def test_extract_retry_delay_string_error():
    """Handles error responses where 'error' is a string, not a dict."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {"error": "Model overloaded"})
    assert _extract_retry_delay(exc) is None


def test_is_retriable_429():
    """429 errors are retriable."""
    from google.genai.errors import ClientError
    exc = ClientError(429, {"error": {"code": 429}})
    assert _is_retriable_genai_error(exc) is True


def test_is_not_retriable_5xx():
    """5xx errors are not retriable — surface immediately."""
    from google.genai.errors import ServerError
    for code in (500, 502, 503, 504):
        exc = ServerError(code, {"error": {"code": code}})
        assert _is_retriable_genai_error(exc) is False, f"{code} should not be retriable"


def test_is_not_retriable_400():
    """400 Bad Request is not retriable."""
    from google.genai.errors import ClientError
    exc = ClientError(400, {"error": {"code": 400}})
    assert _is_retriable_genai_error(exc) is False


def test_is_not_retriable_non_api_error():
    """Non-APIError exceptions are not retriable."""
    assert _is_retriable_genai_error(RuntimeError("boom")) is False


# ─────────────────────────────────────────────────────────────────────────────
# K. Input Path Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_input_path_within_cwd(tmp_path, monkeypatch):
    """Paths under cwd are accepted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("hello")
    result = _validate_input_path(str(tmp_path / "data.txt"))
    assert result is None


def test_validate_input_path_rejects_traversal(tmp_path, monkeypatch):
    """Path traversal outside cwd is rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_input_path("../../../etc/passwd")
    assert result is not None
    assert "outside" in result.lower() or "denied" in result.lower()


def test_validate_input_path_rejects_absolute(tmp_path, monkeypatch):
    """Absolute paths to system dirs are rejected."""
    monkeypatch.chdir(tmp_path)
    result = _validate_input_path("/etc/passwd")
    assert result is not None
    assert "denied" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# L. Search Project Glob Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_search_project_rejects_absolute_glob(tmp_path, monkeypatch):
    """Absolute glob patterns are rejected early."""
    monkeypatch.chdir(tmp_path)
    result = search_project("test", glob_pattern="/etc/**/*")
    assert "absolute" in result.lower()


def test_search_project_rejects_traversal_glob(tmp_path, monkeypatch):
    """Glob patterns with '..' are rejected to prevent filesystem traversal DoS."""
    monkeypatch.chdir(tmp_path)
    result = search_project("test", glob_pattern="../../**/*")
    assert ".." in result
