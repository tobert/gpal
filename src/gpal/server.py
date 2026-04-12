"""
gpal - Your Pal Gemini

An MCP server providing stateful access to Google Gemini models with
autonomous codebase exploration capabilities.
"""

import argparse
import asyncio
import datetime
import io
import json
import logging
import os
import sys
import threading
import tomllib
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

import glob as globlib
from cachetools import TTLCache
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from google import genai
from google.genai import errors as genai_errors, types
from google.genai.types import FileSearch, GoogleSearch, Tool, ToolCodeExecution
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

import random
import time
from dataclasses import dataclass

# OpenTelemetry imports
from opentelemetry import trace, propagate
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from fastmcp.tools.tool import ToolResult
from gpal.git_tools import git

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Model versions (centralized for easy updates)
MODEL_LITE = "gemini-flash-lite-latest"       # Auto-updates to latest stable Lite
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_PRO = "gemini-3.1-pro-preview"
MODEL_SEARCH = "gemini-flash-latest"          # Auto-updates to latest stable Flash
MODEL_CODE_EXEC = "gemini-flash-latest"
MODEL_IMAGE = "imagen-4.0-ultra-generate-001"
MODEL_IMAGE_FAST = "imagen-4.0-fast-generate-001"
MODEL_IMAGE_PRO = "gemini-3-pro-image-preview"      # Gemini 3 Pro image generation
MODEL_IMAGE_FLASH = "gemini-2.5-flash-image"       # Nano Banana Flash
MODEL_SPEECH = "gemini-2.5-pro-preview-tts"
MODEL_SPEECH_FAST = "gemini-2.5-flash-preview-tts"

MODEL_ALIASES: dict[str, str] = {
    "lite": MODEL_LITE,
    "flash": MODEL_FLASH,
    "pro": MODEL_PRO,
    "imagen": MODEL_IMAGE,
    "imagen-fast": MODEL_IMAGE_FAST,
    "nano-pro": MODEL_IMAGE_PRO,
    "nano-flash": MODEL_IMAGE_FLASH,
    "speech": MODEL_SPEECH,
    "speech-fast": MODEL_SPEECH_FAST,
}

# Limits
MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB - prevents accidental DOS
MAX_INLINE_MEDIA = 20 * 1024 * 1024  # 20 MB - inline media limit
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
RESPONSE_MAX_TOOL_CALLS = 25
MAX_SEARCH_RESULTS = 10

# Retriable HTTP status codes from the Gemini API
# Only retry 429 (rate limited) — it's the one case where waiting helps,
# and often comes with RetryInfo. All 5xx = server-side problem, surface
# immediately so the caller can decide whether/when to retry.
_RETRIABLE_STATUS_CODES = {429}


def _is_retriable_genai_error(exc: BaseException) -> bool:
    """Check if a genai exception is retriable based on HTTP status code."""
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in _RETRIABLE_STATUS_CODES
    return False


def _format_api_error(exc: genai_errors.APIError, model_alias: str) -> str:
    """Format an API error into an actionable message for the MCP caller.

    503 errors get special treatment — they're transient capacity issues
    where the right move is to surface the problem and let the user decide.
    """
    code = getattr(exc, "code", None)
    resolved = MODEL_ALIASES.get(model_alias.lower(), model_alias)
    if code == 503:
        return (
            f"Error: {resolved} is temporarily unavailable (503 — high demand). "
            f"This is a transient Google-side capacity issue, not a bug. "
            f"Demand spikes usually clear within a few minutes. "
            f"Please wait briefly and retry the same call."
        )
    if _is_retriable_genai_error(exc):
        return f"Error: Service temporarily unavailable after retries: {exc}"
    return f"Error: {exc}"


def _extract_retry_delay(exc: BaseException) -> float | None:
    """Extract server-suggested retry delay from a Gemini API error.

    Parses the RetryInfo from the error details, e.g.:
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "18s"}
    """
    if not isinstance(exc, genai_errors.APIError):
        return None
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    # Navigate: top-level or nested under 'error' (guard against non-dict 'error' values)
    error_obj = details.get("error", details)
    if not isinstance(error_obj, dict):
        return None
    error_details = error_obj.get("details", [])
    if not isinstance(error_details, list):
        return None
    for detail in error_details:
        if not isinstance(detail, dict):
            continue
        if "RetryInfo" in detail.get("@type", ""):
            delay_str = detail.get("retryDelay", "")
            if isinstance(delay_str, str) and delay_str.endswith("s"):
                try:
                    return float(delay_str[:-1])
                except ValueError:
                    pass
    return None


def _wait_with_retry_delay(retry_state: Any) -> float:
    """Tenacity wait function that honors server-suggested retry delay.

    Falls back to exponential backoff with jitter when no RetryInfo is present.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc:
        delay = _extract_retry_delay(exc)
        if delay is not None and delay > 0:
            # Add small jitter (0-2s) to avoid thundering herd
            return delay + random.uniform(0, 2)
    # Fallback: exponential backoff with jitter
    return wait_exponential_jitter(initial=2, max=60, jitter=5)(retry_state)


def _before_sleep_with_mcp(retry_state: Any) -> None:
    """Log retry attempts via Python logging and MCP notifications when available.

    Reads MCP context from _afc_local thread-local (set by callers that have ctx).
    """
    # Standard Python logging (always)
    before_sleep_log(logging.getLogger(__name__), logging.WARNING)(retry_state)

    # MCP notification (when context available)
    ctx = getattr(_afc_local, "mcp_ctx", None)
    loop = getattr(_afc_local, "mcp_loop", None)
    if ctx and loop:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        attempt = retry_state.attempt_number
        delay = _extract_retry_delay(exc) if exc else None
        if delay:
            msg = f"Rate limited by Gemini API, retry {attempt}/5 in ~{delay:.0f}s (server requested)..."
        else:
            msg = f"Gemini API error, retry {attempt}/5..."
        try:
            asyncio.run_coroutine_threadsafe(ctx.warning(msg), loop).result(timeout=5)
        except Exception:
            pass  # Don't let MCP logging failures break retry logic


# Disable SDK-internal retry for calls wrapped by our decorator.
# The SDK has its own 5-attempt tenacity retry that ignores RetryInfo.
# We suppress it so our retry (which honors RetryInfo) handles everything.
_NO_SDK_RETRY = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))

# Retry configuration (tenacity handles all backoff)
GEMINI_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(5),
    wait=_wait_with_retry_delay,
    retry=retry_if_exception(_is_retriable_genai_error),
    before_sleep=_before_sleep_with_mcp,
    reraise=True,
)

# Dedicated thread pool for Gemini API calls (avoids starving the default executor)
_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gpal")

# Thread-local flag: True while inside Gemini's automatic function calling loop
_afc_local = threading.local()

# Semaphore limiting concurrent outbound API calls from AFC tool callbacks
_afc_api_semaphore = threading.Semaphore(4)

# ─────────────────────────────────────────────────────────────────────────────
# Token Tracking & Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────

# Published TPM limits (tokens per minute) — conservative defaults
RATE_LIMITS_TPM: dict[str, int] = {
    MODEL_LITE: 4_000_000,
    MODEL_PRO: 1_000_000,
    MODEL_FLASH: 4_000_000,
    MODEL_SEARCH: 4_000_000,  # Resolves to a Flash model
    MODEL_SPEECH: 2_000_000,
    MODEL_SPEECH_FAST: 4_000_000,
    MODEL_IMAGE_PRO: 1_000_000,
    MODEL_IMAGE_FLASH: 4_000_000,
}

_KNOWN_MODELS: frozenset[str] = frozenset(
    set(MODEL_ALIASES.values()) | set(RATE_LIMITS_TPM.keys())
)

_token_windows: dict[str, list[tuple[float, int]]] = {}  # model -> [(timestamp, tokens)]
_token_lock = threading.Lock()


def record_tokens(model: str, count: int) -> None:
    """Record token usage for rate limiting.

    Only tracks known models (MODEL_ALIASES values + RATE_LIMITS_TPM keys)
    to prevent unbounded memory growth from arbitrary model strings.
    """
    now = time.monotonic()
    cutoff = now - 60.0
    with _token_lock:
        # Cap to known models to prevent DoS via arbitrary model strings
        if model not in _KNOWN_MODELS:
            return
        window = _token_windows.setdefault(model, [])
        # Prune expired entries to prevent memory leak
        if window and window[0][0] < cutoff:
            _token_windows[model] = window = [(t, c) for t, c in window if t > cutoff]
        window.append((now, count))


def tokens_in_window(model: str, window_secs: float = 60.0) -> int:
    """Return tokens consumed in the last N seconds."""
    now = time.monotonic()
    cutoff = now - window_secs
    with _token_lock:
        window = _token_windows.get(model, [])
        # Prune expired entries
        _token_windows[model] = window = [(t, c) for t, c in window if t > cutoff]
        return sum(c for _, c in window)


def token_stats() -> dict[str, dict[str, int]]:
    """Return current token consumption per model (for gpal://info)."""
    with _token_lock:
        now = time.monotonic()
        cutoff = now - 60.0
        stats = {}
        for model, window in _token_windows.items():
            active = [(t, c) for t, c in window if t > cutoff]
            if active:
                limit = RATE_LIMITS_TPM.get(model, 0)
                stats[model] = {
                    "tokens_last_60s": sum(c for _, c in active),
                    "limit_tpm": limit,
                }
        return stats


def _throttle_delay(model: str) -> float:
    """Calculate seconds to sleep until token usage drops below 90% of limit.

    Returns 0.0 if no throttling needed. Walks the window oldest-first,
    accumulating tokens that will expire, and returns the time until enough
    expire to drop below the threshold.
    """
    limit = RATE_LIMITS_TPM.get(model, 0)
    if limit <= 0:
        return 0.0
    threshold = limit * 0.9
    now = time.monotonic()
    cutoff = now - 60.0
    with _token_lock:
        window = _token_windows.get(model, [])
        total = sum(c for t, c in window if t > cutoff)
        if total <= threshold:
            return 0.0
        # Walk oldest entries: each expires at (timestamp + 60s)
        excess = total - threshold
        for ts, count in sorted(window, key=lambda x: x[0]):
            if ts <= cutoff:
                continue
            excess -= count
            if excess <= 0:
                # This entry's expiry brings us under threshold
                delay = (ts + 60.0) - now + 0.1  # 100ms buffer
                return max(delay, 0.1)
    return 0.1  # shouldn't reach here, but don't spin


def _sync_throttle(model: str) -> None:
    """Block if approaching rate limit. For sync code paths."""
    delay = _throttle_delay(model)
    if delay > 0:
        logging.warning(f"Rate limit approaching for {model}, sleeping {delay:.1f}s")
        time.sleep(delay)


async def _async_throttle(model: str, ctx: Context | None = None) -> None:
    """Async version of rate limit throttle."""
    delay = _throttle_delay(model)
    if delay > 0:
        logging.warning(f"Rate limit approaching for {model}, sleeping {delay:.1f}s")
        if ctx:
            await ctx.warning(f"Rate limit approaching, throttling {delay:.1f}s...")
        await asyncio.sleep(delay)


@dataclass
class GeminiResponse:
    """Result from _send_with_retry including usage metadata."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# MIME type mappings for multimodal support
MIME_TYPES: dict[str, str] = {
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    # Video
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    # Audio
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    # Documents
    ".pdf": "application/pdf",
    # Text/Code (for upload_file with large files)
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".log": "text/plain",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/plain",
    ".tsx": "text/plain",
    ".jsx": "text/plain",
    ".vue": "text/plain",
    ".go": "text/plain",
    ".rs": "text/plain",
    ".rb": "text/plain",
    ".java": "text/plain",
    ".c": "text/plain",
    ".cpp": "text/plain",
    ".h": "text/plain",
    ".hpp": "text/plain",
    ".lua": "text/plain",
    ".sh": "text/x-shellscript",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".toml": "text/plain",
    ".xml": "application/xml",
    ".html": "text/html",
    ".css": "text/css",
    ".sql": "text/plain",
}


def detect_mime_type(path: str) -> str | None:
    """Detect MIME type from file extension, or None if unknown."""
    ext = Path(path).suffix.lower()
    return MIME_TYPES.get(ext)

# ── Role-specific system instructions ──────────────────────────────────────
# Base identity shared by all roles. Role-specific variants append to this.

_SYSTEM_BASE = """\
You are Gemini, a consultant AI accessed via the Model Context Protocol (MCP).

Tools available:
- list_directory, read_file, search_project — explore the local codebase
- git — read-only git operations (status, diff, log, show)
- gemini_search — search the web via Google Search

File stores are searched automatically when available.
Use tools proactively — don't guess when you can look it up.\
"""

# Lite in auto mode — cheap, aggressive, thorough exploration
_SYSTEM_EXPLORER = _SYSTEM_BASE + """

Your job is exploration — loading comprehensive codebase context for a \
synthesis model that follows. Read files in full, follow imports across \
modules, and include tests. Thoroughness over efficiency.\
"""

# Pro synthesis — deep reasoning with tools available to fill gaps
_SYSTEM_THINKER = _SYSTEM_BASE + """

An exploration agent has already loaded codebase context.

Reason deeply: trace contributing factors, weigh trade-offs, \
explain the why before proposing the how.\
"""

# Flash synthesis — fast insight with tools available to fill gaps
_SYSTEM_ANALYST = _SYSTEM_BASE + """

An exploration agent has already loaded codebase context.

Synthesize: identify what matters most, surface non-obvious \
connections, and deliver actionable conclusions. Concise over exhaustive.\
"""

# Direct model calls (flash or pro with tools enabled)
_SYSTEM_AGENT = _SYSTEM_BASE + """

Look at the code first, then say what you think.\
"""

# Kept as the default for _build_system_instruction and backward compat
DEFAULT_SYSTEM_INSTRUCTION = _SYSTEM_AGENT

# Composed user instruction layers — set by main(), appended after role prompts
_user_instruction: str = ""
_user_instruction_sources: list[str] = []

logger = logging.getLogger("gpal")


def _load_config() -> dict:
    """Load config from $XDG_CONFIG_HOME/gpal/config.toml (or ~/.config/gpal/config.toml).

    Returns parsed dict, or empty dict if file doesn't exist.
    Logs a warning on parse errors (non-fatal).
    """
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    config_path = Path(config_home) / "gpal" / "config.toml"

    if not config_path.is_file():
        logger.debug("No config file at %s", config_path)
        return {}

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        logger.info("Loaded config from %s", config_path)
        return config
    except tomllib.TOMLDecodeError as e:
        logger.warning("Invalid TOML in %s: %s", config_path, e)
        return {}


def _build_system_instruction(
    config: dict,
    cli_prompt_files: list[str] | None = None,
    no_default: bool = False,
) -> tuple[str, list[str]]:
    """Compose the user instruction layers from config, files, and CLI flags.

    Returns (instruction_text, list_of_sources) where sources describes
    provenance for debugging. The result is appended after the role-specific
    system prompt (explorer/thinker/agent) at call time.
    """
    parts: list[str] = []
    sources: list[str] = []

    # 1. Built-in default (unless suppressed)
    include_default = config.get("include_default_prompt", True)
    if no_default:
        include_default = False

    if include_default:
        sources.append("built-in (role-specific)")

    # 2. Files from config.toml system_prompts list
    config_prompts = config.get("system_prompts", [])
    if not isinstance(config_prompts, list):
        logger.warning("Config 'system_prompts' must be a list, got %s", type(config_prompts).__name__)
        config_prompts = []
    for path_str in config_prompts:
        expanded = Path(os.path.expandvars(os.path.expanduser(path_str)))
        try:
            content = expanded.read_text(encoding="utf-8")
            parts.append(content.strip())
            sources.append(str(expanded))
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Error reading system prompt %s: %s", expanded, e)

    # 3. Inline system_prompt from config.toml
    inline = config.get("system_prompt")
    if inline and isinstance(inline, str):
        parts.append(inline.strip())
        sources.append("config.toml (inline)")

    # 4. CLI --system-prompt files
    for path_str in cli_prompt_files or []:
        expanded = Path(os.path.expandvars(os.path.expanduser(path_str)))
        try:
            content = expanded.read_text(encoding="utf-8")
            parts.append(content.strip())
            sources.append(f"--system-prompt {expanded}")
        except FileNotFoundError:
            logger.warning("CLI system prompt file not found: %s", expanded)
        except OSError as e:
            logger.warning("Error reading CLI system prompt %s: %s", expanded, e)

    return "\n\n".join(parts), sources


def _compose_instruction(role_prompt: str) -> str:
    """Combine a role-specific prompt with user instruction layers."""
    if _user_instruction:
        return role_prompt.strip() + "\n\n" + _user_instruction
    return role_prompt.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Server & State
# ─────────────────────────────────────────────────────────────────────────────

async def _stdin_watchdog() -> None:
    """Exit if the MCP client disconnects (stdin fd closed).

    Polls every 5s using os.fstat(). If the fd becomes invalid (client
    crashed, pipe broken), the server exits cleanly. This prevents orphaned
    gpal processes from spinning CPU after the client disappears.

    Uses fstat() — not read() — to avoid consuming bytes from the MCP
    stdio transport stream.
    """
    try:
        fd = sys.stdin.fileno()
    except (ValueError, io.UnsupportedOperation):
        logger.warning("stdin has no fileno — watchdog disabled")
        return

    while True:
        await asyncio.sleep(5)
        try:
            os.fstat(fd)
        except OSError:
            logger.info("stdin fd invalid — client disconnected, exiting")
            os._exit(0)


@asynccontextmanager
async def _gpal_lifespan(app):
    """FastMCP lifespan: start/cancel the stdin watchdog."""
    task = asyncio.create_task(_stdin_watchdog())
    logger.info("stdin watchdog started")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("stdin watchdog stopped")


mcp = FastMCP("gpal", lifespan=_gpal_lifespan)
sessions: TTLCache = TTLCache(maxsize=100, ttl=3600)  # Stores (session, lock)
sessions_lock = threading.Lock()
# Note: session_locks dict is removed; locks are now bundled with sessions.
uploaded_files: TTLCache = TTLCache(maxsize=200, ttl=3600)  # 1 hour TTL
uploaded_files_lock = threading.Lock()
# FileSearch store tracking — lazily populated on first consult
_active_stores: set[str] = set()
_stores_lock = threading.Lock()
_stores_initialized = False
_stores_last_attempt: float = 0.0  # monotonic timestamp of last failed attempt


# ─────────────────────────────────────────────────────────────────────────────
# MCP Resources
# ─────────────────────────────────────────────────────────────────────────────


@mcp.resource("gpal://info")
def get_server_info() -> str:
    """Server version, model configuration, and limits."""
    from gpal import __version__

    info = {
        "version": __version__,
        "models": {
            "lite": MODEL_LITE,
            "flash": MODEL_FLASH,
            "pro": MODEL_PRO,
            "search": MODEL_SEARCH,
            "code_exec": MODEL_CODE_EXEC,
            "image": MODEL_IMAGE,
            "image_fast": MODEL_IMAGE_FAST,
            "image_pro": MODEL_IMAGE_PRO,
            "image_flash": MODEL_IMAGE_FLASH,
            "speech": MODEL_SPEECH,
            "speech_fast": MODEL_SPEECH_FAST,
        },
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
            "max_inline_media_mb": MAX_INLINE_MEDIA // (1024 * 1024),
            "max_search_files": MAX_SEARCH_FILES,
            "max_search_matches": MAX_SEARCH_MATCHES,
            "max_tool_calls": RESPONSE_MAX_TOOL_CALLS,
            "session_ttl_seconds": 3600,
            "max_sessions": 100,
        },
        "system_instruction": {
            "roles": ["explorer", "thinker", "analyst", "agent"],
            "user_layers": _user_instruction_sources,
            "agent_length_chars": len(_compose_instruction(_SYSTEM_AGENT)),
        },
        "token_usage": token_stats(),
        "batch": {
            "tools": ["create_batch", "get_batch", "list_batches", "get_batch_results", "cancel_batch", "delete_batch"],
            "discount": "~50%",
            "note": "No tool use in batch — inline all context in prompt",
        },
    }
    return json.dumps(info, indent=2)


@mcp.resource("gpal://sessions")
def list_sessions_resource() -> str:
    """List active session IDs with model and history count."""
    with sessions_lock:
        session_list = []
        for session_id, (session, _) in sessions.items():
            model = getattr(session, "_gpal_model", "unknown")
            # Try _curated_history first, fall back to history
            history = getattr(session, "_curated_history", getattr(session, "history", []))
            history_count = len(history) if history else 0
            session_list.append({
                "session_id": session_id,
                "model": model,
                "history_count": history_count,
            })
    return json.dumps(session_list, indent=2)


@mcp.resource("gpal://session/{session_id}")
async def get_session_detail(session_id: str) -> str:
    """View conversation history for a session."""
    with sessions_lock:
        item = sessions.get(session_id)
        if not item:
            return json.dumps({"error": f"Session '{session_id}' not found or expired"})
        session, lock = item

    # Hold per-session lock while reading history
    async with lock:
        model = getattr(session, "_gpal_model", "unknown")
        history = getattr(session, "_curated_history", getattr(session, "history", []))

        # Convert history to serializable format
        messages = []
        for item in history or []:
            try:
                parts_text = []
                if hasattr(item, "parts"):
                    for part in item.parts:
                        # Handle text parts
                        if hasattr(part, "text") and part.text:
                            parts_text.append(part.text[:500])  # Truncate long messages
                        # Handle function calls
                        elif hasattr(part, "function_call") and part.function_call:
                            parts_text.append(f"[Function: {part.function_call.name}]")
                        # Handle function responses
                        elif hasattr(part, "function_response") and part.function_response:
                            parts_text.append(f"[Response: {part.function_response.name}]")
                        # Handle executable code
                        elif hasattr(part, "executable_code") and part.executable_code:
                            parts_text.append("[Code Execution]")
                        # Handle code execution results
                        elif hasattr(part, "code_execution_result") and part.code_execution_result:
                            parts_text.append("[Code Result]")

                if parts_text:
                    messages.append({
                        "role": getattr(item, "role", "unknown"),
                        "content": " ".join(parts_text),
                    })
            except Exception as e:
                logging.warning(f"Error serializing message in session {session_id}: {e}")
                continue

        return json.dumps({
            "session_id": session_id,
            "model": model,
            "message_count": len(messages),
            "messages": messages,
        }, indent=2)


@mcp.resource("gpal://file_stores")
def get_file_stores_info() -> str:
    """FileSearch store statistics."""
    try:
        client = get_client()
        stores = list(client.file_search_stores.list())
        if not stores:
            return json.dumps({"message": "No file search stores. Use create_file_store to create one."})
        # Refresh tracking
        with _stores_lock:
            _active_stores.clear()
            _active_stores.update(s.name for s in stores)
        result = {}
        for s in stores:
            result[s.name] = {
                "display_name": s.display_name,
                "active_documents": s.active_documents_count or 0,
                "pending_documents": s.pending_documents_count or 0,
                "failed_documents": s.failed_documents_count or 0,
                "size_bytes": s.size_bytes or 0,
                "create_time": str(s.create_time) if s.create_time else None,
                "update_time": str(s.update_time) if s.update_time else None,
            }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("gpal://caches")
def list_context_caches_resource() -> str:
    """List active Gemini context caches."""
    client = get_client()
    caches = []
    try:
        for cache in client.caches.list():
            caches.append({
                "name": cache.name,
                "display_name": cache.display_name,
                "model": cache.model,
                "expire_time": str(cache.expire_time),
                "tokens": getattr(cache.usage_metadata, "total_token_count", 0) if cache.usage_metadata else 0,
            })
    except Exception as e:
        return json.dumps({"error": str(e)})
    return json.dumps(caches, indent=2)


@mcp.resource("gpal://models/check")
def check_model_freshness() -> str:
    """Compare configured models against available models from the API.

    Lists each configured model, whether it exists in the API, and flags
    any -latest aliases that have resolved to newer versions.
    """
    configured = {
        "lite": MODEL_LITE,
        "flash": MODEL_FLASH,
        "pro": MODEL_PRO,
        "search": MODEL_SEARCH,
        "code_exec": MODEL_CODE_EXEC,
        "image": MODEL_IMAGE,
        "image_fast": MODEL_IMAGE_FAST,
        "image_pro": MODEL_IMAGE_PRO,
        "image_flash": MODEL_IMAGE_FLASH,
        "speech": MODEL_SPEECH,
        "speech_fast": MODEL_SPEECH_FAST,
    }

    try:
        client = get_client()
        available = {}
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            available[name] = m
    except Exception as e:
        return json.dumps({"error": f"Could not list models: {e}"})

    results = []
    for alias, model_id in configured.items():
        entry = {"alias": alias, "configured": model_id}
        if model_id in available:
            entry["status"] = "ok"
            model_obj = available[model_id]
            # Check if there's a display name or version hint
            if model_obj.display_name:
                entry["display_name"] = model_obj.display_name
        else:
            # Might be a -latest alias that resolves server-side
            entry["status"] = "not_listed" if "-latest" not in model_id else "alias"
        results.append(entry)

    return json.dumps({"models": results}, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Codebase Exploration Tools (Local)
# ─────────────────────────────────────────────────────────────────────────────


def _validate_input_path(path: str) -> str | None:
    """Validate that an input path is within the project root.
    Returns None if valid, or an error message string if not."""
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        if not p.is_relative_to(cwd):
            return f"Error: Access denied — '{path}' is outside the project root."
        return None
    except Exception as e:
        return f"Error validating path '{path}': {e}"


def _validate_output_path(path: str) -> str | None:
    """Validate that an output path is within the project root.

    Returns None if valid, or an error message string if not.
    The parent directory is created if it doesn't exist.
    """
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        if not p.is_relative_to(cwd):
            return f"Error: Access denied — '{path}' is outside the project root."
        p.parent.mkdir(parents=True, exist_ok=True)
        return None
    except Exception as e:
        return f"Error validating output path '{path}': {e}"


def list_directory(path: str = ".") -> list[str] | str:
    """List files and directories at the given path."""
    err = _validate_input_path(path)
    if err:
        logging.warning(err)
        return err

    try:
        p = Path(path).resolve()
        if not p.exists():
            msg = f"Error: Path '{path}' does not exist"
            logging.warning(msg)
            return msg
        return [item.name for item in p.iterdir()]
    except Exception as e:
        msg = f"Error listing directory: {e}"
        logging.error(msg)
        return msg


def read_file(path: str) -> str:
    """Read the content of a file (up to MAX_FILE_SIZE bytes)."""
    err = _validate_input_path(path)
    if err:
        return err

    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"Error: File '{path}' does not exist."

        if p.stat().st_size > MAX_FILE_SIZE:
            return f"Error: File '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def search_project(search_term: str, glob_pattern: str = "**/*") -> str:
    """
    Search for a text term in files matching the glob pattern.
    Returns a summary of matching files.
    """
    if Path(glob_pattern).is_absolute():
        return "Error: Absolute glob patterns are not allowed. Use relative patterns."

    if ".." in Path(glob_pattern).parts:
        return "Error: Glob patterns cannot contain '..'."

    try:
        cwd = Path.cwd().resolve()

        # Use iglob iterator to avoid loading huge file lists into memory
        matches = []
        files_checked = 0

        for filepath in globlib.iglob(glob_pattern, recursive=True):
            path_obj = Path(filepath).resolve()

            # Ensure file is within project root and is a file
            if not path_obj.is_relative_to(cwd) or not path_obj.is_file():
                continue

            files_checked += 1
            if files_checked > MAX_SEARCH_FILES:
                matches.append(f"... (search stopped: reached {MAX_SEARCH_FILES} file limit)")
                break

            try:
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    if search_term in f.read():
                        matches.append(f"Match in: {filepath}")
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            matches.append("... (truncated)")
                            return "\n".join(matches)
            except OSError:
                continue

        return "\n".join(matches) if matches else "No matches found."

    except Exception as e:
        return f"Error searching project: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Client & Session Management
# ─────────────────────────────────────────────────────────────────────────────

# Default key file locations (checked in order)
DEFAULT_KEY_FILES = [
    Path.home() / ".config" / "gpal" / "api_key",
    Path.home() / ".gemini-api-key",
]

# Set by --api-key-file CLI arg; checked first by _load_api_key()
_cli_key_file: Path | None = None


def _load_api_key() -> str | None:
    """Load API key from environment or key file."""
    # 1. Check CLI-provided key file (highest priority)
    if _cli_key_file is not None:
        try:
            api_key = _cli_key_file.read_text().strip()
            if api_key:
                logging.info(f"Loaded API key from {_cli_key_file} (--api-key-file)")
                return api_key
        except OSError as e:
            logging.warning(f"Could not read --api-key-file {_cli_key_file}: {e}")

    # 2. Check environment variables
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key

    # 3. Check default key files
    for key_file in DEFAULT_KEY_FILES:
        if key_file.exists():
            try:
                api_key = key_file.read_text().strip()
                if api_key:
                    logging.info(f"Loaded API key from {key_file}")
                    return api_key
            except OSError as e:
                logging.warning(f"Could not read {key_file}: {e}")

    return None


def get_client() -> genai.Client:
    """Create a Gemini API client from environment or key file."""
    api_key = _load_api_key()
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY environment variable "
            f"or create {DEFAULT_KEY_FILES[0]}"
        )
    return genai.Client(api_key=api_key)


def _ensure_stores_initialized() -> None:
    """Lazily populate _active_stores on first consult.

    Lists existing FileSearch stores from the API so that stores created
    in a previous server lifetime are available for AFC. On failure,
    retries after a 60-second cooldown to avoid blocking every request.
    """
    global _stores_initialized, _stores_last_attempt
    if _stores_initialized:
        return
    with _stores_lock:
        if _stores_initialized:
            return
        # 60-second cooldown on failures to prevent retry spam
        now = time.monotonic()
        if now - _stores_last_attempt < 60.0:
            return
        _stores_last_attempt = now
        try:
            client = get_client()
            for store in client.file_search_stores.list():
                _active_stores.add(store.name)
            _stores_initialized = True
        except Exception as e:
            logging.debug(f"Deferred FileSearch initialization: {e}")


def _build_afc_tools(include_search: bool = True) -> list:
    """Build the AFC tool list with UrlContext and conditional FileSearch.

    Args:
        include_search: Include gemini_search (web search) in the list.
            False for create_chat's minimal config.
    """
    # Note: url_context cannot be combined with Function Calling (API restriction)
    tools: list = [list_directory, read_file, search_project, git]
    if include_search:
        tools.append(gemini_search)
    # Add FileSearch if any stores are active
    _ensure_stores_initialized()
    with _stores_lock:
        store_names = list(_active_stores)
    if store_names:
        tools.append(Tool(file_search=FileSearch(file_search_store_names=store_names)))
    return tools


def create_chat(
    client: genai.Client,
    model_name: str,
    history: list[Any] | None = None,
    config: types.GenerateContentConfig | None = None,
) -> Any:
    """Create a configured Gemini chat session."""
    if config is None:
        config = types.GenerateContentConfig(
            temperature=0.2,
            tools=_build_afc_tools(include_search=False),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            system_instruction=_compose_instruction(_SYSTEM_AGENT),
        )

    return client.chats.create(
        model=model_name,
        history=history or [],
        config=config,
    )

def _sanitize_history(history: list) -> bool:
    """Remove trailing entries that would cause Gemini API errors.

    Gemini requires history to alternate user/model and end with a model
    response. This handles two cases:
    1. Trailing user turn (failed prior call left it dangling)
    2. Model response ending with function_call but no matching
       function_response from the user — Gemini will 400 on this

    Mutates `history` in place. Returns True if any entries were removed.
    """
    changed = False

    # Strip trailing user turns
    while history and getattr(history[-1], "role", "") == "user":
        history.pop()
        changed = True

    # Check if last model response contains an orphaned function_call
    # (function_call with no subsequent user function_response)
    if history and getattr(history[-1], "role", "") == "model":
        last = history[-1]
        parts = getattr(last, "parts", []) or []
        has_function_call = any(
            getattr(p, "function_call", None) is not None for p in parts
        )
        if has_function_call:
            # The function_call never got a response — drop this exchange
            history.pop()  # orphaned model function_call
            changed = True
            # The preceding user turn is now dangling too
            while history and getattr(history[-1], "role", "") == "user":
                history.pop()

    return changed


async def get_session(
    ctx: Context,
    client: genai.Client,
    model_alias: str,
    config: types.GenerateContentConfig | None = None,
) -> tuple[Any, asyncio.Lock]:
    """Get or create a session, bundled with its own lock."""
    session_id = ctx.session_id
    target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)

    with sessions_lock:
        if session_id in sessions:
            session, lock = sessions[session_id]
        else:
            session = create_chat(client, target_model, config=config)
            session._gpal_model = target_model
            lock = asyncio.Lock()
            sessions[session_id] = (session, lock)
            ctx.set_state("model", target_model)
            return session, lock

    # Use per-session lock for migration and history sanitization
    async with lock:
        current_model = getattr(session, "_gpal_model", None)
        history = list(getattr(session, "_curated_history", getattr(session, "history", [])))

        needs_recreate = _sanitize_history(history)

        if current_model == target_model and not needs_recreate:
            return session, lock

        logging.info(f"Recreating session '{session_id}': {current_model} → {target_model}, sanitized={needs_recreate}")
        try:
            session = create_chat(client, target_model, history=history, config=config)
        except Exception as e:
            logging.error(f"Session recreation failed: {e}")
            session = create_chat(client, target_model, config=config)

        session._gpal_model = target_model
        with sessions_lock:
            sessions[session_id] = (session, lock)
        ctx.set_state("model", target_model)
        return session, lock


# ─────────────────────────────────────────────────────────────────────────────
# Internal Implementations (Testable)
# ─────────────────────────────────────────────────────────────────────────────


@GEMINI_RETRY_DECORATOR
def _gemini_search(
    query: str,
    num_results: int = 5,
    model: str = MODEL_SEARCH,
) -> str:
    """Internal implementation of web search with automatic retry."""
    in_afc = getattr(_afc_local, "in_afc", False)
    # Skip throttle when called from AFC — the outer _consult already throttled
    if not in_afc:
        _sync_throttle(model)
    client = get_client()

    # Clamp num_results
    num_results = max(1, min(num_results, MAX_SEARCH_RESULTS))

    # Cap concurrent outbound API calls when inside AFC
    if in_afc:
        _afc_api_semaphore.acquire()
    try:
        # Single stateless API call with Google Search tool
        response = client.models.generate_content(
            model=model,
            contents=f"Search for: {query}\n\nProvide the top {num_results} most relevant results with titles, URLs, and summaries.",
            config=types.GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())],
                temperature=0.1,
                http_options=_NO_SDK_RETRY,
            ),
        )

        # Record token usage
        usage = getattr(response, "usage_metadata", None)
        total = getattr(usage, "total_token_count", 0) or 0
        if total > 0:
            record_tokens(model, total)

        # Extract text response
        if response.candidates and response.candidates[0].content.parts:
            result_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    result_text += part.text
            return result_text.strip() or "No results returned."

        return "No results returned."
    finally:
        if in_afc:
            _afc_api_semaphore.release()


@GEMINI_RETRY_DECORATOR
def _gemini_code_exec(
    code: str,
    model: str = MODEL_CODE_EXEC,
) -> str:
    """Internal implementation of code execution with automatic retry."""
    _sync_throttle(model)
    client = get_client()

    # Single stateless API call with Code Execution tool
    response = client.models.generate_content(
        model=model,
        contents=f"Execute this Python code and show the output:\n\n```python\n{code}\n```",
        config=types.GenerateContentConfig(
            tools=[Tool(code_execution=ToolCodeExecution())],
            temperature=0,
            http_options=_NO_SDK_RETRY,
        ),
    )

    # Record token usage
    usage = getattr(response, "usage_metadata", None)
    total = getattr(usage, "total_token_count", 0) or 0
    if total > 0:
        record_tokens(model, total)

    # Extract execution results
    if response.candidates and response.candidates[0].content.parts:
        result_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                result_text += part.text
            elif hasattr(part, "executable_code"):
                result_text += f"Executed:\n{part.executable_code.code}\n"
            elif hasattr(part, "code_execution_result"):
                result_text += f"Output:\n{part.code_execution_result.output}\n"

        return result_text.strip() or "Code executed (no output)."

    return "Code executed (no output)."


async def _consult(
    query: str,
    ctx: Context,
    model_alias: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
    cached_content: str | None = None,
    role: str = "agent",
    thinking: str | None = None,
) -> str | ToolResult:
    """Send a query to Gemini with codebase context."""
    t0 = time.monotonic()
    tracer = trace.get_tracer("gpal")

    with tracer.start_as_current_span("gemini_call") as span:
        span.set_attribute("gpal.model_alias", model_alias)
        span.set_attribute("gpal.session_id", ctx.session_id)
        if cached_content:
            span.set_attribute("gpal.cached_content", cached_content)
        
        client = get_client()
        session_id = ctx.session_id

        def _wrap(resp: GeminiResponse) -> ToolResult:
            resolved = MODEL_ALIASES.get(model_alias.lower(), model_alias)
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            # Record token usage for rate limiting
            if resp.total_tokens > 0:
                record_tokens(resolved, resp.total_tokens)
            return ToolResult(
                content=resp.text,
                structured_content={"result": resp.text, "model": resolved},
                meta={
                    "model": resolved,
                    "session_id": session_id,
                    "duration_ms": elapsed_ms,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.total_tokens,
                },
            )

        # Build generation config
        # Tools + AFC always enabled so history with function calls stays valid
        # and synthesis models can fill gaps the explorer missed
        _role_prompts = {
            "explorer": _SYSTEM_EXPLORER,
            "thinker": _SYSTEM_THINKER,
            "analyst": _SYSTEM_ANALYST,
            "agent": _SYSTEM_AGENT,
        }
        role_prompt = _role_prompts.get(role, _SYSTEM_AGENT)
        config_kwargs: dict[str, Any] = {
            "temperature": 0.2,
            "tools": _build_afc_tools(include_search=True),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            "system_instruction": _compose_instruction(role_prompt),
            "http_options": _NO_SDK_RETRY,
        }

        # Thinking level: explicit parameter > role-based default (HIGH for thinker)
        _thinking = thinking
        if _thinking is None and role == "thinker":
            _thinking = "high"
        if _thinking is not None:
            _level_map = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
            sdk_level = _level_map.get(_thinking.lower())
            if not sdk_level:
                return f"Error: Invalid thinking level '{_thinking}'. Must be one of: minimal, low, medium, high."
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=sdk_level)

        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
            if response_schema:
                try:
                    config_kwargs["response_schema"] = json.loads(response_schema)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON schema: {e}"

        if cached_content:
            config_kwargs["cached_content"] = cached_content

        gen_config = types.GenerateContentConfig(**config_kwargs)

        session, lock = await get_session(ctx, client, model_alias, gen_config)

        parts: list[types.Part] = []
        loop = asyncio.get_running_loop()

        # Context: Text files (offloaded to thread pool to avoid blocking event loop)
        for path in file_paths or []:
            err = _validate_input_path(path)
            if err:
                return err
            try:
                p = Path(path)
                size = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.stat().st_size)
                if size > MAX_FILE_SIZE:
                    return f"Error: '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
                content = await loop.run_in_executor(
                    _EXECUTOR, lambda p=p: p.read_text(encoding="utf-8", errors="replace")
                )
                parts.append(types.Part.from_text(
                    text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
                ))
            except Exception as e:
                return f"Error reading file '{path}': {e}"

        # Context: Inline media (offloaded to thread pool)
        for path in media_paths or []:
            err = _validate_input_path(path)
            if err:
                return err
            try:
                p = Path(path)
                size = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.stat().st_size)
                if size > MAX_INLINE_MEDIA:
                    return f"Error: '{path}' exceeds {MAX_INLINE_MEDIA // (1024*1024)}MB inline limit."
                mime_type = detect_mime_type(path)
                if not mime_type:
                    return f"Error: Unknown media type for '{path}'."
                data = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.read_bytes())
                parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            except Exception as e:
                return f"Error reading media '{path}': {e}"

        # Context: File URIs (from upload_file or Gemini Files API)
        for uri in file_uris or []:
            mime = None
            # Check local cache first (populated by upload_file)
            with uploaded_files_lock:
                cached = uploaded_files.get(uri)
            if cached and cached.mime_type:
                mime = cached.mime_type
            else:
                # Try the Files API for URIs not in our cache
                try:
                    file_id = uri.rstrip("/").rsplit("/", 1)[-1]
                    file_meta = client.files.get(name=f"files/{file_id}")
                    mime = file_meta.mime_type
                except Exception:
                    pass
            parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime))

        parts.append(types.Part.from_text(text=query))

        # Proactive rate limiting with re-check and jitter
        resolved_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)
        await _async_throttle(resolved_model, ctx)

        # Send with one retry on stale client
        for attempt in range(2):
            async with lock:
                try:
                    def _run():
                        _afc_local.mcp_ctx = ctx
                        _afc_local.mcp_loop = loop
                        try:
                            return _send_with_retry(session, parts, gen_config)
                        finally:
                            _afc_local.mcp_ctx = None
                            _afc_local.mcp_loop = None
                    resp = await loop.run_in_executor(_EXECUTOR, _run)
                    return _wrap(resp)
                except genai_errors.APIError as e:
                    return _format_api_error(e, model_alias)
                except Exception as e:
                    error_msg = str(e).lower()
                    is_stale = "client" in error_msg and ("closed" in error_msg or "shutdown" in error_msg)
                    if not is_stale or attempt > 0:
                        return f"Error: {e}"

                    logging.warning(f"Session '{session_id}' has stale client, recreating...")
                    try:
                        prev_history = list(getattr(session, "_curated_history", getattr(session, "history", [])))
                        _sanitize_history(prev_history)
                        new_client = get_client()
                        target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)
                        session = create_chat(new_client, target_model, history=prev_history, config=gen_config)
                        session._gpal_model = target_model
                    except Exception as retry_e:
                        return f"Error recreating session: {retry_e}"

            # Update global state before retry
            with sessions_lock:
                sessions[session_id] = (session, lock)

@GEMINI_RETRY_DECORATOR
def _send_with_retry(session: Any, parts: list[types.Part], config: types.GenerateContentConfig) -> GeminiResponse:
    """Send message with automatic retry on transient errors."""
    _afc_local.in_afc = True
    try:
        response = session.send_message(parts, config=config)
    finally:
        _afc_local.in_afc = False

    # Extract usage metadata
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", 0) or 0

    try:
        if response.text:
            return GeminiResponse(response.text, prompt_tokens, completion_tokens, total_tokens)
    except ValueError:
        pass  # No text parts — fall through to detailed diagnostics

    # Handle cases where no text was generated (e.g., max tool calls, safety block)
    candidate = response.candidates[0] if response.candidates else None

    # finish_reason might be an enum or int, try to get a string representation
    finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
    if hasattr(finish_reason, "name"):
        finish_reason = finish_reason.name

    details = []
    if candidate and candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                details.append(f"[Function Call: {part.function_call.name}]")
            elif hasattr(part, "function_response") and part.function_response:
                details.append(f"[Function Result: {part.function_response.name}]")
            elif hasattr(part, "executable_code") and part.executable_code:
                details.append("[Code Execution]")
            elif hasattr(part, "code_execution_result") and part.code_execution_result:
                details.append("[Code Result]")

    text = f"System: No text generated. Finish Reason: {finish_reason}. Partial content: {', '.join(details)}"
    return GeminiResponse(text, prompt_tokens, completion_tokens, total_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools (Exposed)
# ─────────────────────────────────────────────────────────────────────────────


def gemini_search(
    query: str,
    num_results: int = 5,
    model: str = MODEL_SEARCH,
) -> str:
    """
    Search the web using Gemini's built-in Google Search.

    Returns formatted search results (titles, URLs, snippets).
    Stateless utility.
    """
    return _gemini_search(query, num_results, model)

mcp.tool(gemini_search)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def gemini_code_exec(
    code: Annotated[str, Field(description="Python code to execute")],
    model: Annotated[str, Field(default=MODEL_CODE_EXEC, description="Model to use for code execution")] = MODEL_CODE_EXEC,
) -> str:
    """Execute Python code using Gemini's built-in code execution sandbox.

    Returns stdout, stderr, and any execution results.
    Stateless utility."""
    return _gemini_code_exec(code, model)


_EXPLORE_PROMPT = """\
We are in exploration mode. Our goal is to load comprehensive codebase context \
for a follow-up synthesis phase. The synthesis model has tools but works best \
from context we've already loaded — our thoroughness here saves it round-trips.

TASK: {query}

STRATEGY:
- Start with `search_project` to find entry points, then `list_directory` to map structure.
- **Read files in full.** Complete source files enable holistic observations that snippets miss.
- Read related files: tests, configs, type definitions, imports. Cast a wide net.
- Follow imports, call chains, and data flow wherever they lead.
- Use `git` to check recent changes, blame, or history when provenance matters.

WE DO NOT:
- Propose fixes, write code, or provide recommendations in this phase.
- Stop until we have a clear picture of the task's "surface area."
- Skip files to save tokens — thoroughness is the priority.

OUTPUT — Structured Inventory (the synthesis model reads this summary AND the \
full file contents loaded into conversation history):
1. **Key Modules**: File paths with brief roles and important line numbers.
2. **Data Flow**: How data moves through the components found.
3. **Patterns & Constants**: Architectural patterns, config values, shared conventions.
4. **Files Read in Full**: List every file read completely.
5. **Blind Spots**: Files referenced but not found, unclear ownership, anything unresolved.
"""

_SYNTHESIS_PROMPT = """\
We are in the SYNTHESIS phase. An exploration agent has loaded extensive codebase \
context into this conversation. We have tools available to fill gaps, but most \
context is already present.

CONTEXT:
The conversation history contains complete file contents, search results, and \
directory listings from a thorough exploration agent.

TASK: {query}

APPROACH:
1. **Analysis**: Use the loaded source code to explain *why* things are the way \
they are. Identify the specific logic relevant to the task.
2. **Response**: Match the form to the question — explain if asked to explain, \
propose changes if asked to change, review if asked to review.
3. **Gaps**: If the exploration missed something critical, use our tools to fill it.
4. **Specificity**: When proposing code changes, include exact file paths and enough \
surrounding context so changes can be applied unambiguously.

GOAL: A complete, actionable response grounded in the code we've read.
"""


@mcp.tool(timeout=660, annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def consult_gemini(
    query: Annotated[str, Field(description="The question or instruction")],
    model: Annotated[str, Field(default="auto", description='"auto" (default, Lite explore → Flash analyze), "flash", "pro", "lite", or full model ID')] = "auto",
    file_paths: Annotated[list[str] | None, Field(default=None, description="Local text files to read and include inline as context")] = None,
    media_paths: Annotated[list[str] | None, Field(default=None, description="Local image/PDF files (.png, .jpg, .webp, .gif, .pdf) for vision analysis")] = None,
    file_uris: Annotated[list[str] | None, Field(default=None, description="Gemini File API URIs (from upload_file). Use for large files that exceed inline limits")] = None,
    json_mode: Annotated[bool, Field(default=False, description="Return structured JSON output")] = False,
    response_schema: Annotated[str | None, Field(default=None, description="JSON schema string for structured output")] = None,
    cached_content: Annotated[str | None, Field(default=None, description="Gemini context cache name")] = None,
    thinking: Annotated[str | None, Field(default=None, description='Thinking level: "minimal", "low", "medium", "high", or None. Pro defaults to "high"')] = None,
    ctx: Context | None = None,
) -> str | ToolResult:
    """Consult Gemini for codebase analysis.

    Gemini autonomously explores our project — reading files, listing
    directories, and searching code — so we don't need to pre-read files.
    Just describe what we need. Use file_paths only when specific files
    must be included.

    Pipeline: For auto, flash, and pro, Lite explores quickly first, then
    our selected model synthesizes. "lite" and explicit model IDs skip
    the exploration phase and query directly.
    Gemini's tools: list_directory, read_file, search_project."""
    if ctx:
        await ctx.debug(f"consult_gemini: model={model}, session={ctx.session_id}, files={len(file_paths or [])}")

    # Determine synthesis model — all recognized aliases except "lite" get Lite explore first
    if model == "auto":
        synth_model = "flash"
    elif model in ("flash", "pro"):
        synth_model = model
    elif model == "lite":
        # Direct pass-through, no explore phase (Lite doesn't support thinking)
        return await _consult(
            query, ctx, "lite", file_paths,
            media_paths, file_uris, json_mode, response_schema, cached_content
        )
    else:
        # Full model ID → direct pass-through
        return await _consult(
            query, ctx, model, file_paths,
            media_paths, file_uris, json_mode, response_schema, cached_content,
            thinking=thinking
        )

    # Phase 1: Lite explores (aggressive tool use, loads full file contents)
    if ctx:
        await ctx.info("Phase 1: Lite exploration...")
    explore_query = _EXPLORE_PROMPT.format(query=query)
    explore_result = await _consult(
        explore_query, ctx, "lite", file_paths,
        media_paths, file_uris, False, None, cached_content,
        role="explorer"
    )

    # Check for internal system errors (plain str return = system failure)
    # ToolResult means the model responded — even if its text mentions "Error"
    if isinstance(explore_result, str) and explore_result.startswith("Error:"):
        return explore_result

    # Phase 2: Synthesis model analyzes (tools available to fill gaps)
    synth_role = "thinker" if synth_model == "pro" else "analyst"
    if ctx:
        await ctx.info(f"Phase 2: {synth_model.capitalize()} synthesis (role={synth_role})...")
    synthesis_query = _SYNTHESIS_PROMPT.format(query=query)
    return await _consult(
        synthesis_query, ctx, synth_model, None,
        None, None, json_mode, response_schema, cached_content,
        role=synth_role, thinking=thinking
    )


@mcp.tool(timeout=600, annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def consult_gemini_oneshot(
    query: Annotated[str, Field(description="The question or instruction")],
    model: Annotated[str, Field(default="pro", description='"flash", "pro" (default), or full model ID')] = "pro",
    file_paths: Annotated[list[str] | None, Field(default=None, description="Local text files to read and include inline as context")] = None,
    media_paths: Annotated[list[str] | None, Field(default=None, description="Local image/PDF files (.png, .jpg, .webp, .gif, .pdf) for vision analysis")] = None,
    file_uris: Annotated[list[str] | None, Field(default=None, description="Gemini File API URIs (from upload_file). Use for large files that exceed inline limits")] = None,
    json_mode: Annotated[bool, Field(default=False, description="Return structured JSON output")] = False,
    response_schema: Annotated[str | None, Field(default=None, description="JSON schema string for structured output")] = None,
    cached_content: Annotated[str | None, Field(default=None, description="Gemini context cache name")] = None,
    thinking: Annotated[str | None, Field(default=None, description='Thinking level: "minimal", "low", "medium", "high", or None. Pro defaults to "high"')] = None,
    ctx: Context | None = None,
) -> str | ToolResult:
    """Stateless single-shot Gemini query with no session history.

    Use for independent questions, one-off lookups, or batch-style queries
    where conversation context would be noise. Still has tool access
    (list_directory, read_file, etc.) and retry logic."""
    t0 = time.monotonic()
    tracer = trace.get_tracer("gpal")

    with tracer.start_as_current_span("gemini_oneshot") as span:
        resolved_model = MODEL_ALIASES.get(model.lower(), model)
        span.set_attribute("gpal.model", resolved_model)

        client = get_client()
        loop = asyncio.get_running_loop()

        # Build parts
        parts: list[types.Part] = []

        for path in file_paths or []:
            err = _validate_input_path(path)
            if err:
                return err
            try:
                p = Path(path)
                size = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.stat().st_size)
                if size > MAX_FILE_SIZE:
                    return f"Error: '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
                content = await loop.run_in_executor(
                    _EXECUTOR, lambda p=p: p.read_text(encoding="utf-8", errors="replace")
                )
                parts.append(types.Part.from_text(
                    text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
                ))
            except Exception as e:
                return f"Error reading file '{path}': {e}"

        for path in media_paths or []:
            err = _validate_input_path(path)
            if err:
                return err
            try:
                p = Path(path)
                size = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.stat().st_size)
                if size > MAX_INLINE_MEDIA:
                    return f"Error: '{path}' exceeds {MAX_INLINE_MEDIA // (1024*1024)}MB inline limit."
                mime_type = detect_mime_type(path)
                if not mime_type:
                    return f"Error: Unknown media type for '{path}'."
                data = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.read_bytes())
                parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            except Exception as e:
                return f"Error reading media '{path}': {e}"

        for uri in file_uris or []:
            mime = None
            with uploaded_files_lock:
                cached = uploaded_files.get(uri)
            if cached and cached.mime_type:
                mime = cached.mime_type
            else:
                try:
                    file_id = uri.rstrip("/").rsplit("/", 1)[-1]
                    file_meta = client.files.get(name=f"files/{file_id}")
                    mime = file_meta.mime_type
                except Exception:
                    pass
            parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime))

        parts.append(types.Part.from_text(text=query))

        # Build config
        config_kwargs: dict[str, Any] = {
            "temperature": 0.2,
            "tools": _build_afc_tools(include_search=True),
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            "system_instruction": _compose_instruction(_SYSTEM_AGENT),
            "http_options": _NO_SDK_RETRY,
        }
        if cached_content:
            config_kwargs["cached_content"] = cached_content
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
            if response_schema:
                try:
                    config_kwargs["response_schema"] = json.loads(response_schema)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON schema: {e}"

        # Thinking level: explicit parameter > model-based default (HIGH for Pro)
        _thinking = thinking
        if _thinking is None and resolved_model == MODEL_PRO:
            _thinking = "high"
        if _thinking is not None:
            _level_map = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
            sdk_level = _level_map.get(_thinking.lower())
            if not sdk_level:
                return f"Error: Invalid thinking level '{_thinking}'. Must be one of: minimal, low, medium, high."
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=sdk_level)

        gen_config = types.GenerateContentConfig(**config_kwargs)

        # Proactive rate limiting with re-check and jitter
        await _async_throttle(resolved_model)

        # Stateless call via chats.create with empty history
        try:
            session = create_chat(client, resolved_model, config=gen_config)
            def _run():
                _afc_local.mcp_ctx = ctx
                _afc_local.mcp_loop = loop
                try:
                    return _send_with_retry(session, parts, gen_config)
                finally:
                    _afc_local.mcp_ctx = None
                    _afc_local.mcp_loop = None
            resp = await loop.run_in_executor(_EXECUTOR, _run)

            if resp.total_tokens > 0:
                record_tokens(resolved_model, resp.total_tokens)

            elapsed_ms = round((time.monotonic() - t0) * 1000)
            return ToolResult(
                content=resp.text,
                structured_content={"result": resp.text, "model": resolved_model},
                meta={
                    "model": resolved_model,
                    "duration_ms": elapsed_ms,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.total_tokens,
                },
            )
        except genai_errors.APIError as e:
            return _format_api_error(e, model)
        except Exception as e:
            return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(idempotentHint=True, openWorldHint=True))
def upload_file(
    file_path: Annotated[str, Field(description="Path to the local file to upload")],
    display_name: Annotated[str | None, Field(default=None, description="Display name in the Files API (defaults to filename)")] = None,
) -> str:
    """Upload a large file to Gemini's File API."""
    err = _validate_input_path(file_path)
    if err:
        return err
    client = get_client()
    path = Path(file_path)
    if not path.exists():
        return f"Error: File '{file_path}' does not exist."
    mime_type = detect_mime_type(file_path)
    config = types.UploadFileConfig(display_name=display_name or path.name, mime_type=mime_type)
    try:
        file = client.files.upload(file=str(path), config=config)
        with uploaded_files_lock:
            uploaded_files[file.uri] = file
        return f"Uploaded: {file.uri}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(idempotentHint=True, openWorldHint=True))
def create_context_cache(
    file_uris: Annotated[list[str], Field(description="Gemini File API URIs (from upload_file)")],
    model: Annotated[str, Field(default="flash", description="Model alias or explicit version ID")] = "flash",
    display_name: Annotated[str | None, Field(default=None, description="Display name for the cache")] = None,
    ttl_seconds: Annotated[int, Field(default=3600, description="Cache TTL in seconds (default 1 hour)")] = 3600,
) -> str:
    """Create a Gemini context cache for a set of files.

    Caching is useful for large files (>32k tokens) used across multiple turns.
    Model must be an explicit version (e.g., gemini-1.5-flash-001) or a supported alias."""
    client = get_client()
    resolved_model = MODEL_ALIASES.get(model.lower(), model)
    
    # Caching requires stable versioned IDs, not short preview aliases
    original_model = resolved_model
    if resolved_model in ["gemini-3-flash-preview", "gemini-2.0-flash-001"]:
        resolved_model = "gemini-2.5-flash"
    elif resolved_model in ["gemini-3-pro-preview", "gemini-3.1-pro-preview"]:
        resolved_model = "gemini-2.5-pro"
    if original_model != resolved_model:
        logging.info(f"Context cache: downgraded '{original_model}' to stable '{resolved_model}'")

    try:
        contents = [types.Part.from_uri(file_uri=uri) for uri in file_uris]
        cache = client.caches.create(
            model=resolved_model,
            config=types.CreateCachedContentConfig(
                display_name=display_name or f"Cache {datetime.datetime.now().isoformat()}",
                contents=contents,
                ttl=f"{ttl_seconds}s",
            ),
        )
        return f"Cache created: {cache.name} (expires in {ttl_seconds}s)"
    except Exception as e:
        return f"Error creating cache: {e}"


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True, openWorldHint=True))
def delete_context_cache(
    cache_name: Annotated[str, Field(description="Cache name to delete (from create_context_cache)")],
) -> str:
    """Delete a Gemini context cache."""
    client = get_client()
    try:
        client.caches.delete(name=cache_name)
        return f"Cache '{cache_name}' deleted."
    except Exception as e:
        return f"Error deleting cache: {e}"


@mcp.tool(annotations=ToolAnnotations(openWorldHint=True))
def create_file_store(
    display_name: Annotated[str, Field(description="Display name for the store")],
) -> str:
    """Create a new FileSearch store for semantic code search.

    Upload files to the store with upload_to_file_store. Once populated,
    Gemini will automatically search the store during consult_gemini calls."""
    client = get_client()
    try:
        store = client.file_search_stores.create(
            config=types.CreateFileSearchStoreConfig(display_name=display_name)
        )
        with _stores_lock:
            _active_stores.add(store.name)
        return f"Store created: {store.name} (display_name: {store.display_name})"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def list_file_stores() -> str:
    """List all FileSearch stores with document counts and sizes."""
    client = get_client()
    try:
        stores = list(client.file_search_stores.list())
        if not stores:
            return "No file search stores found."
        lines = []
        for s in stores:
            lines.append(
                f"- {s.name} ({s.display_name}): "
                f"{s.active_documents_count or 0} active, "
                f"{s.pending_documents_count or 0} pending, "
                f"{s.size_bytes or 0} bytes"
            )
        # Refresh active stores tracking
        with _stores_lock:
            _active_stores.clear()
            _active_stores.update(s.name for s in stores)
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True))
def delete_file_store(
    name: Annotated[str, Field(description="Store resource name (e.g. 'fileSearchStores/xxx')")],
) -> str:
    """Delete a FileSearch store and all its documents."""
    client = get_client()
    try:
        client.file_search_stores.delete(name=name)
        with _stores_lock:
            _active_stores.discard(name)
        return f"Store '{name}' deleted."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(openWorldHint=True))
def upload_to_file_store(
    store_name: Annotated[str, Field(description="Store resource name (from create_file_store)")],
    file_path: Annotated[str, Field(description="Path to the local file to upload")],
    display_name: Annotated[str | None, Field(default=None, description="Display name for the file in the store")] = None,
) -> str:
    """Upload a file to a FileSearch store for semantic search.

    Supported: text, code, PDF, and other document formats.
    Files are chunked and embedded by Google for retrieval."""
    err = _validate_input_path(file_path)
    if err:
        return err
    client = get_client()
    path = Path(file_path)
    if not path.exists():
        return f"Error: File '{file_path}' does not exist."
    try:
        config = None
        if display_name:
            config = types.UploadToFileSearchStoreConfig(display_name=display_name)
        result = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=str(path),
            config=config,
        )
        op_name = getattr(result, "name", None)
        doc_name = getattr(getattr(result, "response", None), "document_name", None)
        if op_name and not doc_name:
            return f"Upload initiated for '{path.name}' to {store_name}. Operation: {op_name} (indexing in progress — search may not work immediately)"
        return f"Uploaded '{path.name}' to {store_name}: {doc_name or 'OK'}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True))
def list_models() -> str:
    """List available Gemini models grouped by capability.

    Queries the Gemini API for all accessible models and groups them
    by supported actions (generateContent, generateImages, embedContent, etc.).
    """
    client = get_client()
    try:
        groups: dict[str, list[str]] = {}
        for model in client.models.list():
            name = model.name or "unknown"
            for action in getattr(model, "supported_actions", None) or []:
                groups.setdefault(action, []).append(name)

        if not groups:
            return "No models found."

        lines = []
        for action in sorted(groups):
            lines.append(f"## {action}")
            for name in sorted(groups[action]):
                lines.append(f"  - {name}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Error listing models: {e}"


def _generate_image_imagen(
    client: genai.Client,
    model: str,
    prompt: str,
    aspect_ratio: str | None,
) -> bytes:
    """Generate an image via the Imagen generate_images API."""
    config_kwargs: dict[str, Any] = {"number_of_images": 1}
    if aspect_ratio:
        config_kwargs["aspect_ratio"] = aspect_ratio
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(**config_kwargs),
    )
    if response.generated_images:
        return response.generated_images[0].image.image_bytes
    
    # Check for filtered content or other reasons
    raise ValueError(f"Imagen returned no images. Response: {response}")


def _generate_image_nano_banana(
    client: genai.Client,
    model: str,
    prompt: str,
    aspect_ratio: str | None,
    image_size: str | None,
) -> bytes:
    """Generate an image via Gemini generate_content with IMAGE modality (Nano Banana)."""
    _sync_throttle(model)
    config_kwargs: dict[str, Any] = {
        "response_modalities": ["TEXT", "IMAGE"],
    }
    image_config_kwargs: dict[str, Any] = {}
    if aspect_ratio:
        image_config_kwargs["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config_kwargs["image_size"] = image_size
    if image_config_kwargs:
        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    # Record token usage for Nano Banana (they consume tokens unlike Imagen)
    usage = getattr(response, "usage_metadata", None)
    total = getattr(usage, "total_token_count", 0) or 0
    if total > 0:
        record_tokens(model, total)

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return part.inline_data.data

        # If we got candidates but no image part
        finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")
        if hasattr(finish_reason, "name"):
            finish_reason = finish_reason.name
        raise ValueError(f"Nano Banana returned no image data. Finish reason: {finish_reason}")

    raise ValueError("Nano Banana returned no candidates (possible safety filter)")


NANO_BANANA_MODELS = {MODEL_IMAGE_PRO, MODEL_IMAGE_FLASH, "nano-banana-pro-preview"}


@mcp.tool(annotations=ToolAnnotations(openWorldHint=True))
def generate_image(
    prompt: Annotated[str, Field(description="Text description of the image to generate")],
    output_path: Annotated[str, Field(description="File path to save the generated image")],
    model: Annotated[str, Field(default="imagen", description='"imagen" (default, ultra), "imagen-fast", "nano-pro", or "nano-flash"')] = "imagen",
    aspect_ratio: Annotated[str | None, Field(default=None, description='Aspect ratio (e.g. "1:1", "16:9", "9:16", "4:3", "3:4")')] = None,
    image_size: Annotated[str | None, Field(default=None, description='Output size for Nano Banana only (e.g. "1024x1024"). Not supported by Imagen')] = None,
) -> str:
    """Generates an image using Imagen or Nano Banana (Gemini image) models.

    Args:
        prompt: Text description of the image to generate.
        output_path: File path to save the generated image.
        model: Model alias or ID. "imagen" (default, ultra), "imagen-fast", "nano-pro", or "nano-flash".
        aspect_ratio: Aspect ratio (e.g. "1:1", "16:9", "9:16", "4:3", "3:4").
        image_size: Output size for Nano Banana only (e.g. "1024x1024"). Not supported by Imagen.
    """
    err = _validate_output_path(output_path)
    if err:
        return err
    client = get_client()
    resolved = MODEL_ALIASES.get(model.lower(), model)
    try:
        if resolved in NANO_BANANA_MODELS:
            image_bytes = _generate_image_nano_banana(client, resolved, prompt, aspect_ratio, image_size)
        else:
            if image_size:
                return f"Error: image_size is only supported by Nano Banana models (nano-pro, nano-flash), not {resolved}"
            image_bytes = _generate_image_imagen(client, resolved, prompt, aspect_ratio)

        Path(output_path).write_bytes(image_bytes)
        return f"Image generated ({resolved}) and saved to {output_path}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(annotations=ToolAnnotations(openWorldHint=True))
def generate_speech(
    text: Annotated[str, Field(description="Text to synthesize into speech")],
    output_path: Annotated[str, Field(description="File path to save the generated audio (.wav)")],
    voice_name: Annotated[str, Field(default="Puck", description="Voice name (e.g. Puck, Charon, Kore, Fenrir, Aoede)")] = "Puck",
    model: Annotated[str, Field(default="speech", description='"speech" (default, Pro quality) or "speech-fast"')] = "speech",
) -> str:
    """Synthesizes speech from text."""
    err = _validate_output_path(output_path)
    if err:
        return err
    if not output_path.lower().endswith(".wav"):
        return "Error: output_path must end with .wav (Gemini returns raw PCM audio which we format as WAV)."
    resolved = MODEL_ALIASES.get(model.lower(), model)
    _sync_throttle(resolved)
    client = get_client()
    try:
        response = client.models.generate_content(
            model=resolved,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )
        # Record token usage
        usage = getattr(response, "usage_metadata", None)
        total = getattr(usage, "total_token_count", 0) or 0
        if total > 0:
            record_tokens(resolved, total)

        audio_bytes = b""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.inline_data and "audio" in part.inline_data.mime_type:
                    audio_bytes += part.inline_data.data

        if audio_bytes:
            # Gemini TTS returns raw PCM (16-bit, 24kHz, mono). 
            # If saving as .wav, we must add the header.
            if output_path.lower().endswith(".wav") and not audio_bytes.startswith(b"RIFF"):
                with io.BytesIO() as wav_io:
                    with wave.open(wav_io, "wb") as wav_file:
                        wav_file.setnchannels(1)      # Mono
                        wav_file.setsampwidth(2)     # 16-bit
                        wav_file.setframerate(24000) # 24kHz
                        wav_file.writeframes(audio_bytes)
                    audio_bytes = wav_io.getvalue()

            Path(output_path).write_bytes(audio_bytes)
            return f"Speech generated and saved to {output_path}"
        
        # Handle cases where no audio was generated (e.g. safety block)
        finish_reason = "UNKNOWN"
        if response.candidates:
            finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")
            if hasattr(finish_reason, "name"):
                finish_reason = finish_reason.name
        
        return f"Error: No audio content generated. Finish reason: {finish_reason}"
    except Exception as e:
        return f"Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Batch
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(timeout=120, annotations=ToolAnnotations(openWorldHint=True))
async def create_batch(
    queries: list[dict[str, str]],
    model: str = "flash",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Submit a batch of queries for async processing (~50% cost discount).

    Batches run asynchronously (up to 24h). No tool use — inline all relevant
    context in each prompt. Use get_batch/list_batches to check status and
    get_batch_results when the job completes.

    The system prompt follows the same configuration as consult_gemini
    (config.toml, --system-prompt CLI flags).

    Args:
        queries: List of {custom_id: str, prompt: str} dicts.
        model: Model alias or ID (default: "flash"). "pro" enables deep thinking.
        temperature: Sampling temperature (default: 0.2).
    """
    try:
        client = get_client()
        model_id = MODEL_ALIASES.get(model.lower(), model)
        system = _compose_instruction(_SYSTEM_AGENT)

        requests = []
        for item in queries:
            custom_id = item.get("custom_id", "")
            prompt = item.get("prompt", "")
            if not custom_id or not prompt:
                return {"error": "Each query must have 'custom_id' and 'prompt' fields."}
            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "system_instruction": system,
            }
            if model_id == MODEL_PRO:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="HIGH")
            requests.append(types.InlinedRequest(
                contents=prompt,
                metadata={"custom_id": custom_id},
                config=types.GenerateContentConfig(**config_kwargs),
            ))

        job = await client.aio.batches.create(
            model=model_id,
            src=types.BatchJobSource(inlined_requests=requests),
        )
        return {
            "name": job.name,
            "state": job.state.value if job.state else None,
            "request_count": len(requests),
            "created_time": str(job.create_time) if job.create_time else None,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(timeout=30, annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_batch(name: str) -> dict[str, Any]:
    """Get the status of a batch job.

    Args:
        name: Batch job name (e.g. "batches/abc") from create_batch.
    """
    try:
        client = get_client()
        job = await client.aio.batches.get(name=name)
        result: dict[str, Any] = {
            "name": job.name,
            "state": job.state.value if job.state else None,
        }
        if job.completion_stats:
            result["completion_stats"] = {
                "successful": job.completion_stats.successful_count,
                "failed": job.completion_stats.failed_count,
                "incomplete": job.completion_stats.incomplete_count,
            }
        if job.create_time:
            result["created_time"] = str(job.create_time)
        if job.end_time:
            result["end_time"] = str(job.end_time)
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(timeout=30, annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def list_batches(limit: int = 20) -> dict[str, Any]:
    """List recent batch jobs.

    Args:
        limit: Maximum number of batches to return (default: 20, max: 100).
    """
    try:
        limit = max(1, min(limit, 100))
        client = get_client()
        batch_list = []
        count = 0
        async for job in await client.aio.batches.list():
            if count >= limit:
                break
            entry: dict[str, Any] = {
                "name": job.name,
                "state": job.state.value if job.state else None,
            }
            if job.completion_stats:
                entry["completion_stats"] = {
                    "successful": job.completion_stats.successful_count,
                    "failed": job.completion_stats.failed_count,
                }
            if job.create_time:
                entry["created_time"] = str(job.create_time)
            batch_list.append(entry)
            count += 1
        return {"count": len(batch_list), "batches": batch_list}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(timeout=60, annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def get_batch_results(name: str) -> dict[str, Any]:
    """Get results from a completed batch job.

    Only available when state is JOB_STATE_SUCCEEDED or JOB_STATE_PARTIALLY_SUCCEEDED.
    Results include text extracted from each response, keyed by custom_id.

    Args:
        name: Batch job name from create_batch.
    """
    try:
        client = get_client()
        job = await client.aio.batches.get(name=name)
        state = job.state.value if job.state else None
        # SDK's JOB_STATES_SUCCEEDED omits PARTIALLY_SUCCEEDED — add it explicitly
        valid_states = list(types.JOB_STATES_SUCCEEDED) + ["JOB_STATE_PARTIALLY_SUCCEEDED"]
        if state not in valid_states:
            return {"error": f"Batch not complete (state: {state}). Use get_batch to check status."}

        results = []
        # job.dest can be None if the job failed early before producing any output
        inlined_responses = (job.dest.inlined_responses or []) if job.dest else []
        for resp in inlined_responses:
            custom_id = (resp.metadata or {}).get("custom_id", "")
            item: dict[str, Any] = {"custom_id": custom_id}
            if resp.error:
                item["status"] = "error"
                item["error"] = str(resp.error)
            elif resp.response:
                item["status"] = "succeeded"
                item["text"] = resp.response.text or ""
                usage = resp.response.usage_metadata
                if usage:
                    item["usage"] = {
                        "prompt_tokens": usage.prompt_token_count,
                        "completion_tokens": usage.candidates_token_count,
                        "total_tokens": usage.total_token_count,
                    }
            results.append(item)
        return {"count": len(results), "results": results}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(timeout=30, annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True))
async def cancel_batch(name: str) -> dict[str, Any]:
    """Cancel a running batch job. Already-completed requests are unaffected.

    Args:
        name: Batch job name from create_batch.
    """
    try:
        client = get_client()
        await client.aio.batches.cancel(name=name)  # returns None
        job = await client.aio.batches.get(name=name)
        return {
            "name": job.name,
            "state": job.state.value if job.state else None,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(timeout=30, annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True, openWorldHint=True))
async def delete_batch(name: str) -> dict[str, Any]:
    """Delete a batch job. Only works on ended (succeeded/failed/cancelled) jobs.

    Args:
        name: Batch job name from create_batch.
    """
    try:
        client = get_client()
        job = await client.aio.batches.get(name=name)
        state = job.state.value if job.state else None
        # SDK's JOB_STATES_ENDED omits PARTIALLY_SUCCEEDED — add it explicitly
        ended_states = list(types.JOB_STATES_ENDED) + ["JOB_STATE_PARTIALLY_SUCCEEDED"]
        if state not in ended_states:
            return {"error": f"Cannot delete batch in state {state}. Cancel it first."}
        await client.aio.batches.delete(name=name)
        return {"deleted": name}
    except Exception as e:
        return {"error": str(e)}


def setup_otel(endpoint: str | None = None) -> None:
    """Configure OpenTelemetry for OTLP export using standard variables."""
    # Priority: 1. CLI Arg, 2. GPAL specific ENV, 3. Standard OTel ENV
    if not endpoint:
        endpoint = os.getenv("GPAL_OTEL_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    
    if not endpoint:
        return

    # Standard OTel service name
    service_name = os.getenv("OTEL_SERVICE_NAME", "gpal")

    logging.info(f"Configuring OpenTelemetry OTLP export for '{service_name}' to {endpoint}")
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    # Insecure for local/internal development as requested
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    # Enable W3C Trace Context propagation
    propagate.set_global_textmap(TraceContextTextMapPropagator())

def main() -> None:
    global _cli_key_file, _user_instruction, _user_instruction_sources
    parser = argparse.ArgumentParser(description="gpal - Your Pal Gemini")
    parser.add_argument("--otel-endpoint", help="OTLP gRPC endpoint (e.g., localhost:4317)")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Path to file containing the Gemini API key",
    )
    parser.add_argument(
        "--system-prompt",
        action="append",
        default=[],
        metavar="FILE",
        help="Additional system prompt file (repeatable, appended after config)",
    )
    parser.add_argument(
        "--no-default-prompt",
        action="store_true",
        help="Exclude the built-in default system instruction",
    )
    args, _ = parser.parse_known_args()

    if args.api_key_file:
        _cli_key_file = args.api_key_file

    # Load config and compose user instruction layers
    config = _load_config()
    _user_instruction, _user_instruction_sources = _build_system_instruction(
        config,
        cli_prompt_files=args.system_prompt,
        no_default=args.no_default_prompt,
    )
    logger.info("System instruction sources: %s", _user_instruction_sources)

    setup_otel(args.otel_endpoint)
    mcp.run()

if __name__ == "__main__":
    main()
