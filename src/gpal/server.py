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
from functools import partial
from pathlib import Path
from typing import Any

import glob as globlib
from cachetools import TTLCache
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from google import genai
from google.api_core.exceptions import (
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.genai import types
from google.genai.types import GoogleSearch, Tool, ToolCodeExecution
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
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
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_PRO = "gemini-3-pro-preview"
MODEL_SEARCH = "gemini-flash-latest"          # Auto-updates to latest stable Flash
MODEL_CODE_EXEC = "gemini-flash-latest"
MODEL_IMAGE = "imagen-4.0-ultra-generate-001"
MODEL_IMAGE_FAST = "imagen-4.0-fast-generate-001"
MODEL_IMAGE_PRO = "nano-banana-pro-preview"        # Nano Banana Pro
MODEL_IMAGE_FLASH = "gemini-2.5-flash-image"       # Nano Banana Flash
MODEL_SPEECH = "gemini-2.5-pro-preview-tts"

MODEL_ALIASES: dict[str, str] = {
    "flash": MODEL_FLASH,
    "pro": MODEL_PRO,
    "imagen": MODEL_IMAGE,
    "imagen-fast": MODEL_IMAGE_FAST,
    "nano-pro": MODEL_IMAGE_PRO,
    "nano-flash": MODEL_IMAGE_FLASH,
    "speech": MODEL_SPEECH,
}

# Limits
MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB - prevents accidental DOS
MAX_INLINE_MEDIA = 20 * 1024 * 1024  # 20 MB - inline media limit
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
RESPONSE_MAX_TOOL_CALLS = 25
MAX_SEARCH_RESULTS = 10

# Retry configuration (tenacity handles all backoff)
GEMINI_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(
        (ServiceUnavailable, ResourceExhausted, InternalServerError)
    ),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
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
    MODEL_PRO: 1_000_000,
    MODEL_FLASH: 4_000_000,
    MODEL_SEARCH: 4_000_000,  # Resolves to a Flash model
    MODEL_SPEECH: 2_000_000,
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


def _sync_throttle(model: str) -> None:
    """Block if approaching rate limit. For sync code paths."""
    limit = RATE_LIMITS_TPM.get(model, 0)
    while limit > 0:
        current = tokens_in_window(model)
        if current <= limit * 0.9:
            break
        jitter = random.uniform(1.0, 5.0)
        logging.warning(f"Rate limit approaching for {model}: {current}/{limit} TPM, sleeping {jitter:.1f}s")
        time.sleep(jitter)


async def _async_throttle(model: str, ctx: Context | None = None) -> None:
    """Async version of rate limit throttle with re-check after sleep."""
    limit = RATE_LIMITS_TPM.get(model, 0)
    while limit > 0:
        current = tokens_in_window(model)
        if current <= limit * 0.9:
            break
        jitter = random.uniform(1.0, 5.0)
        logging.warning(f"Rate limit approaching for {model}: {current}/{limit} TPM, sleeping {jitter:.1f}s")
        if ctx:
            await ctx.warning(f"Rate limit approaching ({current}/{limit} TPM), throttling...")
        await asyncio.sleep(jitter)


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

DEFAULT_SYSTEM_INSTRUCTION = """
You are a consultant AI accessed via the Model Context Protocol (MCP).
Your role is to provide high-agency, deep reasoning and analysis on tasks,
usually in git repositories.

You have tools:
- list_directory, read_file, search_project — explore the local codebase
- git — read-only git operations (status, diff, log, show)
- gemini_search — search the web via Google Search
- semantic_search — find code by meaning using vector embeddings

Use them proactively to explore and verify—don't guess when you can look it up.

You have a massive context window (2M tokens). Read entire files or multiple
modules when needed to gather complete context.
"""

# Composed system instruction — set by main(), defaults to built-in
_system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
_system_instruction_sources: list[str] = ["built-in"]

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
    """Compose the system instruction from config, files, and CLI flags.

    Returns (instruction_text, list_of_sources) where sources describes
    provenance for debugging.
    """
    parts: list[str] = []
    sources: list[str] = []

    # 1. Built-in default (unless suppressed)
    include_default = config.get("include_default_prompt", True)
    if no_default:
        include_default = False

    if include_default:
        parts.append(DEFAULT_SYSTEM_INSTRUCTION.strip())
        sources.append("built-in")

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

    if not parts:
        # Fallback: if everything was suppressed and no files provided,
        # use the default anyway to avoid an empty instruction
        parts.append(DEFAULT_SYSTEM_INSTRUCTION.strip())
        sources.append("built-in (fallback)")

    return "\n\n".join(parts), sources


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
_indexes: dict = {}  # Cache for semantic search indexes
_indexes_lock = threading.Lock()  # Thread safety for _indexes


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
            "flash": MODEL_FLASH,
            "pro": MODEL_PRO,
            "search": MODEL_SEARCH,
            "code_exec": MODEL_CODE_EXEC,
            "image": MODEL_IMAGE,
            "image_fast": MODEL_IMAGE_FAST,
            "image_pro": MODEL_IMAGE_PRO,
            "image_flash": MODEL_IMAGE_FLASH,
            "speech": MODEL_SPEECH,
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
            "sources": _system_instruction_sources,
            "length_chars": len(_system_instruction),
        },
        "token_usage": token_stats(),
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
def get_session_detail(session_id: str) -> str:
    """View conversation history for a session."""
    with sessions_lock:
        item = sessions.get(session_id)
        if not item:
            return json.dumps({"error": f"Session '{session_id}' not found or expired"})
        session, lock = item

    # Hold per-session lock while reading history
    with lock:
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


@mcp.resource("gpal://index/stats")
def get_index_stats() -> str:
    """Semantic index statistics (files, chunks per indexed path)."""
    # Take snapshot under lock to avoid iteration issues
    with _indexes_lock:
        indexes_snapshot = list(_indexes.items())

    if not indexes_snapshot:
        return json.dumps({"message": "No indexes loaded. Use semantic_search or rebuild_index first."})

    stats = {}
    for path, index in indexes_snapshot:
        try:
            # Get collection stats
            code_count = index.collection.count()
            meta_count = index.meta_collection.count()
            stats[path] = {
                "chunks": code_count,
                "files_indexed": meta_count,
                "db_path": str(index.db_path),
            }
        except Exception as e:
            stats[path] = {"error": str(e)}

    return json.dumps(stats, indent=2)


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
        "flash": MODEL_FLASH,
        "pro": MODEL_PRO,
        "search": MODEL_SEARCH,
        "code_exec": MODEL_CODE_EXEC,
        "image": MODEL_IMAGE,
        "image_fast": MODEL_IMAGE_FAST,
        "image_pro": MODEL_IMAGE_PRO,
        "image_flash": MODEL_IMAGE_FLASH,
        "speech": MODEL_SPEECH,
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
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        
        if not p.is_relative_to(cwd):
            msg = f"Error: Access denied to '{path}' (outside project root)"
            logging.warning(msg)
            return msg
            
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
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        
        if not p.is_relative_to(cwd):
            return f"Error: Access denied to '{path}' (outside project root)."
            
        if not p.exists():
            return f"Error: File '{path}' does not exist."
            
        if p.stat().st_size > MAX_FILE_SIZE:
            return f"Error: File '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file '{path}': {e}"


def search_project(search_term: str, glob_pattern: str = "**/*") -> str:
    """
    Search for a text term in files matching the glob pattern.
    Returns a summary of matching files.
    """
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
                return (
                    f"Error: Too many files match '{glob_pattern}' (>{MAX_SEARCH_FILES}). "
                    "Use a more specific pattern, or try semantic_search for large codebases."
                )

            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
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


def get_index(root: str = "."):
    """Get or create a semantic search index for a project root."""
    from gpal.index import CodebaseIndex  # lazy import
    root_path = Path(root).resolve()
    key = str(root_path)
    
    with _indexes_lock:
        if key in _indexes:
            return _indexes[key]
    
    # Create outside global lock to avoid blocking other project lookups
    # Only one thread will win the assignment race below
    new_index = CodebaseIndex(root_path, get_client())
    
    with _indexes_lock:
        if key not in _indexes:
            _indexes[key] = new_index
        return _indexes[key]

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
            tools=[list_directory, read_file, search_project, git],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            system_instruction=_system_instruction,
        )

    return client.chats.create(
        model=model_name,
        history=history or [],
        config=config,
    )

async def get_session(
    ctx: Context,
    client: genai.Client,
    model_alias: str,
    config: types.GenerateContentConfig | None = None,
) -> tuple[Any, threading.Lock]:
    """Get or create a session, bundled with its own lock."""
    session_id = ctx.session_id
    target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)

    with sessions_lock:
        if session_id in sessions:
            session, lock = sessions[session_id]
        else:
            session = create_chat(client, target_model, config=config)
            session._gpal_model = target_model
            lock = threading.Lock()
            sessions[session_id] = (session, lock)
            ctx.set_state("model", target_model)
            return session, lock

    # Use per-session lock for migration
    with lock:
        current_model = getattr(session, "_gpal_model", None)
        if current_model == target_model:
            return session, lock

        logging.info(f"Migrating session '{session_id}': {current_model} → {target_model}")
        try:
            history = getattr(session, "_curated_history", getattr(session, "history", []))
            session = create_chat(client, target_model, history=history, config=config)
        except Exception as e:
            logging.error(f"History migration failed: {e}")
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
        config_kwargs = {
            "temperature": 0.2,
            "tools": [list_directory, read_file, search_project, gemini_search, semantic_search, git],
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            "system_instruction": _system_instruction,
        }

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
            try:
                p = Path(path)
                size = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.stat().st_size)
                if size > MAX_FILE_SIZE:
                    return f"Error: '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
                content = await loop.run_in_executor(
                    None, lambda p=p: p.read_text(encoding="utf-8")
                )
                parts.append(types.Part.from_text(
                    text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
                ))
            except Exception as e:
                return f"Error reading file '{path}': {e}"

        # Context: Inline media (offloaded to thread pool)
        for path in media_paths or []:
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
            with lock:
                try:
                    func = partial(_send_with_retry, session, parts, gen_config)
                    resp = await loop.run_in_executor(_EXECUTOR, func)
                    return _wrap(resp)
                except (ServiceUnavailable, ResourceExhausted, InternalServerError) as e:
                    return f"Error: Service temporarily unavailable after retries: {e}"
                except Exception as e:
                    error_msg = str(e).lower()
                    is_stale = "client" in error_msg and ("closed" in error_msg or "shutdown" in error_msg)
                    if not is_stale or attempt > 0:
                        return f"Error: {e}"

                    logging.warning(f"Session '{session_id}' has stale client, recreating...")
                    try:
                        prev_history = getattr(session, "_curated_history", getattr(session, "history", []))
                        new_client = get_client()
                        target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)
                        session = create_chat(new_client, target_model, history=prev_history, config=gen_config)
                        session._gpal_model = getattr(session, "_gpal_model", model_alias)
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


@mcp.tool()
def gemini_code_exec(
    code: str,
    model: str = MODEL_CODE_EXEC,
) -> str:
    """
    Execute Python code using Gemini's built-in code execution sandbox.

    Returns stdout, stderr, and any execution results.
    Stateless utility.
    """
    return _gemini_code_exec(code, model)


_FLASH_EXPLORE_PROMPT = """\
You are in exploration mode. Your goal is to map the codebase thoroughly for a \
follow-up synthesis phase. Map the territory — don't plan the journey.

TASK: {query}

STRATEGY:
- Start with `semantic_search` to find conceptual entry points.
- Use `list_directory` and `search_project` for specific definitions and patterns.
- If a file is relevant and under 1000 lines, read the whole thing — you have a \
massive context window; don't hunt for snippets when full context is cheap.
- Use reasoning to follow imports, call chains, and data flow — that's navigation, \
not analysis. Follow the thread wherever it leads.
- Use `git` to check recent changes, blame, or history when provenance matters.

DO NOT:
- Propose fixes, write code, or provide recommendations.
- Stop until you have a clear picture of the task's "surface area."

OUTPUT — Structured Inventory:
1. **Key Modules**: File paths with brief roles and important line numbers.
2. **Data Flow**: How data moves through the components you found.
3. **Patterns & Constants**: Architectural patterns, config values, shared conventions.
4. **Files Read in Full**: List every file you read completely (so the synthesis phase \
knows what's already in context).
5. **Blind Spots**: Files referenced but not found, unclear ownership, anything unresolved.
"""

_PRO_SYNTHESIS_PROMPT = """\
You are in the SYNTHESIS phase.

CONTEXT:
The conversation history above contains raw outputs from an exploration agent — \
`read_file`, `search_project`, and `list_directory` results. Treat these as your \
working memory. The exploration was thorough but may have gaps.

TASK: {query}

APPROACH:
1. **Gap Check**: Scan the exploration data. If a critical file was referenced but \
not read, use `read_file` to fetch it now. Do NOT restart broad exploration.
2. **Root Cause / Analysis**: Explain *why* the code is structured this way. \
Identify the specific logic relevant to the task.
3. **Solution Design**: Propose a concrete, step-by-step plan.
4. **Implementation**: Provide exact code changes with file paths. Include enough \
surrounding context (or rewrite the full function) so changes can be applied unambiguously.

GOAL: A complete, actionable response the user can apply immediately.
"""


@mcp.tool(timeout=660)
async def consult_gemini(
    query: str,
    model: str = "auto",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
    cached_content: str | None = None,
    ctx: Context | None = None,
) -> str | ToolResult:
    """Consults Gemini for codebase analysis.

    Args:
        query: The question or instruction.
        model: "auto" (default, Flash explore → Pro analyze), "flash", "pro", or full model ID.
        file_paths: Text files to include as context.
        media_paths: Images (.png, .jpg, .webp, .gif) or PDFs for vision analysis.
        file_uris: Gemini File API URIs (from upload_file).
        json_mode: Return structured JSON output.
        response_schema: JSON schema string for structured output.
        cached_content: Gemini context cache name.

    Gemini has autonomous access to: list_directory, read_file, search_project.
    """
    if ctx:
        await ctx.debug(f"consult_gemini: model={model}, session={ctx.session_id}, files={len(file_paths or [])}")

    if model == "auto":
        # Phase 1: Flash explores
        if ctx:
            await ctx.info("Phase 1: Flash exploration...")
        explore_query = _FLASH_EXPLORE_PROMPT.format(query=query)
        flash_result = await _consult(
            explore_query, ctx, "flash", file_paths,
            media_paths, file_uris, False, None, cached_content
        )

        # Extract text from ToolResult or string
        if isinstance(flash_result, ToolResult):
            flash_text = flash_result.content or ""
        else:
            flash_text = flash_result

        # Check for errors in flash phase
        if isinstance(flash_text, str) and flash_text.startswith("Error:"):
            return flash_result

        # Phase 2: Pro synthesizes (same session — history migrates automatically)
        if ctx:
            await ctx.info("Phase 2: Pro synthesis...")
        synthesis_query = _PRO_SYNTHESIS_PROMPT.format(query=query)
        return await _consult(
            synthesis_query, ctx, "pro", None,
            None, None, json_mode, response_schema, None
        )

    # Direct model pass-through
    alias = model if model in MODEL_ALIASES else model
    return await _consult(
        query, ctx, alias, file_paths,
        media_paths, file_uris, json_mode, response_schema, cached_content
    )


@mcp.tool(timeout=120)
async def consult_gemini_oneshot(
    query: str,
    model: str = "pro",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
    ctx: Context | None = None,
) -> str | ToolResult:
    """Stateless single-shot Gemini query with no session history.

    Use for independent questions, one-off lookups, or batch-style queries
    where conversation context would be noise. Still has tool access
    (list_directory, read_file, etc.) and retry logic.

    Args:
        query: The question or instruction.
        model: "flash", "pro" (default), or full model ID.
        file_paths: Text files to include as context.
        media_paths: Images for vision analysis.
        json_mode: Return structured JSON output.
        response_schema: JSON schema string for structured output.
    """
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
            try:
                p = Path(path)
                size = await loop.run_in_executor(_EXECUTOR, lambda p=p: p.stat().st_size)
                if size > MAX_FILE_SIZE:
                    return f"Error: '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
                content = await loop.run_in_executor(
                    None, lambda p=p: p.read_text(encoding="utf-8")
                )
                parts.append(types.Part.from_text(
                    text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
                ))
            except Exception as e:
                return f"Error reading file '{path}': {e}"

        for path in media_paths or []:
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

        parts.append(types.Part.from_text(text=query))

        # Build config
        config_kwargs: dict[str, Any] = {
            "temperature": 0.2,
            "tools": [list_directory, read_file, search_project, gemini_search, semantic_search, git],
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            "system_instruction": _system_instruction,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
            if response_schema:
                try:
                    config_kwargs["response_schema"] = json.loads(response_schema)
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON schema: {e}"

        gen_config = types.GenerateContentConfig(**config_kwargs)

        # Proactive rate limiting with re-check and jitter
        await _async_throttle(resolved_model)

        # Stateless call via chats.create with empty history
        try:
            session = create_chat(client, resolved_model, config=gen_config)
            func = partial(_send_with_retry, session, parts, gen_config)
            resp = await loop.run_in_executor(_EXECUTOR, func)

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
        except (ServiceUnavailable, ResourceExhausted, InternalServerError) as e:
            return f"Error: Service temporarily unavailable after retries: {e}"
        except Exception as e:
            return f"Error: {e}"


@mcp.tool()
def upload_file(file_path: str, display_name: str | None = None) -> str:
    """Upload a large file to Gemini's File API."""
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


@mcp.tool()
def create_context_cache(
    file_uris: list[str],
    model: str = "flash",
    display_name: str | None = None,
    ttl_seconds: int = 3600,
) -> str:
    """
    Create a Gemini context cache for a set of files.
    
    Caching is useful for large files (>32k tokens) used across multiple turns.
    Model must be an explicit version (e.g., gemini-1.5-flash-001) or a supported alias.
    """
    client = get_client()
    resolved_model = MODEL_ALIASES.get(model.lower(), model)
    
    # Caching requires stable versioned IDs, not short preview aliases
    if resolved_model in ["gemini-3-flash-preview", "gemini-2.0-flash-001"]:
        resolved_model = "gemini-1.5-flash-001"
    elif resolved_model in ["gemini-3-pro-preview"]:
        resolved_model = "gemini-1.5-pro-001"

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


@mcp.tool()
def delete_context_cache(cache_name: str) -> str:
    """Delete a Gemini context cache."""
    client = get_client()
    try:
        client.caches.delete(name=cache_name)
        return f"Cache '{cache_name}' deleted."
    except Exception as e:
        return f"Error deleting cache: {e}"


def semantic_search(query: str, limit: int = 5, path: str = ".") -> str:
    """Find code semantically related to the query using vector embeddings."""
    in_afc = getattr(_afc_local, "in_afc", False)
    if in_afc:
        _afc_api_semaphore.acquire()
    try:
        index = get_index(path)
        results = index.search(query, limit)
        if not results:
            return "No matches found. Try rebuilding the index."
        output = []
        for r in results:
            output.append(f"**{r['file']}:{r['lines']}** (score: {r['score']})\n```\n{r['snippet']}\n```\n")
        return "\n".join(output)
    except Exception as e:
        return f"Error: {e}"
    finally:
        if in_afc:
            _afc_api_semaphore.release()

mcp.tool(semantic_search)


@mcp.tool(timeout=300)
async def rebuild_index(path: str = ".", ctx: Context | None = None) -> str:
    """Rebuild the semantic search index for a codebase."""
    try:
        if ctx:
            await ctx.info(f"Rebuilding semantic index for {path}")

        index = get_index(path)

        # Create progress callback that uses MCP Context
        async def progress_callback(message: str, current: int = 0, total: int = 0) -> None:
            if ctx:
                await ctx.info(f"[Index] {message}")
                if total > 0:
                    await ctx.report_progress(progress=current, total=total)

        result = await index.rebuild_async(progress_callback=progress_callback)

        if ctx:
            await ctx.info(f"Index complete: {result.get('indexed', 0)} indexed, {result.get('skipped', 0)} skipped")

        return f"Index rebuilt: {result.get('indexed', 0)} indexed, {result.get('skipped', 0)} unchanged."
    except Exception as e:
        if ctx:
            await ctx.error(f"Index rebuild failed: {e}")
        return f"Error: {e}"


@mcp.tool()
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


NANO_BANANA_MODELS = {MODEL_IMAGE_PRO, MODEL_IMAGE_FLASH, "gemini-3-pro-image-preview"}


@mcp.tool()
def generate_image(
    prompt: str,
    output_path: str,
    model: str = "imagen",
    aspect_ratio: str | None = None,
    image_size: str | None = None,
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


@mcp.tool()
def generate_speech(text: str, output_path: str, voice_name: str = "Puck") -> str:
    """Synthesizes speech from text."""
    err = _validate_output_path(output_path)
    if err:
        return err
    _sync_throttle(MODEL_SPEECH)
    client = get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_SPEECH,
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
            record_tokens(MODEL_SPEECH, total)

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
    global _cli_key_file, _system_instruction, _system_instruction_sources
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

    # Load config and compose system instruction
    config = _load_config()
    _system_instruction, _system_instruction_sources = _build_system_instruction(
        config,
        cli_prompt_files=args.system_prompt,
        no_default=args.no_default_prompt,
    )
    logger.info("System instruction sources: %s", _system_instruction_sources)

    setup_otel(args.otel_endpoint)
    mcp.run()

if __name__ == "__main__":
    main()
