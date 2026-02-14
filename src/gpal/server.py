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
import threading
import wave
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

import time

# OpenTelemetry imports
from opentelemetry import trace, propagate
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from fastmcp.tools.tool import ToolResult

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

SYSTEM_INSTRUCTION = """
You are a consultant AI accessed via the Model Context Protocol (MCP).
Your role is to provide high-agency, deep reasoning and analysis on tasks,
usually in git repositories.

You have tools:
- list_directory, read_file, search_project — explore the local codebase
- gemini_search — search the web via Google Search
- semantic_search — find code by meaning using vector embeddings

Use them proactively to explore and verify—don't guess when you can look it up.

You have a massive context window (2M tokens). Read entire files or multiple
modules when needed to gather complete context.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Server & State
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("gpal")
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
            tools=[list_directory, read_file, search_project],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            system_instruction=SYSTEM_INSTRUCTION,
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
    client = get_client()

    # Clamp num_results
    num_results = max(1, min(num_results, MAX_SEARCH_RESULTS))

    # Single stateless API call with Google Search tool
    response = client.models.generate_content(
        model=model,
        contents=f"Search for: {query}\n\nProvide the top {num_results} most relevant results with titles, URLs, and summaries.",
        config=types.GenerateContentConfig(
            tools=[Tool(google_search=GoogleSearch())],
            temperature=0.1,
        ),
    )

    # Extract text response
    if response.candidates and response.candidates[0].content.parts:
        result_text = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text"):
                result_text += part.text
        return result_text.strip() or "No results returned."

    return "No results returned."


@GEMINI_RETRY_DECORATOR
def _gemini_code_exec(
    code: str,
    model: str = MODEL_CODE_EXEC,
) -> str:
    """Internal implementation of code execution with automatic retry."""
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

        def _wrap(text: str) -> ToolResult:
            resolved = MODEL_ALIASES.get(model_alias.lower(), model_alias)
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            return ToolResult(
                content=text,
                structured_content={"result": text, "model": resolved},
                meta={"model": resolved, "session_id": session_id, "duration_ms": elapsed_ms},
            )

        # Build generation config
        config_kwargs = {
            "temperature": 0.2,
            "tools": [list_directory, read_file, search_project, gemini_search, semantic_search],
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=RESPONSE_MAX_TOOL_CALLS,
            ),
            "system_instruction": SYSTEM_INSTRUCTION,
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
                size = await loop.run_in_executor(None, lambda p=p: p.stat().st_size)
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
                size = await loop.run_in_executor(None, lambda p=p: p.stat().st_size)
                if size > MAX_INLINE_MEDIA:
                    return f"Error: '{path}' exceeds {MAX_INLINE_MEDIA // (1024*1024)}MB inline limit."
                mime_type = detect_mime_type(path)
                if not mime_type:
                    return f"Error: Unknown media type for '{path}'."
                data = await loop.run_in_executor(None, lambda p=p: p.read_bytes())
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

        # Hold lock during send to prevent concurrent access to same session
        replacement = None
        with lock:
            try:
                func = partial(_send_with_retry, session, parts, gen_config)
                text = await loop.run_in_executor(None, func)
                return _wrap(text)
            except (ServiceUnavailable, ResourceExhausted, InternalServerError) as e:
                return f"Error: Service temporarily unavailable after retries: {e}"
            except Exception as e:
                error_msg = str(e).lower()
                if "client" not in error_msg or ("closed" not in error_msg and "shutdown" not in error_msg):
                    return f"Error: {e}"

                logging.warning(f"Session '{session_id}' has stale client, recreating...")
                prev_history = getattr(session, "_curated_history", getattr(session, "history", []))
                prev_model = getattr(session, "_gpal_model", model_alias)

                try:
                    new_client = get_client()
                    target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)
                    replacement = create_chat(new_client, target_model, history=prev_history, config=gen_config)
                    replacement._gpal_model = prev_model
                except Exception as retry_e:
                    return f"Error after retry: {retry_e}"

        if replacement:
            with sessions_lock:
                sessions[session_id] = (replacement, lock)
            with lock:
                try:
                    func = partial(_send_with_retry, replacement, parts, gen_config)
                    text = await loop.run_in_executor(None, func)
                    return _wrap(text)
                except Exception as retry_e:
                    return f"Error after retry: {retry_e}"

@GEMINI_RETRY_DECORATOR
def _send_with_retry(session: Any, parts: list[types.Part], config: types.GenerateContentConfig) -> str:
    """Send message with automatic retry on transient errors."""
    response = session.send_message(parts, config=config)
    try:
        if response.text:
            return response.text
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

    return f"System: No text generated. Finish Reason: {finish_reason}. Partial content: {', '.join(details)}"


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


@mcp.tool(timeout=60)
async def consult_gemini_flash(
    query: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
    cached_content: str | None = None,
    ctx: Context | None = None,
) -> str | ToolResult:
    """
    Consults Gemini 3 Flash (Fast/Efficient) for codebase exploration.

    Use Flash only for quick exploration tasks (listing files, searching, navigating)
    or when the user explicitly requests it. For code review, analysis, synthesis,
    and all other tasks, use consult_gemini_pro instead — Pro is the default.

    Gemini has autonomous access to: list_directory, read_file, search_project.
    """
    if ctx:
        await ctx.debug(f"Flash query: session={ctx.session_id}, files={len(file_paths or [])}")

    return await _consult(
        query, ctx, "flash", file_paths,
        media_paths, file_uris, json_mode, response_schema, cached_content
    )


@mcp.tool(timeout=600)
async def consult_gemini_pro(
    query: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
    cached_content: str | None = None,
    ctx: Context | None = None,
) -> str | ToolResult:
    """
    Consults Gemini 3 Pro (Reasoning/Deep) for deep codebase analysis.

    Gemini has autonomous access to: list_directory, read_file, search_project.
    """
    if ctx:
        await ctx.debug(f"Pro query: session={ctx.session_id}, files={len(file_paths or [])}")

    return await _consult(
        query, ctx, "pro", file_paths,
        media_paths, file_uris, json_mode, response_schema, cached_content
    )


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
    global _cli_key_file
    parser = argparse.ArgumentParser(description="gpal - Your Pal Gemini")
    parser.add_argument("--otel-endpoint", help="OTLP gRPC endpoint (e.g., localhost:4317)")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Path to file containing the Gemini API key",
    )
    args, _ = parser.parse_known_args()

    if args.api_key_file:
        _cli_key_file = args.api_key_file

    setup_otel(args.otel_endpoint)
    mcp.run()

if __name__ == "__main__":
    main()
