"""
gpal - Gemini Principal Assistant Layer

An MCP server providing stateful access to Google Gemini models with
autonomous codebase exploration capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import glob as globlib
from cachetools import TTLCache
from dotenv import load_dotenv
from fastmcp import FastMCP
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
MODEL_SEARCH = "gemini-2.0-flash"        # Search/code exec use stable 2.0
MODEL_CODE_EXEC = "gemini-2.0-flash"
MODEL_IMAGE = "imagen-4.0-generate-001"
MODEL_SPEECH = "gemini-2.5-flash-preview-tts"

MODEL_ALIASES: dict[str, str] = {
    "flash": MODEL_FLASH,
    "pro": MODEL_PRO,
}

# Limits
MAX_FILE_SIZE = 10 * 1024 * 1024    # 10 MB - prevents accidental DOS
MAX_INLINE_MEDIA = 20 * 1024 * 1024  # 20 MB - inline media limit
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
MAX_TOOL_CALLS = 10
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
}


def detect_mime_type(path: str) -> str | None:
    """Detect MIME type from file extension, or None if unknown."""
    ext = Path(path).suffix.lower()
    return MIME_TYPES.get(ext)

SYSTEM_INSTRUCTION = """
You are a consultant AI accessed via the Model Context Protocol (MCP).
Your role is to provide high-agency, deep reasoning and analysis on tasks,
usually in git repositories.

You have tools: list_directory, read_file, and search_project.
Use them proactively to explore the codebase—don't guess when you can verify.

You have a massive context window (2M tokens). Read entire files or multiple
modules when needed to gather complete context.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Server & State
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("gpal")
sessions: TTLCache = TTLCache(maxsize=100, ttl=3600)  # 1 hour TTL
sessions_lock = threading.Lock()
session_locks: dict[str, threading.Lock] = {}
uploaded_files: dict[str, types.File] = {}  # Cache for Gemini File API uploads
_indexes: dict = {}  # Cache for semantic search indexes

# ─────────────────────────────────────────────────────────────────────────────
# Codebase Exploration Tools (Local)
# ─────────────────────────────────────────────────────────────────────────────


def list_directory(path: str = ".") -> list[str]:
    """List files and directories at the given path."""
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        
        if not p.is_relative_to(cwd):
            return [f"Error: Access denied to '{path}' (outside project root)."]
            
        if not p.exists():
            return [f"Error: Path '{path}' does not exist."]
        return [item.name for item in p.iterdir()]
    except Exception as e:
        return [f"Error listing directory: {e}"]


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

        # Basic validation to prevent obvious traversal
        if ".." in glob_pattern:
            return "Error: Glob pattern cannot contain '..'"

        # Use iglob iterator to avoid loading huge file lists into memory
        matches = []
        files_checked = 0

        for filepath in globlib.iglob(glob_pattern, recursive=True):
            files_checked += 1
            if files_checked > MAX_SEARCH_FILES:
                return (
                    f"Error: Too many files match '{glob_pattern}' (>{MAX_SEARCH_FILES}). "
                    "Use a more specific pattern, or try semantic_search for large codebases."
                )

            path_obj = Path(filepath).resolve()

            # Ensure file is within project root
            if not path_obj.is_relative_to(cwd) or not path_obj.is_file():
                continue

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


def get_client() -> genai.Client:
    """Create a Gemini API client from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set.")
    return genai.Client(api_key=api_key)


def get_index(root: str = "."):
    """Get or create a semantic search index for a project root."""
    from gpal.index import CodebaseIndex  # lazy import
    root_path = Path(root).resolve()
    key = str(root_path)
    if key not in _indexes:
        _indexes[key] = CodebaseIndex(root_path, get_client())
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
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
            system_instruction=SYSTEM_INSTRUCTION,
        )

    return client.chats.create(
        model=model_name,
        history=history or [],
        config=config,
    )

def get_session(
    session_id: str,
    client: genai.Client,
    model_alias: str,
    config: types.GenerateContentConfig | None = None,
) -> Any:
    """Get or create a session, migrating history when switching models."""
    target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)

    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = create_chat(client, target_model, config=config)
            sessions[session_id]._gpal_model = target_model
            session_locks[session_id] = threading.Lock()
            return sessions[session_id]
        
        # Ensure lock exists for existing session
        if session_id not in session_locks:
            session_locks[session_id] = threading.Lock()

    # Use per-session lock for migration checks and updates
    with session_locks[session_id]:
        session = sessions[session_id]
        current_model = getattr(session, "_gpal_model", None)

        # Update config if provided (handles json_mode/schema switches)
        if config:
            # Note: This relies on the SDK exposing a way to update config, or we recreate the chat
            # Since chat objects are stateful, we might need to recreate if config is fundamentally different
            # For now, we'll recreate if JSON mode requirements change, or just set it on the next message
            # The SDK's send_message accepts a config override, which we use in _consult.
            # So the session object itself doesn't strictly need the new config permanently, 
            # as long as we pass it to send_message.
            pass

        if current_model == target_model:
            return session

        # Migrate history to new model
        logging.info(f"Migrating session '{session_id}': {current_model} → {target_model}")
        try:
            # Try _curated_history first (google-genai internal), fall back to .history
            history = getattr(session, "_curated_history", getattr(session, "history", []))
            sessions[session_id] = create_chat(client, target_model, history=history, config=config)
        except Exception as e:
            logging.error(f"History migration failed: {e}")
            sessions[session_id] = create_chat(client, target_model, config=config)

        sessions[session_id]._gpal_model = target_model
        return sessions[session_id]


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


def _consult(
    query: str,
    session_id: str,
    model_alias: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
) -> str:
    """Send a query to Gemini with codebase context."""
    client = get_client()

    # Build generation config
    config_kwargs = {
        "temperature": 0.2,
        "tools": [list_directory, read_file, search_project],
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=MAX_TOOL_CALLS,
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

    gen_config = types.GenerateContentConfig(**config_kwargs)

    session = get_session(session_id, client, model_alias, gen_config)

    parts: list[types.Part] = []

    # Context: Text files
    for path in file_paths or []:
        try:
            content = Path(path).read_text(encoding="utf-8")
            parts.append(types.Part.from_text(
                text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
            ))
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    # Context: Inline media
    for path in media_paths or []:
        try:
            p = Path(path)
            if p.stat().st_size > MAX_INLINE_MEDIA:
                return f"Error: '{path}' exceeds {MAX_INLINE_MEDIA // (1024*1024)}MB inline limit."
            mime_type = detect_mime_type(path)
            if not mime_type:
                return f"Error: Unknown media type for '{path}'."
            parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime_type))
        except Exception as e:
            return f"Error reading media '{path}': {e}"

    # Context: File URIs
    for uri in file_uris or []:
        parts.append(types.Part.from_uri(file_uri=uri))

    parts.append(types.Part.from_text(text=query))

    # Get the per-session lock for thread-safe send_message
    with sessions_lock:
        if session_id not in session_locks:
            session_locks[session_id] = threading.Lock()
        lock = session_locks[session_id]

    # Hold lock during send to prevent concurrent access to same session
    with lock:
        try:
            return _send_with_retry(session, parts, gen_config)
        except (ServiceUnavailable, ResourceExhausted, InternalServerError) as e:
            return f"Error: Service temporarily unavailable after retries: {e}"
        except Exception as e:
            return f"Error: {e}"


@GEMINI_RETRY_DECORATOR
def _send_with_retry(session: Any, parts: list[types.Part], config: types.GenerateContentConfig) -> str:
    """Send message with automatic retry on transient errors."""
    return session.send_message(parts, config=config).text


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools (Exposed)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
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


@mcp.tool()
def consult_gemini_flash(
    query: str,
    session_id: str = "default",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
) -> str:
    """
    Consults Gemini 3 Flash (Fast/Efficient) for codebase exploration.
    
    Gemini has autonomous access to: list_directory, read_file, search_project.
    """
    return _consult(
        query, session_id, "flash", file_paths,
        media_paths, file_uris, json_mode, response_schema
    )


@mcp.tool()
def consult_gemini_pro(
    query: str,
    session_id: str = "default",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    file_uris: list[str] | None = None,
    json_mode: bool = False,
    response_schema: str | None = None,
) -> str:
    """
    Consults Gemini 3 Pro (Reasoning/Deep) for deep codebase analysis.
    
    Gemini has autonomous access to: list_directory, read_file, search_project.
    """
    return _consult(
        query, session_id, "pro", file_paths,
        media_paths, file_uris, json_mode, response_schema
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
        uploaded_files[file.uri] = file
        return f"Uploaded: {file.uri}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
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


@mcp.tool()
def rebuild_index(path: str = ".") -> str:
    """Rebuild the semantic search index for a codebase."""
    try:
        index = get_index(path)
        result = index.rebuild()
        return f"Index rebuilt: {result.get('indexed', 0)} indexed, {result.get('skipped', 0)} unchanged."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def generate_image(prompt: str, output_path: str) -> str:
    """Generates an image using Imagen models."""
    client = get_client()
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        response = client.models.generate_images(
            model=MODEL_IMAGE,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        if response.generated_images:
            Path(output_path).write_bytes(response.generated_images[0].image.image_bytes)
            return f"Image generated and saved to {output_path}"
        return "No image generated."
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def generate_speech(text: str, output_path: str, voice_name: str = "Puck") -> str:
    """Synthesizes speech from text."""
    client = get_client()
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
            Path(output_path).write_bytes(audio_bytes)
            return f"Speech generated and saved to {output_path}"
        return "No audio content generated."
    except Exception as e:
        return f"Error: {e}"

def main() -> None:
    mcp.run()

if __name__ == "__main__":
    main()
