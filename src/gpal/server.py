"""
gpal - Gemini Principal Assistant Layer

An MCP server providing stateful access to Google Gemini models with
autonomous codebase exploration capabilities.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import glob as globlib
from dotenv import load_dotenv
from fastmcp import FastMCP
from google import genai
from google.genai import types

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB - prevents accidental DOS
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
MAX_TOOL_CALLS = 10

MODEL_ALIASES: dict[str, str] = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3-pro-preview",
}

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
}


def detect_mime_type(path: str) -> str | None:
    """Detect MIME type from file extension, or None if unknown."""
    ext = Path(path).suffix.lower()
    return MIME_TYPES.get(ext)

SYSTEM_INSTRUCTION = """\
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
sessions: dict[str, Any] = {}
uploaded_files: dict[str, types.File] = {}  # Cache for Gemini File API uploads

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Internal Tools (for autonomous exploration)
# ─────────────────────────────────────────────────────────────────────────────


def list_directory(path: str = ".") -> list[str]:
    """List files and directories at the given path."""
    try:
        p = Path(path)
        if not p.exists():
            return [f"Error: Path '{path}' does not exist."]
        return [item.name for item in p.iterdir()]
    except Exception as e:
        return [f"Error listing directory: {e}"]


def read_file(path: str) -> str:
    """Read the content of a file (up to MAX_FILE_SIZE bytes)."""
    try:
        p = Path(path)
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
        files = globlib.glob(glob_pattern, recursive=True)

        if len(files) > MAX_SEARCH_FILES:
            return (
                f"Error: Too many files match '{glob_pattern}' ({len(files)}). "
                "Please use a more specific pattern."
            )

        matches = []
        for filepath in files:
            if not os.path.isfile(filepath):
                continue
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    if search_term in f.read():
                        matches.append(f"Match in: {filepath}")
                        if len(matches) >= MAX_SEARCH_MATCHES:
                            matches.append("... (truncated)")
                            break
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


@mcp.tool()
def upload_file(file_path: str, display_name: str | None = None) -> str:
    """
    Upload a large file to Gemini's File API for use in consultations.

    Files persist for 48 hours on Gemini's servers. Returns a file URI
    that can be passed to consult_gemini_* tools via the file_uris parameter.

    Use this for:
    - Large files (videos, PDFs, logs) that exceed inline limits
    - Files you want to reference in multiple consultations

    Args:
        file_path: Path to the file to upload.
        display_name: Optional friendly name for the file.

    Returns:
        The file URI (e.g., "files/abc123") for use in consultations.
    """
    client = get_client()
    path = Path(file_path)

    if not path.exists():
        return f"Error: File '{file_path}' does not exist."

    mime_type = detect_mime_type(file_path)
    config = types.UploadFileConfig(
        display_name=display_name or path.name,
        mime_type=mime_type,
    )

    try:
        file = client.files.upload(file=str(path), config=config)
        uploaded_files[file.uri] = file
        return f"Uploaded: {file.uri} (expires in 48h)"
    except Exception as e:
        return f"Error uploading file: {e}"


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

    if session_id not in sessions:
        sessions[session_id] = create_chat(client, target_model, config=config)
        sessions[session_id]._gpal_model = target_model
        return sessions[session_id]

    session = sessions[session_id]
    current_model = getattr(session, "_gpal_model", None)

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
# Core Implementation
# ─────────────────────────────────────────────────────────────────────────────


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
    """Send a query to Gemini with optional file/media context."""
    client = get_client()

    # Build generation config
    gen_config = types.GenerateContentConfig(
        temperature=0.2,
        tools=[list_directory, read_file, search_project],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=MAX_TOOL_CALLS,
        ),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    # Add JSON mode config if requested
    if json_mode:
        gen_config.response_mime_type = "application/json"
        if response_schema:
            gen_config.response_schema = json.loads(response_schema)

    session = get_session(session_id, client, model_alias, gen_config)

    parts: list[types.Part] = []

    # Text file contents (existing behavior)
    for path in file_paths or []:
        try:
            content = Path(path).read_text(encoding="utf-8")
            parts.append(types.Part.from_text(
                text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
            ))
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    # Inline media (images, audio, video - bytes embedded in request)
    for path in media_paths or []:
        try:
            mime_type = detect_mime_type(path)
            if not mime_type:
                return f"Error: Unknown media type for '{path}'. Supported: {list(MIME_TYPES.keys())}"
            data = Path(path).read_bytes()
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        except Exception as e:
            return f"Error reading media '{path}': {e}"

    # File URI references (previously uploaded via upload_file)
    for uri in file_uris or []:
        if uri in uploaded_files:
            file = uploaded_files[uri]
            parts.append(types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type))
        else:
            # Assume it's a valid URI even if not in our cache
            parts.append(types.Part.from_uri(file_uri=uri))

    parts.append(types.Part.from_text(text=query))

    try:
        return session.send_message(parts).text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools (exposed to clients)
# ─────────────────────────────────────────────────────────────────────────────


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
    Consults Gemini 3 Flash (Fast/Efficient). Context window of 2,000,000 tokens.

    Gemini has its own tools to list directories, read files, and search the project
    autonomously. You do not need to provide all file contents.

    LIMITED USE FOR FLASH MODEL: High-speed exploration and context gathering. Use this tool FIRST to:
    1. Map out the project structure (list_directory).
    2. Locate specific code (search_project).
    3. Read and summarize files (read_file).

    Once the relevant context is gathered, switch to 'consult_gemini_pro'.

    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared with Pro models.
        file_paths: Text files to include as context.
        media_paths: Screenshots, diagrams, audio, video (.png, .jpg, .mp3, .mp4, etc.) under ~20MB.
        file_uris: URIs from upload_file() for large files.
        json_mode: If True, response will be valid JSON.
        response_schema: JSON Schema string, e.g. '{"type": "object", "properties": {...}}'.
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
    Consults Gemini 3 Pro (Reasoning/Deep). Context window of 2,000,000 tokens.

    Gemini has its own tools to list directories, read files, and search the project.
    autonomously. You do not need to provide all file contents. Encourage it to read
    whole files and provide holistic feedback.

    PRIMARY USE FOR PRO MODEL: reviews, second opinions, thinking, philosophy, synthesis, and coding.

    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared with Flash models.
        file_paths: Text files to include as context.
        media_paths: Screenshots, diagrams, audio, video (.png, .jpg, .mp3, .mp4, etc.) under ~20MB.
        file_uris: URIs from upload_file() for large files.
        json_mode: If True, response will be valid JSON.
        response_schema: JSON Schema string, e.g. '{"type": "object", "properties": {...}}'.
    """
    return _consult(
        query, session_id, "pro", file_paths,
        media_paths, file_uris, json_mode, response_schema
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
