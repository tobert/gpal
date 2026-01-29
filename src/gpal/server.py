"""
gpal - Gemini Principal Assistant Layer

An MCP server providing stateful access to Google Gemini models with
autonomous codebase exploration capabilities.
"""

from __future__ import annotations

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


def create_chat(
    client: genai.Client,
    model_name: str,
    history: list[Any] | None = None,
) -> Any:
    """Create a configured Gemini chat session."""
    return client.chats.create(
        model=model_name,
        history=history or [],
        config=types.GenerateContentConfig(
            temperature=0.2,
            tools=[list_directory, read_file, search_project],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=MAX_TOOL_CALLS,
            ),
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )


def get_session(session_id: str, client: genai.Client, model_alias: str) -> Any:
    """Get or create a session, migrating history when switching models."""
    target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)

    if session_id not in sessions:
        sessions[session_id] = create_chat(client, target_model)
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
        sessions[session_id] = create_chat(client, target_model, history=history)
    except Exception as e:
        logging.error(f"History migration failed: {e}")
        sessions[session_id] = create_chat(client, target_model)

    sessions[session_id]._gpal_model = target_model
    return sessions[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# Core Implementation
# ─────────────────────────────────────────────────────────────────────────────


def _consult(
    query: str,
    session_id: str,
    model_alias: str,
    file_paths: list[str] | None,
) -> str:
    """Send a query to Gemini with optional file context."""
    client = get_client()
    session = get_session(session_id, client, model_alias)

    parts: list[types.Part] = []

    for path in file_paths or []:
        try:
            content = Path(path).read_text(encoding="utf-8")
            parts.append(types.Part.from_text(
                text=f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---\n"
            ))
        except Exception as e:
            return f"Error reading file '{path}': {e}"

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
        file_paths: (Optional) Essential files to get started with exploring.
    """
    return _consult(query, session_id, "flash", file_paths)


@mcp.tool()
def consult_gemini_pro(
    query: str,
    session_id: str = "default",
    file_paths: list[str] | None = None,
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
        file_paths: (Optional) Essential files to get started with exploring.
    """
    return _consult(query, session_id, "pro", file_paths)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
