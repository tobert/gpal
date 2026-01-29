# gpal (Gemini Principal Assistant Layer)

`gpal` is a Model Context Protocol (MCP) server designed to act as a "Second Brain" for software engineering. It provides a stateful, context-aware interface to Google's Gemini 3 models (Principal Software Engineer persona).

## Project Structure

*   `src/gpal/server.py`: Main MCP server implementation using `fastmcp`.
*   `tests/`: Utility scripts for verification.
    *   `test_consult.py`: Tests the `consult_gemini` tool.
    *   `list_available_models.py`: Helper to see which Gemini models your API key can access.
*   `pyproject.toml`: Dependency management using `uv`.
*   `LICENSE`: MIT License.

## Setup & Installation

1.  **Prerequisites:**
    *   `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
    *   Google Gemini API Key in `~/.gpal-api-key`.

2.  **Running with Key Injection:**
    ```bash
    GEMINI_API_KEY=$(< ~/.gpal-api-key) uv run gpal
    ```

3.  **Integrating with Claude Desktop:**
    Add to your config:
    ```json
    "gpal": {
      "command": "bash",
      "args": [
        "-c",
        "GEMINI_API_KEY=$(< ~/.gpal-api-key) uv --directory /home/atobey/src/gpal run gpal"
      ]
    }
    ```

## Capabilities

### Tool: `consult_gemini`
*   **Persona:** Principal Software Engineer.
*   **Stateful:** Uses `session_id` to maintain multi-turn conversations.
*   **Multi-Model:** Supports seamless switching between models within a session.
    *   `model="flash"` (Default): Gemini 3 Flash Preview (Fast, efficient).
    *   `model="pro"`: Gemini 3 Pro Preview (Deep reasoning).
    *   *Usage:* Start with Flash to explore, then switch to Pro for complex analysis. History is preserved.
*   **Agentic:** Gemini has access to internal tools to explore your codebase autonomously:
    *   `list_directory`: Lists files in the repo.
    *   `read_file`: Reads file contents.
    *   `search_project`: Greps for patterns.
*   **Contextual:** Accepts `file_paths` to ingest local codebases into the Gemini context window.

## Testing
Run the full test suite:
```bash
uv run pytest
```

Individual tests:
*   `tests/test_tools.py`: Unit tests for file system tools.
*   `tests/test_agentic.py`: Verifies Gemini's ability to use tools autonomously.
*   `tests/test_consult.py`: Basic connectivity check.