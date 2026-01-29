# gpal - Gemini as an MCP with tools to explore the repo

`gpal` is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives
your IDE or agentic workflow access to Gemini models. It wraps the latest Google Gemini models
in a stateful, tool-equipped interface designed for deep code analysis.

## Features

*   **Stateful Consulting:** Maintains conversation history (`session_id`), allowing for iterative debugging and architectural debates.
*   **High-Agency:** Equipped with internal tools (`list_directory`, `read_file`, `search_project`) to autonomously explore your codebase. When you ask a question, Gemini doesn't just guess; it lists, reads, and searches files itself to gather context.
*   **Nested Agency:** You can ask your host model (e.g. Claude) to "Ask Gemini to find the bug," and Claude will delegate the entire exploration and analysis task to Gemini.
*   **Massive Context Window:** Leverages Gemini 3's 2M token context window to ingest entire modules, large files, or extensive documentation sets.
*   **Multi-Tier Consultation:** Choose between speed and depth:
    *   `consult_gemini_flash`: **The Scout.** Optimized for exploration, searching, and context gathering. Use this *first* to map the codebase and find the right files.
    *   `consult_gemini_pro`: **The Architect.** Optimized for deep reasoning and synthesis. Use this *after* Flash has found the relevant context to perform complex audits or architectural reviews.
    *   **Seamless Switching:** Conversation history is preserved when switching between Flash and Pro for the same `session_id`.
*   **Open Standard:** Built on MCP, making it compatible with Claude Desktop, Cursor, VS Code, and other MCP clients.

## Installation

### Prerequisites

*   Python 3.12+
*   [`uv`](https://github.com/astral-sh/uv) (Recommended)
*   Google Gemini API Key (Get one at [AI Studio](https://aistudio.google.com/))

### Running Standalone

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/tobert/gpal.git
    cd gpal
    ```

2.  **Run with `uv`:**
    ```bash
    export GEMINI_API_KEY="your_key_here"
    uv run gpal
    ```

## Usage

### with Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gpal": {
      "command": "bash",
      "args": [
        "-c",
        "GEMINI_API_KEY=$(< ~/.gpal-api-key) uv --directory /absolute/path/to/gpal run gpal"
      ]
    }
  }
}
```

Once connected, you will have two new tools: `consult_gemini_flash` and `consult_gemini_pro`.

**Examples:**
> *"Ask `consult_gemini_flash` to summarize `src/main.py`."*
> *"Ask `consult_gemini_pro` to identify potential security vulnerabilities in the `auth` module based on the files in `src/gpal/auth.py`."*

### Programmatic Usage

You can use `gpal` as a library in your own Python agents:

```python
from gpal.server import consult_gemini_flash, consult_gemini_pro

# Use Flash for quick queries
flash_response = consult_gemini_flash.fn(
    "List the top-level files in this directory.", 
    session_id="dev-session-1"
)
print(f"Flash: {flash_response}")

# Continue the same session with Pro for deeper analysis
pro_response = consult_gemini_pro.fn(
    "From the files you just listed, what is the main entry point of the application?", 
    session_id="dev-session-1"
)
print(f"Pro: {pro_response}")
```

## Development

*   **Test:** `uv run pytest`
*   **Lint:** `uv run ruff check .` (if configured)

## License

MIT License. See [LICENSE](LICENSE) for details.
