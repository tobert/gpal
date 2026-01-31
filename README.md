# gpal

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

An MCP server that gives your IDE or agent access to Google Gemini with autonomous codebase exploration. Your pal Gemini.

## Why gpal?

When you ask gpal a question, Gemini doesn't just guess — it **explores your codebase itself**. It lists directories, reads files, and searches for patterns before answering. This makes it ideal for:

- 🔍 **Deep code analysis** — "Find all error handling patterns in this codebase"
- 🏗️ **Architectural reviews** — "How is authentication implemented?"
- 🐛 **Bug hunting** — "Why might this function return null?"
- 📚 **Codebase onboarding** — "Explain how the request pipeline works"
- 🖼️ **Visual review** — Analyze screenshots, diagrams, video via `media_paths`
- 📋 **Structured extraction** — "List all API endpoints as JSON"

## Features

| Feature | Description |
|---------|-------------|
| **Stateful sessions** | Maintains conversation history via `session_id` |
| **Autonomous exploration** | Gemini has tools to list, read, and search files |
| **Semantic search** | Find code by meaning using Gemini embeddings + chromadb |
| **2M token context** | Leverages Gemini 3's massive context window |
| **Two-tier consultation** | Flash for speed, Pro for depth |
| **Seamless switching** | History preserved when switching between Flash and Pro |
| **Multimodal** | Analyze images, audio, video, PDFs |
| **File uploads** | Upload large files to Gemini's File API |
| **Structured output** | JSON mode with optional schema constraints |
| **Nested agency** | Claude can delegate entire tasks to Gemini |

**Limits:** 10MB file reads, 20MB inline media, 20 search matches max.

### Flash vs Pro

| Model | Use Case | Strengths |
|-------|----------|-----------|
| `consult_gemini_flash` | **Scout** — exploration first | Fast, efficient, great for searching and mapping |
| `consult_gemini_pro` | **Architect** — analysis second | Deep reasoning, synthesis, complex reviews |

**Workflow:** Start with Flash to gather context, then switch to Pro for analysis. Both share the same session history.

### Semantic Search

Find code by meaning, not just keywords:

```python
# First, build the index (run once per project, or after major changes)
rebuild_index("/path/to/project")

# Then search by concept
semantic_search("authentication logic")      # finds verify_jwt_token()
semantic_search("error handling patterns")   # finds try/catch blocks
semantic_search("database connection setup") # finds pool initialization
```

- Uses Gemini's `text-embedding-004` model + chromadb for vector search
- Index stored at `~/.local/share/gpal/index/` (XDG compliant)
- Respects `.gitignore`, skips binary/hidden files

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended)
- [Gemini API key](https://aistudio.google.com/)

### Quick Start

```bash
git clone https://github.com/tobert/gpal.git
cd gpal
export GEMINI_API_KEY="your_key_here"  # or GOOGLE_API_KEY
uv run gpal
```

## Usage

### Claude Desktop / Cursor / VS Code

Add to your MCP config (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gpal": {
      "command": "uv",
      "args": ["--directory", "/path/to/gpal", "run", "gpal"],
      "env": {
        "GEMINI_API_KEY": "your_key_here"
      }
    }
  }
}
```

Then ask your AI assistant:

> "Ask Gemini to analyze the authentication flow in this codebase"

> "Use `consult_gemini_flash` to find where errors are handled"

## Development

```bash
uv run pytest              # Run tests
uv run pytest -v           # Verbose output
```

⚠️ **Note:** Integration tests (`test_connectivity.py`, `test_agentic.py`, `test_switching.py`) make live API calls and will incur Gemini API costs.

## Known Limitations / TODO

- **Semantic search is MCP-only**: `semantic_search` and `rebuild_index` are available to MCP clients (Claude, Cursor) but not to Gemini's internal autonomous tools. Adding chromadb-based functions to Gemini's tool list causes mysterious failures (likely google-genai + chromadb compatibility issue). Investigate later.
- **Thread safety**: The global index cache (`_indexes`) and `rebuild()` vs `search()` operations are not thread-safe. Concurrent rebuilds during search may cause empty results. Acceptable for single-user MCP usage.
- **Serial indexing**: `rebuild_index()` processes files sequentially. For large codebases this is slow. Future: parallelize with `ThreadPoolExecutor`.
- **Nested .gitignore**: Only reads root `.gitignore`, ignores nested ones (common in monorepos).

## See Also

- **[cpal](https://github.com/tobert/cpal)** — The inverse: an MCP server that lets Gemini (or any MCP client) consult Claude. Your pal Claude.

## License

MIT — see [LICENSE](LICENSE)
