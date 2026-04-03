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
| **Stateful sessions** | Maintains conversation history via `ctx.session_id` |
| **Autonomous exploration** | Gemini has tools to list, read, and search files |
| **FileSearch** | Semantic code search via Google's native FileSearch API |
| **Gemini 3 Series** | Supports Flash and Pro with unified auto mode |
| **Context Caching** | Store large code contexts to reduce costs and latency |
| **Observability** | Native OpenTelemetry support (OTLP gRPC) |
| **Distributed Tracing** | Propagates `traceparent` from MCP requests |
| **Multimodal** | Analyze images, audio, video, PDFs |
| **Batch Processing** | Async discounted (~50%) Gemini batch API |

**Limits:** 10MB file reads, 20MB inline media, 20 search matches max.

### Model Tiers

| Tool | Model | Use Case |
|------|-------|----------|
| `consult_gemini` | `auto` (default) | Lite explores, then Flash synthesizes |
| `consult_gemini` | `flash` | Fast, efficient mapping and searching |
| `consult_gemini` | `pro` | Deep reasoning, complex reviews |
| `consult_gemini_oneshot` | `flash` / `pro` | Stateless single-shot queries, no session history |

**Auto mode:** Lite autonomously explores the codebase (cheap, thorough), then Flash synthesizes over what Lite found. Use `model="pro"` for deep reasoning (Lite explores, then Pro with thinking HIGH).

### Observability & Tracing

gpal supports native OpenTelemetry for monitoring and distributed tracing. It automatically propagates `traceparent` headers from incoming MCP requests.

```bash
# Configure via standard environment variables
export OTEL_SERVICE_NAME="gpal-server"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Or via CLI argument
uv run gpal --otel-endpoint localhost:4317
```

### Context Caching

Reduce costs for large projects by caching context on Google's servers:

1.  Upload large files using `upload_file`.
2.  Create a cache using `create_context_cache` with the returned URIs.
3.  Reference the cache name in `consult_gemini` calls via the `cached_content` parameter.
4.  View active caches via the `gpal://caches` resource.

### FileSearch

Semantic code search using Google's native FileSearch API — no local embeddings or databases:

```python
# Create a store and upload files
create_file_store("my-project")
upload_to_file_store("stores/...", "src/server.py")

# Gemini searches stores automatically during generation
consult_gemini("find authentication logic", model="auto")
```

- Google handles chunking, embedding, and retrieval
- Stores managed via `create_file_store`, `upload_to_file_store`, `list_file_stores`, `delete_file_store`
- When stores exist, Gemini searches them automatically during `consult_gemini` calls

### Custom System Prompts

Customize what Gemini "knows" about you, your project, or your workflow by composing system prompts from multiple sources.

**Config file** (`~/.config/gpal/config.toml`):

```toml
# Files loaded in order and concatenated
system_prompts = [
    "~/.config/gpal/GEMINI.md",
    "~/CLAUDE.md",
]

# Inline text appended after files
system_prompt = "常に日本語で回答してください (Always respond in Japanese)"

# Set to false to fully replace the built-in prompt with your own
include_default_prompt = true
```

Paths support `~` and `$ENV_VAR` expansion, so you can use `$WORKSPACE/CLAUDE.md` etc.

**CLI flags** (repeatable, concatenated in order):

```bash
# Append additional prompt files
uv run gpal --system-prompt /path/to/project-context.md

# Multiple files
uv run gpal --system-prompt ~/GEMINI.md --system-prompt ./CLAUDE.md

# Replace the built-in prompt entirely
uv run gpal --system-prompt ~/my-prompt.md --no-default-prompt
```

**Composition order:**
1. Built-in gpal system instruction (unless `include_default_prompt = false` or `--no-default-prompt`)
2. Files from `system_prompts` in config.toml
3. Inline `system_prompt` from config.toml
4. Files from `--system-prompt` CLI flags

Check what's active via the `gpal://info` resource — it shows which sources contributed and the total instruction length.

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

> "Use `consult_gemini` to find where errors are handled"

## Development

```bash
uv run pytest              # Run tests
uv run pytest -v           # Verbose output
```

⚠️ **Note:** Integration tests (`test_connectivity.py`, `test_agentic.py`, `test_switching.py`) make live API calls and will incur Gemini API costs.

## See Also

- **[cpal](https://github.com/tobert/cpal)** — The inverse: an MCP server that lets Gemini (or any MCP client) consult Claude. Your pal Claude.

## License

MIT — see [LICENSE](LICENSE)

## Roadmap / TODO

- **Refactoring Agent:** A loop that edits files, runs tests (via `code_execution` or shell), and iterates until green.
- **Review Agent:** specialized system instruction for code review that outputs structured comments.
