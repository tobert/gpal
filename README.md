# gpal

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io/)

**Gemini Principal Assistant Layer** — an MCP server that gives your IDE or agent access to Google Gemini with autonomous codebase exploration.

## Why gpal?

When you ask gpal a question, Gemini doesn't just guess — it **explores your codebase itself**. It lists directories, reads files, and searches for patterns before answering. This makes it ideal for:

- 🔍 **Deep code analysis** — "Find all error handling patterns in this codebase"
- 🏗️ **Architectural reviews** — "How is authentication implemented?"
- 🐛 **Bug hunting** — "Why might this function return null?"
- 📚 **Codebase onboarding** — "Explain how the request pipeline works"
- 🖼️ **Visual review** — "What's wrong with this screenshot?" (images, video, audio)
- 📋 **Structured extraction** — "List all API endpoints as JSON"

## Features

| Feature | Description |
|---------|-------------|
| **Stateful sessions** | Maintains conversation history via `session_id` |
| **Autonomous exploration** | Gemini has tools to list, read, and search files |
| **2M token context** | Leverages Gemini 3's massive context window |
| **Two-tier consultation** | Flash for speed, Pro for depth |
| **Seamless switching** | History preserved when switching between Flash and Pro |
| **Multimodal** | Analyze images, audio, video, PDFs |
| **File uploads** | Upload large files to Gemini's File API |
| **Structured output** | JSON mode with optional schema constraints |
| **Nested agency** | Claude can delegate entire tasks to Gemini |

### Flash vs Pro

| Model | Use Case | Strengths |
|-------|----------|-----------|
| `consult_gemini_flash` | **Scout** — exploration first | Fast, efficient, great for searching and mapping |
| `consult_gemini_pro` | **Architect** — analysis second | Deep reasoning, synthesis, complex reviews |

**Workflow:** Start with Flash to gather context, then switch to Pro for analysis. Both share the same session history.

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
      "command": "bash",
      "args": [
        "-c",
        "GEMINI_API_KEY=$(< ~/.gpal-api-key) uv --directory /path/to/gpal run gpal"
      ]
    }
  }
}
```

Then ask your AI assistant:

> "Ask Gemini to analyze the authentication flow in this codebase"

> "Use `consult_gemini_flash` to find where errors are handled"

### Programmatic Usage

```python
from gpal.server import consult_gemini_flash, consult_gemini_pro, upload_file

# Flash: Quick exploration
result = consult_gemini_flash.fn(
    "What files are in the src directory?",
    session_id="review-1"
)

# Pro: Deep analysis (same session, history preserved)
analysis = consult_gemini_pro.fn(
    "Based on those files, explain the architecture",
    session_id="review-1"
)

# Multimodal: Analyze screenshots or diagrams
review = consult_gemini_pro.fn(
    "What's wrong with this UI?",
    media_paths=["screenshot.png"]
)

# File upload: For large files (videos, big PDFs)
uri = upload_file.fn("demo.mp4")
feedback = consult_gemini_pro.fn(
    "Review this product demo",
    file_uris=[uri.split(": ")[1].split(" ")[0]]  # extract URI
)

# JSON mode: Structured output
data = consult_gemini_flash.fn(
    "List the top 5 largest functions",
    json_mode=True
)
```

## Development

```bash
uv run pytest              # Run tests
uv run pytest -v           # Verbose output
```

## License

MIT — see [LICENSE](LICENSE)
