# Development Guide

Internal documentation for gpal development.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              MCP Client                             │
│    (Claude Desktop, Cursor, VS Code, etc.)          │
└─────────────────────┬───────────────────────────────┘
                      │ MCP Protocol
                      ▼
┌─────────────────────────────────────────────────────┐
│                 gpal Server                         │
│                                                     │
│  ┌─────────────────┐  ┌─────────────────┐          │
│  │ consult_flash   │  │ consult_pro     │  ← Tools │
│  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │
│           └──────────┬─────────┘                    │
│                      ▼                              │
│  ┌──────────────────────────────────────┐          │
│  │       Session Manager                 │          │
│  │  (history preservation, model switch) │          │
│  └──────────────────┬───────────────────┘          │
│                      │                              │
└──────────────────────┼──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Google Gemini API                      │
│                                                     │
│  Gemini has internal tools:                         │
│  • list_directory  • read_file  • search_project   │
│                                                     │
│  Automatic Function Calling enabled                 │
│  (Gemini autonomously explores the codebase)        │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
gpal/
├── src/gpal/
│   ├── __init__.py       # Package metadata (__version__)
│   └── server.py         # MCP server + all logic
├── tests/
│   ├── test_tools.py     # Unit tests (pytest)
│   ├── test_agentic.py   # Manual: autonomous exploration
│   ├── test_connectivity.py  # Manual: API ping
│   └── test_switching.py     # Manual: Flash→Pro history
└── pyproject.toml        # Dependencies & entry point
```

## Key Design Decisions

### Stateful Sessions

Sessions live in memory (`sessions` dict). Same `session_id` = same conversation. History migrates when switching models.

⚠️ **Limitation**: Sessions are not persisted. Server restart = fresh state.

### Two-Tier Model Strategy

| Model | Alias | Best For |
|-------|-------|----------|
| `gemini-3-flash-preview` | `flash` | Fast exploration, searching, listing |
| `gemini-3-pro-preview` | `pro` | Deep reasoning, synthesis, code review |

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_FILE_SIZE` | 10 MB | Prevents accidental large file reads |
| `MAX_SEARCH_FILES` | 1000 | Caps glob expansion |
| `MAX_SEARCH_MATCHES` | 20 | Truncates search results |
| `MAX_TOOL_CALLS` | 10 | Limits autonomous tool use per response |

## Testing

```bash
# Install dev dependencies first
uv sync --all-extras

# Unit tests (no API key needed)
uv run pytest tests/test_tools.py -v

# Manual integration tests (requires GEMINI_API_KEY)
export GEMINI_API_KEY="..."
uv run python tests/test_connectivity.py
uv run python tests/test_agentic.py
uv run python tests/test_switching.py
```

## Private API Usage

⚠️ gpal uses some internal attributes from `google-genai`:

- `session._curated_history` — preferred for history migration (falls back to `.history`)
- `session._gpal_model` — custom attribute we add to track current model

These may break with library updates. A future version could wrap sessions in a custom class.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes* | Google Gemini API key |
| `GOOGLE_API_KEY` | Yes* | Alternative name (same purpose) |

*One of these must be set.
