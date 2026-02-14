# Development Guide

Internal documentation for gpal development.

## Dogfooding

Before committing changes to gpal, use gpal to review them:

```
consult_gemini_pro(
    query="Review server.py for bugs, edge cases, and API misuse",
    file_paths=["src/gpal/server.py"]
)
```

Gemini catches real issues — see git history for proof.

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
│   ├── index.py          # Semantic search index (chromadb + Gemini embeddings)
│   └── server.py         # MCP server + all logic
├── tests/
│   ├── test_tools.py     # Unit tests (pytest)
│   ├── test_server.py    # Integration tests (FastMCP Client, in-process)
│   ├── test_index.py     # Index unit tests
│   ├── test_agentic.py   # Manual: autonomous exploration
│   ├── test_connectivity.py  # Manual: API ping
│   └── test_switching.py     # Manual: Flash→Pro history
└── pyproject.toml        # Dependencies & entry point
```

## Key Design Decisions

### Stateful Sessions

Sessions live in memory (`sessions` dict). Same `session_id` = same conversation. History migrates when switching models.

⚠️ **Limitation**: Sessions are not persisted. Server restart = fresh state.

### Model Strategy

We always prefer the latest and most capable models available from Google.

| Model | Alias | Best For |
|-------|-------|----------|
| `gemini-3-flash-preview` | `flash` | Fast exploration, searching, listing |
| `gemini-3-pro-preview` | `pro` | Deep reasoning, synthesis, code review |
| `gemini-flash-latest` | — | Web search, code execution (auto-updates) |
| `imagen-4.0-ultra-generate-001` | `imagen` | Ultra quality image generation (default) |
| `imagen-4.0-fast-generate-001` | `imagen-fast` | Fast image generation |
| `nano-banana-pro-preview` | `nano-pro` | Best quality images, text rendering, 4K |
| `gemini-2.5-flash-image` | `nano-flash` | Fast, efficient image generation |
| `gemini-2.5-pro-preview-tts` | `speech` | Text-to-speech synthesis (Pro quality) |

> **Note**: There is no separate "deep think" model. Gemini thinking mode is enabled via
> `ThinkingConfig(thinking_level="HIGH")` on Pro. The `consult_gemini_deep_think` tool was
> removed in v0.2.0 — use `consult_gemini_pro` instead.

### FastMCP Version

Using FastMCP 3.0.0rc1+. Upgraded from 2.14.x which had a regression in 2.14.5 breaking
async tool functions. The 3.x upgrade was clean — only required dropping `task=True` from
`rebuild_index` (background tasks now need `fastmcp[tasks]` extra).

`ctx.debug/info/warning/error/report_progress` are **async** — must be `await`ed.
`ctx.set_state/get_state` are **sync**.

### FastMCP 3.0 Feature Adoption (v0.3.1)

**Tool Timeouts** — `timeout=` on `@mcp.tool()` decorators. FastMCP uses `anyio.fail_after()`
and converts `TimeoutError` to `McpError(-32000)`.

| Tool | Timeout | Rationale |
|------|---------|-----------|
| `consult_gemini_flash` | 60s | Fast model, 1 min ceiling |
| `consult_gemini_pro` | 600s | Deep reasoning with tool calls |
| `rebuild_index` | 300s | Large index rebuilds |
| All others | None | Quick sync operations |

**Rich ToolResult** — `_consult` returns `ToolResult` (from `fastmcp.tools.tool`) on success
instead of plain `str`. Provides `structured_content` (model ID) and `meta` (model, session_id,
duration_ms) for client introspection. Error paths still return plain strings.

**OTel Simplification** — Removed `opentelemetry-instrumentation-fastapi` (unused — no FastAPI
app to instrument). Removed manual trace context extraction from request headers in `_consult`;
FastMCP 3.0 handles distributed trace context automatically via `inject_trace_context`/
`extract_trace_context`. Our child span (`gemini_call`) nests under FastMCP's automatic
`tools/call` span. The `setup_otel()` function is kept — it configures the OTel SDK
(TracerProvider + OTLP exporter) that makes FastMCP's built-in instrumentation actually export.

### Updating Google Models

Tips for keeping model IDs current:

- Use the `list_models` tool (or `client.models.list()` in the SDK) to discover current models
- Check https://ai.google.dev/gemini-api/docs/models for the latest model IDs
- Google's naming: `gemini-{version}-{variant}` for text/multimodal, `imagen-{version}` for image-only
- **Nano Banana** models use `generate_content` (NOT `generate_images`) — they're Gemini models with image output modality
- Preview models (`-preview` suffix) may change; GA models have dated suffixes like `-001`
- When updating: change constants at top of `server.py`, update `MODEL_ALIASES`, update `gpal://info` resource, update this doc

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_FILE_SIZE` | 10 MB | Prevents accidental large file reads |
| `MAX_INLINE_MEDIA` | 20 MB | Caps inline media size (use upload_file for larger) |
| `MAX_SEARCH_FILES` | 1000 | Caps glob expansion |
| `MAX_SEARCH_MATCHES` | 20 | Truncates search results |
| `MAX_SEARCH_RESULTS` | 10 | Limits web search results |
| `RESPONSE_MAX_TOOL_CALLS` | 25 | Limits autonomous tool calls per response |

### Semantic Search Index (index.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `CHUNK_SIZE` | 50 lines | Lines per code chunk |
| `CHUNK_OVERLAP` | 10 lines | Overlap between chunks for context |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `EMBEDDING_BATCH_SIZE` | 100 | Max chunks per API call |
| `RATE_LIMIT_DELAY` | 50ms | Delay between API batches |
| `MAX_RETRIES` | 3 | Retry attempts on 429 errors |
| `MAX_CONCURRENT_EMBEDS` | 10 | Concurrent embedding requests |

**Features**:
- **Incremental indexing**: Only re-indexes files that changed (by mtime/size)
- **Async concurrency**: Parallel embedding requests with semaphore control
- **Rate limiting**: Automatic retry on 429 errors with exponential backoff
- **Dry run mode**: Count files/chunks without API calls
- **Max files limit**: Cap indexing for testing/budget control

**Usage in MCP tools**:
```python
# rebuild_index() uses these options internally:
result = index.rebuild(
    force=False,      # True = full rebuild, False = incremental
    dry_run=False,    # True = count only, no API calls
    max_files=None,   # Limit files to index (for testing)
)
# Returns: {"indexed": N, "skipped": M, "removed": K}
```

## Testing

```bash
# Install dev dependencies first
uv sync --all-extras

# Unit tests (no API key needed)
uv run pytest tests/test_server.py tests/test_tools.py tests/test_index.py -v

# Manual integration tests (requires GEMINI_API_KEY)
# ⚠️ These make live API calls and will incur Gemini API costs!
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
