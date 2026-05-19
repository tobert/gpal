# Development Guide

Internal documentation for gpal development.

## Dogfooding

Before committing changes to gpal, use gpal to review them:

```
consult_gemini(
    query="Review server.py for bugs, edge cases, and API misuse",
    model="pro",
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
│  │ consult_gemini  │  │ consult_oneshot │  ← Tools │
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
│  • url_context    • file_search (when stores exist) │
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
│   ├── test_server.py    # Integration tests (FastMCP Client, in-process)
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
| `gemini-flash-lite-latest` | `lite` | Cheap exploration (auto mode Phase 1, auto-updates) |
| `gemini-3.5-flash` | `flash` | Frontier synthesis, searching, analysis |
| `gemini-3.1-pro-preview` | `pro` | Deep reasoning, synthesis, code review |
| `gemini-flash-latest` | — | Web search, code execution (auto-updates) |
| `imagen-4.0-ultra-generate-001` | `imagen` | Ultra quality image generation (default) |
| `imagen-4.0-fast-generate-001` | `imagen-fast` | Fast image generation |
| `nano-banana-pro-preview` | `nano-pro` | Best quality images, text rendering, 4K |
| `gemini-3.1-flash-image-preview` | `nano-flash` | Nano Banana 2 — fast image generation |
| `gemini-3.1-flash-tts-preview` | `speech` | Text-to-speech with expressive audio tags |
| `gemini-2.5-flash-preview-tts` | `speech-fast` | Fast, cheaper text-to-speech |

> **Note**: There is no separate "deep think" model. Gemini thinking mode is enabled via
> `ThinkingConfig(thinking_level="HIGH")` on Pro.

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
| `consult_gemini` | 660s | Unified tool (auto mode: Lite explore + synthesis) |
| `consult_gemini_oneshot` | 600s | Stateless queries (Pro+thinking can be slow) |
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

### `consult_gemini` Tool

Single unified tool for all Gemini consultations. Always runs a Lite exploration phase
first, then hands off to the requested model for synthesis:

| `model=` | Phase 1 (explore) | Phase 2 (synthesize) | Notes |
|----------|-------------------|---------------------|-------|
| `"auto"` | Lite | Flash | New default |
| `"flash"` | Lite | Flash | Same as auto |
| `"pro"` | Lite | Pro (thinking HIGH) | Deep synthesis |
| `"lite"` | — | Lite direct | No pipeline |
| full model ID | — | Direct pass-through | Backward compat |

**Thinking level** — controlled via `thinking` parameter (`"minimal"`, `"low"`, `"medium"`,
`"high"`, or `None`). Pro defaults to `"high"` when `thinking` is unset. Other models default
to no thinking unless explicitly requested.

- **Lite exploration**: Lite (Haiku-class) does cheap mechanical exploration — `cat` and `ls`
  with navigation. 1M context window, AFC enabled.
- **Flash synthesis** (role=analyst): Frontier model with tools still available to fill gaps.
- **Pro synthesis** (role=thinker): Deep thinking enabled, tools available to fill gaps.
- **`consult_gemini_oneshot`**: Separate stateless tool, no session. For independent questions
  where conversation context is noise. Pro also defaults to `thinking="high"`.

### Token Tracking & Rate Limiting

Sliding-window token tracker in `server.py`:

- `record_tokens(model, count)` — records usage after each Gemini call
- `tokens_in_window(model)` — returns tokens consumed in last 60s
- `token_stats()` — current usage per model (exposed in `gpal://info`)
- **Proactive throttling**: Before sending, checks if at 90% of TPM limit and sleeps 5s if so
- Token counts included in `ToolResult.meta` (`prompt_tokens`, `completion_tokens`, `total_tokens`)

### Updating Google Models

Tips for keeping model IDs current:

- Use the `list_models` tool (or `client.models.list()` in the SDK) to discover current models
- Check https://ai.google.dev/gemini-api/docs/models for the latest model IDs
- Google's naming: `gemini-{version}-{variant}` for text/multimodal, `imagen-{version}` for image-only
- **Nano Banana** models use `generate_content` (NOT `generate_images`) — they're Gemini models with image output modality
- Preview models (`-preview` suffix) may change; GA models have dated suffixes like `-001`
- When updating: change constants at top of `server.py`, update `MODEL_ALIASES`, update `gpal://info` resource, update this doc

### Thread Pool & AFC Safety

**`_EXECUTOR`** — Dedicated `ThreadPoolExecutor(max_workers=16, thread_name_prefix="gpal")`.
All `run_in_executor` calls use this instead of the default executor to avoid starving the
event loop's default pool. 16 workers is sufficient since AFC tool calls within one
`send_message()` are serial, not parallel.

**`_afc_local.in_afc`** — Thread-local flag set to `True` inside `_send_with_retry` around
`session.send_message()`, cleared in a `finally` block. Functions that check it:
- `_gemini_search`: skips `_sync_throttle` (outer `_consult` already throttled) and acquires
  `_afc_api_semaphore` to cap concurrent outbound API calls
- `semantic_search`: acquires `_afc_api_semaphore` for the same reason

**`_afc_api_semaphore`** — `threading.Semaphore(4)` limiting concurrent outbound Gemini API
calls from within AFC tool callbacks, preventing connection saturation when multiple sessions
run concurrently.

**Stdin watchdog** — Background asyncio task (`_stdin_watchdog`) polls `os.fstat()` on stdin
every 5s. If the fd is invalid (client disconnected/crashed), calls `os._exit(0)`
to prevent orphaned CPU-spinning processes. Wired via FastMCP's `lifespan=` parameter.

### Batch Tools

Six tools for async, discounted (~50%) Gemini batch processing. No AFC — pure
`generate_content`. Stateless: no in-memory tracking, all status queries hit the API.

| Tool | Annotations | Timeout |
|------|-------------|---------|
| `create_batch(queries, model, temperature)` | openWorld | 120s |
| `get_batch(name)` | readOnly, openWorld | 30s |
| `list_batches(limit=20)` | readOnly, openWorld | 30s |
| `get_batch_results(name)` | readOnly, openWorld | 60s |
| `cancel_batch(name)` | destructive, openWorld | 30s |
| `delete_batch(name)` | destructive, idempotent, openWorld | 30s |

- `queries` items: `{custom_id: str, prompt: str}` — `custom_id` stored in `InlinedRequest.metadata`
- System prompt: `_compose_instruction(_SYSTEM_AGENT)` — user config layers included automatically
- Results: `job.dest.inlined_responses` (fetched by re-calling `get()`)
- `cancel()` returns `None` — must call `get()` for final state
- State helpers: `types.JOB_STATES_SUCCEEDED`, `types.JOB_STATES_ENDED`
- `async for job in await client.aio.batches.list()` — note the `await` before iteration

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_FILE_SIZE` | 10 MB | Prevents accidental large file reads |
| `MAX_INLINE_MEDIA` | 20 MB | Caps inline media size (use upload_file for larger) |
| `MAX_SEARCH_FILES` | 1000 | Caps glob expansion |
| `MAX_SEARCH_MATCHES` | 20 | Truncates search results |
| `MAX_SEARCH_RESULTS` | 10 | Limits web search results |
| `RESPONSE_MAX_TOOL_CALLS` | 25 | Limits autonomous tool calls per response |

### FileSearch (replaces chromadb index)

Semantic code search uses Google's native FileSearch API instead of a local chromadb index.
Files are uploaded to FileSearch stores, where Google handles chunking, embedding, and retrieval.

**MCP tools:**
- `create_file_store(display_name)` — create a new store
- `upload_to_file_store(store_name, file_path)` — upload a file
- `list_file_stores()` — list stores with stats
- `delete_file_store(name)` — delete a store and its documents

**AFC integration:**
- `_active_stores` set tracks known stores (lazily populated on first consult)
- When stores exist, `Tool(file_search=FileSearch(file_search_store_names=[...]))` is added
  to the AFC tool config automatically — Gemini searches stores during generation
- Store management tools update `_active_stores` synchronously

**Resource:** `gpal://file_stores` — store statistics (replaces `gpal://index/stats`)

## Testing

```bash
# Install dev dependencies first
uv sync --all-extras

# Unit tests (no API key needed)
uv run pytest tests/test_server.py tests/test_tools.py -v

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

## Configuration

### CLI Options

| Option | Description |
|--------|-------------|
| `--otel-endpoint` | OTLP gRPC endpoint (e.g., `localhost:4317`) |
| `--api-key-file PATH` | Path to file containing the Gemini API key |
| `--system-prompt FILE` | Additional system prompt file (repeatable) |
| `--no-default-prompt` | Exclude the built-in default system instruction |

### Config File: `~/.config/gpal/config.toml`

Uses `$XDG_CONFIG_HOME/gpal/config.toml` (falls back to `~/.config/gpal/config.toml`).
Parsed with Python 3.12 stdlib `tomllib` — no extra dependency.

```toml
# System prompt files, loaded in order and concatenated
system_prompts = [
    "~/.config/gpal/GEMINI.md",
]

# Inline system prompt text (appended after files)
system_prompt = "常に日本語で回答してください (Always respond in Japanese)"

# If true (default), prepend the built-in gpal system instruction.
# Set to false to fully replace it with your own.
include_default_prompt = true
```

Paths support `~` and `$ENV_VAR` expansion.

**Composition order:**
1. Role-specific built-in prompt (`_SYSTEM_EXPLORER`, `_SYSTEM_THINKER`, or `_SYSTEM_AGENT`)
2. User layers (appended after the role prompt):
   a. Files from `system_prompts` list (in order)
   b. Inline `system_prompt` from config.toml
   c. Files from `--system-prompt` CLI flags (in order)

Role prompts share a common `_SYSTEM_BASE` identity. The role is selected automatically:
- **explorer**: Lite in explore phase (aggressive tool use, thoroughness)
- **analyst**: Flash in synthesis phase (frontier model, tools available)
- **thinker**: Pro in synthesis phase (deep thinking HIGH, tools available)
- **agent**: Direct model calls and oneshot (tools enabled)

All joined with `\n\n`. Provenance visible in `gpal://info`.
