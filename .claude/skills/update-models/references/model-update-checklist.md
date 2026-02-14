# Model Update Checklist

Detailed reference for updating Gemini model IDs in gpal.

## Files to Modify

When changing any model constant, update **all** of these locations:

### 1. `src/gpal/server.py` — Model Constants (top of file)

These values are illustrative; always check the actual file for current values:

```python
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_PRO = "gemini-3-pro-preview"
MODEL_SEARCH = "gemini-flash-latest"
MODEL_CODE_EXEC = "gemini-flash-latest"
MODEL_IMAGE = "imagen-4.0-ultra-generate-001"
MODEL_IMAGE_FAST = "imagen-4.0-fast-generate-001"
MODEL_IMAGE_PRO = "nano-banana-pro-preview"
MODEL_IMAGE_FLASH = "gemini-2.5-flash-image"
MODEL_SPEECH = "gemini-2.5-pro-preview-tts"
```

### 2. `src/gpal/server.py` — `MODEL_ALIASES` dict

Maps user-facing short names to model IDs. Update if adding or renaming aliases.

### 3. `src/gpal/server.py` — `NANO_BANANA_MODELS` set

Contains models that use `generate_content` for image generation (not `generate_images`).
If adding a new Gemini image model, add it here.

### 4. `src/gpal/server.py` — `gpal://info` resource

The `get_server_info()` resource returns all model IDs as JSON. Verify it references
the correct constants.

### 5. `src/gpal/server.py` — `create_context_cache()` fallback mapping

Contains hardcoded model IDs for caching fallback (caching requires stable versioned IDs).
Update when preview models get GA versions.

### 6. `CLAUDE.md` — Model Strategy table

The markdown table under "### Model Strategy" lists all models with aliases and use cases.

### 7. `pyproject.toml` + `src/gpal/__init__.py` — Version bump

Always bump the version when changing models. Both files must match.

## Model Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| `gemini-{N}-{variant}-preview` | Preview model, may change | `gemini-3-flash-preview` |
| `gemini-{N}.{M}-{variant}-{suffix}` | Dated/versioned GA | `gemini-2.5-flash-image` |
| `gemini-{variant}-latest` | Auto-updating alias | `gemini-flash-latest` |
| `imagen-{N}.{M}-{tier}-generate-{suffix}` | Imagen (image-only) | `imagen-4.0-ultra-generate-001` |
| `nano-banana-{variant}-preview` | Nano Banana (generateContent) | `nano-banana-pro-preview` |

## When to Use `-latest` Aliases

Safe for **stateless utility calls** where exact version doesn't matter:
- Web search (`MODEL_SEARCH`)
- Code execution (`MODEL_CODE_EXEC`)

**Do not use `-latest`** for:
- Consult tools (Flash/Pro) — capabilities differ significantly between versions
- Image generation — output quality varies between versions
- Speech — voice behavior changes between versions

## Nano Banana vs Imagen

- **Imagen** models use `client.models.generate_images()` API
- **Nano Banana** models use `client.models.generate_content()` with `response_modalities=["TEXT", "IMAGE"]`
- Check `NANO_BANANA_MODELS` set when adding new image models

## Verification Steps

After updating model IDs:

1. Run unit tests: `uv run pytest tests/test_server.py tests/test_tools.py -v`
2. Read `gpal://models/check` resource to verify API availability
3. Use `list_models` tool to confirm model exists in API
4. Bump version, reinstall: `uv cache clean --force gpal && uv tool install --force .`
5. Reconnect MCP (`/mcp`) and test the changed tools
