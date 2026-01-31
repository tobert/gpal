# gpal Future Features

This document captures high-value features identified during the initial expansion of `gpal`.

## 1. Agentic Workflows (Planned)

Moving beyond "consultation" to "delegation".
- **Refactoring Agent:** A loop that edits files, runs tests (via `code_execution` or shell), and iterates until green.
- **Review Agent:** specialized system instruction for code review that outputs structured comments.

## Completed Features

- **Semantic Search:** Vector-based code search using Gemini embeddings and ChromaDB.
- **Multimodal Support:** Inline media (images/audio/video) and File API integration for large files.
- **Structured Output:** JSON mode support with schema validation.
- **Generative Capabilities:** `generate_image` (Imagen 4) and `generate_speech` tools.
