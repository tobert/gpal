"""Tests for semantic search index functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpal.index import (
    CodebaseIndex,
    get_index_path,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BINARY_EXTENSIONS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    MAX_CONCURRENT_EMBEDS,
    MAX_FILE_SIZE,
    MAX_RETRIES,
    RATE_LIMIT_DELAY,
)


# ─────────────────────────────────────────────────────────────────────────────
# XDG Path Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_xdg_path_default():
    """Verify get_index_path returns XDG-compliant path when XDG_DATA_HOME is not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove XDG_DATA_HOME if present
        os.environ.pop("XDG_DATA_HOME", None)
        path = get_index_path()
        assert path == Path.home() / ".local" / "share" / "gpal" / "index"


def test_xdg_path_custom():
    """Verify get_index_path respects XDG_DATA_HOME."""
    with patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/data"}):
        path = get_index_path()
        assert path == Path("/custom/data/gpal/index")


# ─────────────────────────────────────────────────────────────────────────────
# _should_index Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    return MagicMock()


@pytest.fixture
def index_with_gitignore(tmp_path, mock_client):
    """Create a CodebaseIndex with a .gitignore file."""
    # Create .gitignore
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n*.log\nbuild/\n__pycache__/\n")

    # Create the index (mocking chromadb)
    with patch("gpal.index.chromadb.PersistentClient"):
        index = CodebaseIndex(tmp_path, mock_client)
    return index


def test_should_index_normal_file(index_with_gitignore, tmp_path):
    """Normal source files should be indexed."""
    test_file = tmp_path / "main.py"
    test_file.write_text("print('hello')")
    assert index_with_gitignore._should_index(test_file) is True


def test_should_index_hidden_file(index_with_gitignore, tmp_path):
    """Hidden files should not be indexed."""
    hidden = tmp_path / ".hidden"
    hidden.write_text("secret")
    assert index_with_gitignore._should_index(hidden) is False


def test_should_index_hidden_dir(index_with_gitignore, tmp_path):
    """Files in hidden directories should not be indexed."""
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    config = hidden_dir / "config"
    config.write_text("git config")
    assert index_with_gitignore._should_index(config) is False


def test_should_index_binary_file(index_with_gitignore, tmp_path):
    """Binary files should not be indexed."""
    binary = tmp_path / "compiled.pyc"
    binary.write_bytes(b"\x00\x01\x02")
    assert index_with_gitignore._should_index(binary) is False


def test_should_index_gitignore_pattern(index_with_gitignore, tmp_path):
    """Files matching .gitignore patterns should not be indexed."""
    # node_modules should be ignored
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    pkg = node_modules / "some_pkg" / "index.js"
    pkg.parent.mkdir()
    pkg.write_text("module.exports = {}")
    assert index_with_gitignore._should_index(pkg) is False

    # .log files should be ignored
    log_file = tmp_path / "debug.log"
    log_file.write_text("log content")
    assert index_with_gitignore._should_index(log_file) is False


def test_should_index_respects_pycache(index_with_gitignore, tmp_path):
    """__pycache__ directories should be ignored per .gitignore."""
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    cache_file = pycache / "module.cpython-312.pyc"
    cache_file.write_bytes(b"\x00\x01\x02")
    assert index_with_gitignore._should_index(cache_file) is False


def test_should_index_multipart_extension(index_with_gitignore, tmp_path):
    """Multi-part extensions like .min.js should not be indexed."""
    minified_js = tmp_path / "bundle.min.js"
    minified_js.write_text("!function(){console.log('minified')}();")
    assert index_with_gitignore._should_index(minified_js) is False

    minified_css = tmp_path / "styles.min.css"
    minified_css.write_text("body{margin:0}")
    assert index_with_gitignore._should_index(minified_css) is False

    # But normal .js files should be indexed
    normal_js = tmp_path / "app.js"
    normal_js.write_text("console.log('hello');")
    assert index_with_gitignore._should_index(normal_js) is True


# ─────────────────────────────────────────────────────────────────────────────
# _chunk_file Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_index(tmp_path, mock_client):
    """Create a simple CodebaseIndex without .gitignore."""
    with patch("gpal.index.chromadb.PersistentClient"):
        index = CodebaseIndex(tmp_path, mock_client)
    return index


def test_chunk_file_small(simple_index, tmp_path):
    """Small files produce a single chunk."""
    small_file = tmp_path / "small.py"
    small_file.write_text("line1\nline2\nline3")

    chunks = simple_index._chunk_file(small_file)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["file"] == "small.py"
    assert chunks[0]["metadata"]["start_line"] == 1
    assert chunks[0]["metadata"]["end_line"] == 3
    assert "line1" in chunks[0]["text"]


def test_chunk_file_large(simple_index, tmp_path):
    """Large files are split into overlapping chunks."""
    # Create a file with 100 lines
    lines = [f"line {i}" for i in range(1, 101)]
    large_file = tmp_path / "large.py"
    large_file.write_text("\n".join(lines))

    chunks = simple_index._chunk_file(large_file)

    # With CHUNK_SIZE=50 and CHUNK_OVERLAP=10, 100 lines should produce:
    # Chunk 1: lines 1-50
    # Chunk 2: lines 41-90 (starting at 40, which is 50-10)
    # Chunk 3: lines 81-100 (starting at 80, which is 80)
    assert len(chunks) >= 2

    # First chunk starts at line 1
    assert chunks[0]["metadata"]["start_line"] == 1

    # Check overlap - second chunk should start before first chunk ends
    if len(chunks) > 1:
        first_end = chunks[0]["metadata"]["end_line"]
        second_start = chunks[1]["metadata"]["start_line"]
        assert second_start < first_end  # Overlap exists


def test_chunk_file_empty(simple_index, tmp_path):
    """Empty files produce no chunks."""
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    chunks = simple_index._chunk_file(empty_file)
    assert chunks == []


def test_chunk_file_binary_content(simple_index, tmp_path):
    """Files with non-UTF8 bytes get replacement characters instead of crashing.

    Binary files are filtered by _should_index() (extension check), but if
    _chunk_file is called directly, it should handle bad bytes gracefully.
    """
    binary_file = tmp_path / "data.bin"
    binary_file.write_bytes(b"\x80\x81\x82\x83")

    chunks = simple_index._chunk_file(binary_file)
    # errors="replace" produces replacement chars, so we get chunks
    assert len(chunks) == 1
    assert "\ufffd" in chunks[0]["text"]  # U+FFFD replacement character


def test_chunk_file_id_format(simple_index, tmp_path):
    """Chunk IDs follow the expected format."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    chunks = simple_index._chunk_file(test_file)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "test.py:1-1"


# ─────────────────────────────────────────────────────────────────────────────
# Constants Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_binary_extensions_coverage():
    """Verify common binary extensions are covered."""
    expected = {".pyc", ".so", ".exe", ".png", ".jpg", ".mp3", ".zip", ".pdf"}
    for ext in expected:
        assert ext in BINARY_EXTENSIONS, f"Missing extension: {ext}"


def test_chunk_constants():
    """Verify chunking constants are sensible."""
    assert CHUNK_SIZE > CHUNK_OVERLAP, "Overlap must be smaller than chunk size"
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0


def test_max_file_size():
    """Verify MAX_FILE_SIZE matches server.py."""
    from gpal.server import MAX_FILE_SIZE as SERVER_MAX
    assert MAX_FILE_SIZE == SERVER_MAX


def test_embedding_batch_size():
    """Verify EMBEDDING_BATCH_SIZE is reasonable for API limits."""
    assert 50 <= EMBEDDING_BATCH_SIZE <= 250, "Batch size should be within typical API limits"


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting & Retry Constants Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rate_limit_constants():
    """Verify rate limiting constants are reasonable."""
    assert RATE_LIMIT_DELAY >= 0.01, "Rate limit delay should be at least 10ms"
    assert RATE_LIMIT_DELAY <= 1.0, "Rate limit delay shouldn't be too long"
    assert MAX_RETRIES >= 1, "Should retry at least once"
    assert MAX_RETRIES <= 10, "Shouldn't retry too many times"


def test_concurrency_constant():
    """Verify concurrency limit is reasonable."""
    assert MAX_CONCURRENT_EMBEDS >= 1, "Should allow at least 1 concurrent request"
    assert MAX_CONCURRENT_EMBEDS <= 50, "Shouldn't overwhelm the API"


def test_embedding_model():
    """Verify embedding model is set to the newer model."""
    assert EMBEDDING_MODEL == "gemini-embedding-001", "Should use newer embedding model"


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Indexing Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_file_needs_reindex_new_file(simple_index, tmp_path):
    """New files always need reindexing."""
    new_file = tmp_path / "new.py"
    new_file.write_text("print('hello')")
    assert simple_index._file_needs_reindex(new_file) is True


def test_file_needs_reindex_nonexistent(simple_index, tmp_path):
    """Non-existent files don't need reindexing."""
    nonexistent = tmp_path / "nonexistent.py"
    assert simple_index._file_needs_reindex(nonexistent) is False


def test_get_file_metadata_not_indexed(simple_index, tmp_path):
    """Files not yet indexed return None for metadata."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    # Mock meta_collection to return empty result
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": [], "metadatas": []}

    assert simple_index._get_file_metadata(test_file) is None


def test_update_file_metadata(simple_index, tmp_path):
    """File metadata can be stored and retrieved."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    # Mock the chromadb collection
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    simple_index._update_file_metadata(test_file, 3)

    # Verify add was called with correct data
    simple_index.meta_collection.add.assert_called_once()
    call_kwargs = simple_index.meta_collection.add.call_args
    assert "test.py" in call_kwargs.kwargs["ids"]
    metadata = call_kwargs.kwargs["metadatas"][0]
    assert metadata["chunk_count"] == 3
    assert "mtime" in metadata
    assert "size" in metadata


# ─────────────────────────────────────────────────────────────────────────────
# Dry Run Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rebuild_dry_run_returns_expected_keys(simple_index, tmp_path):
    """Dry run returns expected dict keys."""
    # Create some test files
    (tmp_path / "file1.py").write_text("print('hello')")
    (tmp_path / "file2.py").write_text("print('world')")

    # Mock the collection to avoid actual chromadb operations
    simple_index.collection = MagicMock()
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    result = simple_index.rebuild(dry_run=True)

    assert "indexed" in result
    assert "skipped" in result
    assert "removed" in result
    assert "chunks" in result
    assert result.get("dry_run") is True


def test_rebuild_dry_run_counts_files_and_chunks(simple_index, tmp_path):
    """Dry run correctly counts files and chunks."""
    # Create test files with known line counts
    (tmp_path / "small.py").write_text("line1\nline2\nline3")  # 1 chunk
    (tmp_path / "medium.py").write_text("\n".join([f"line{i}" for i in range(60)]))  # Multiple chunks

    simple_index.collection = MagicMock()
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    result = simple_index.rebuild(dry_run=True)

    assert result["indexed"] == 2
    assert result["chunks"] > 0
    assert result["dry_run"] is True


def test_rebuild_dry_run_no_api_calls(simple_index, tmp_path, mock_client):
    """Dry run should not make any API calls."""
    (tmp_path / "test.py").write_text("content")

    simple_index.collection = MagicMock()
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    simple_index.rebuild(dry_run=True)

    # Client embed methods should not be called
    mock_client.models.embed_content.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Max Files Limit Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rebuild_max_files_limit(simple_index, tmp_path):
    """max_files parameter limits the number of files indexed."""
    # Create more files than the limit
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text(f"print({i})")

    simple_index.collection = MagicMock()
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    result = simple_index.rebuild(dry_run=True, max_files=3)

    assert result["indexed"] == 3


def test_rebuild_max_files_none_indexes_all(simple_index, tmp_path):
    """max_files=None indexes all files."""
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text(f"print({i})")

    simple_index.collection = MagicMock()
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    result = simple_index.rebuild(dry_run=True, max_files=None)

    assert result["indexed"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Progress Callback Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rebuild_progress_callback_called(simple_index, tmp_path):
    """Progress callback is called during rebuild."""
    (tmp_path / "test.py").write_text("content")

    simple_index.collection = MagicMock()
    simple_index.meta_collection = MagicMock()
    simple_index.meta_collection.get.return_value = {"ids": []}

    progress_messages = []

    def callback(msg):
        progress_messages.append(msg)

    simple_index.rebuild(dry_run=True, progress_callback=callback)

    assert len(progress_messages) > 0
    # Should contain "Dry run" since we're in dry_run mode
    assert any("Dry run" in msg for msg in progress_messages)


# ─────────────────────────────────────────────────────────────────────────────
# Force Rebuild Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rebuild_force_clears_collections(simple_index, tmp_path):
    """force=True should delete collections before recreating them."""
    (tmp_path / "test.py").write_text("content")

    # Track collection deletions
    deleted_collections = []

    def track_delete(name):
        deleted_collections.append(name)

    # Mock chroma client
    simple_index.chroma = MagicMock()
    simple_index.chroma.delete_collection.side_effect = track_delete
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}
    simple_index.chroma.create_collection.return_value = mock_collection

    # Note: force=True without dry_run should call delete_collection
    # (dry_run=True skips the deletion)
    simple_index._rebuild_sync(force=True, dry_run=False)

    # Should have deleted both collections
    assert "code" in deleted_collections
    assert "file_metadata" in deleted_collections
