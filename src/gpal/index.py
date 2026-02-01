"""
gpal semantic search index using Gemini embeddings + chromadb.

Provides vector-based code search that finds code by meaning rather than
exact text matching.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Union

import chromadb
import pathspec
from google import genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB - matches server.py
CHUNK_SIZE = 50  # lines per chunk
CHUNK_OVERLAP = 10  # overlapping lines between chunks
EMBEDDING_MODEL = "gemini-embedding-001"  # Newer model, text-embedding-004 deprecated Jan 2026
EMBEDDING_BATCH_SIZE = 100  # Max chunks per Gemini API call

# Rate limiting and retry configuration
RATE_LIMIT_DELAY = 0.05  # 50ms between batches
MAX_RETRIES = 3  # Max retry attempts on failure
MAX_CONCURRENT_EMBEDS = 10  # Max concurrent embedding requests

# Binary/generated file extensions to skip
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".obj", ".bin", ".exe", ".dll",
    ".class", ".jar", ".war", ".ear",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".whl", ".egg",
    ".min.js", ".min.css",  # minified files
}


# ─────────────────────────────────────────────────────────────────────────────
# XDG Path Helper
# ─────────────────────────────────────────────────────────────────────────────


def get_index_path() -> Path:
    """Get XDG-compliant path for index storage."""
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        base = Path(xdg_data)
    else:
        base = Path.home() / ".local" / "share"
    return base / "gpal" / "index"


# ─────────────────────────────────────────────────────────────────────────────
# CodebaseIndex
# ─────────────────────────────────────────────────────────────────────────────


class CodebaseIndex:
    """
    Semantic code search index using Gemini embeddings and chromadb.

    Each project root gets a unique index directory based on a hash of
    the absolute path. The index persists across sessions.
    """

    def __init__(self, root: Path, client: genai.Client):
        """
        Initialize the index for a project root.

        Args:
            root: The project root directory to index.
            client: A configured Gemini API client.
        """
        self.root = root.resolve()
        self.client = client
        self.db_path = get_index_path() / self._path_hash()

        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.chroma = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.chroma.get_or_create_collection(
            name="code",
            metadata={"hnsw:space": "cosine"},
        )
        # Metadata collection for incremental indexing
        self.meta_collection = self.chroma.get_or_create_collection(
            name="file_metadata",
        )
        self._load_gitignore()

    def _path_hash(self) -> str:
        """Generate a unique hash for this project root."""
        return hashlib.md5(str(self.root).encode()).hexdigest()[:12]

    def _load_gitignore(self) -> None:
        """Load .gitignore patterns for filtering files."""
        self.ignore_spec: pathspec.PathSpec | None = None
        gitignore = self.root / ".gitignore"
        if gitignore.exists():
            try:
                patterns = gitignore.read_text(encoding="utf-8").splitlines()
                self.ignore_spec = pathspec.PathSpec.from_lines("gitignore", patterns)
            except (OSError, UnicodeDecodeError):
                pass  # If we can't read .gitignore, just skip it

    def _should_index(self, path: Path) -> bool:
        """
        Check if a file should be indexed.

        Skips:
        - Hidden files/directories (starting with .)
        - Binary files
        - Files over MAX_FILE_SIZE
        - Files matching .gitignore patterns
        """
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            return False

        # Skip hidden files/directories
        if any(part.startswith(".") for part in rel.parts):
            return False

        # Skip binary extensions (check full filename for multi-part like .min.js)
        name = path.name.lower()
        if any(name.endswith(ext) for ext in BINARY_EXTENSIONS):
            return False

        # Skip large files
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                return False
        except OSError:
            return False

        # Check .gitignore patterns
        if self.ignore_spec and self.ignore_spec.match_file(str(rel)):
            return False

        return True

    def _get_file_metadata(self, path: Path) -> dict | None:
        """Get stored metadata for a file, or None if not indexed."""
        try:
            rel_path = str(path.relative_to(self.root))
        except ValueError:
            return None

        result = self.meta_collection.get(ids=[rel_path])
        if result["ids"]:
            meta = result["metadatas"][0] if result["metadatas"] else {}
            return meta
        return None

    def _file_needs_reindex(self, path: Path) -> bool:
        """
        Check if a file needs to be re-indexed.

        Returns True if:
        - File has never been indexed
        - File mtime or size has changed since last index
        """
        try:
            stat = path.stat()
        except OSError:
            return False

        stored = self._get_file_metadata(path)
        if stored is None:
            return True

        # Compare mtime and size
        stored_mtime = stored.get("mtime", 0)
        stored_size = stored.get("size", -1)

        return stat.st_mtime != stored_mtime or stat.st_size != stored_size

    def _update_file_metadata(self, path: Path, chunk_count: int) -> None:
        """Store metadata for a file after successful indexing."""
        try:
            rel_path = str(path.relative_to(self.root))
            stat = path.stat()
        except (ValueError, OSError):
            return

        metadata = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "chunk_count": chunk_count,
        }

        # Upsert the metadata
        existing = self.meta_collection.get(ids=[rel_path])
        if existing["ids"]:
            self.meta_collection.update(
                ids=[rel_path],
                metadatas=[metadata],
            )
        else:
            self.meta_collection.add(
                ids=[rel_path],
                metadatas=[metadata],
                documents=[""],  # chromadb requires documents
            )

    def _remove_file_from_index(self, rel_path: str) -> None:
        """Remove a file's chunks and metadata from the index."""
        # Delete chunks (limit=None to get all chunks for this file)
        existing = self.collection.get(where={"file": rel_path}, limit=None)
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])

        # Delete metadata
        meta_existing = self.meta_collection.get(ids=[rel_path])
        if meta_existing["ids"]:
            self.meta_collection.delete(ids=[rel_path])

    def _remove_stale_files(self, current_files: set[str]) -> int:
        """
        Remove files from index that no longer exist on disk.

        Args:
            current_files: Set of relative paths currently in the codebase.

        Returns:
            Number of files removed.
        """
        # Get all indexed files from metadata (limit=None to get all)
        all_meta = self.meta_collection.get(limit=None)
        indexed_files = set(all_meta["ids"]) if all_meta["ids"] else set()

        # Find stale files
        stale = indexed_files - current_files
        for rel_path in stale:
            self._remove_file_from_index(rel_path)

        return len(stale)

    def _chunk_file(self, path: Path) -> list[dict]:
        """
        Split a file into overlapping chunks with metadata.

        Returns a list of dicts with id, text, and metadata for each chunk.
        """
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            return []

        if not lines:
            return []

        rel_path = str(path.relative_to(self.root))
        chunks = []

        for i in range(0, len(lines), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_lines = lines[i : i + CHUNK_SIZE]
            if not chunk_lines:
                continue

            start_line = i + 1
            end_line = i + len(chunk_lines)

            chunks.append({
                "id": f"{rel_path}:{start_line}-{end_line}",
                "text": "\n".join(chunk_lines),
                "metadata": {
                    "file": rel_path,
                    "start_line": start_line,
                    "end_line": end_line,
                },
            })

        return chunks

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(ResourceExhausted),
        reraise=True,
    )
    def _embed_batch(
        self, batch: list[str], task_type: str
    ) -> list[list[float]]:
        """
        Embed a single batch of texts with retry on rate limit.

        Args:
            batch: List of text strings (max EMBEDDING_BATCH_SIZE).
            task_type: Task type for embeddings.

        Returns:
            List of embedding vectors for this batch.
        """
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config={"task_type": task_type},
        )
        time.sleep(RATE_LIMIT_DELAY)
        return [e.values for e in response.embeddings]

    def _embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """
        Get embeddings from Gemini for a list of texts.

        Automatically batches requests to stay within API limits.
        Includes retry logic for rate limiting (429) errors.

        Args:
            texts: List of text strings to embed.
            task_type: Either "RETRIEVAL_DOCUMENT" for indexing or
                       "RETRIEVAL_QUERY" for searching.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            embeddings = self._embed_batch(batch, task_type)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(ResourceExhausted),
        reraise=True,
    )
    async def _embed_batch_async(
        self,
        batch: list[str],
        task_type: str,
        semaphore: asyncio.Semaphore,
    ) -> list[list[float]]:
        """
        Embed a single batch of texts asynchronously with concurrency control.

        Includes retry logic for rate limit errors.

        Args:
            batch: List of text strings (max EMBEDDING_BATCH_SIZE).
            task_type: Task type for embeddings.
            semaphore: Semaphore for concurrency control.

        Returns:
            List of embedding vectors for this batch.
        """
        async with semaphore:
            # Use the async API client
            response = await self.client.aio.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config={"task_type": task_type},
            )
            await asyncio.sleep(RATE_LIMIT_DELAY)
            return [e.values for e in response.embeddings]

    async def _embed_async(
        self,
        texts: list[str],
        task_type: str,
        semaphore: asyncio.Semaphore,
    ) -> list[list[float]]:
        """
        Get embeddings from Gemini asynchronously.

        Args:
            texts: List of text strings to embed.
            task_type: Task type for embeddings.
            semaphore: Semaphore for concurrency control.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Create batch tasks
        tasks = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            tasks.append(self._embed_batch_async(batch, task_type, semaphore))

        # Run all batches concurrently (semaphore limits actual concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results, handling any exceptions
        all_embeddings: list[list[float]] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            # result is list[list[float]] here
            all_embeddings.extend(result)  # type: ignore[arg-type]

        return all_embeddings

    async def index_file_async(
        self,
        path: Path,
        semaphore: asyncio.Semaphore,
    ) -> str | None:
        """
        Index a single file asynchronously.

        Removes existing chunks and metadata first to handle partial failures.

        Args:
            path: Path to the file to index.
            semaphore: Semaphore for concurrency control.

        Returns:
            Relative path if indexed, None if skipped.
        """
        try:
            rel_path = str(path.relative_to(self.root))
        except ValueError:
            return None

        # Skip unchanged files
        if not self._file_needs_reindex(path):
            return None

        # Delete old chunks AND metadata first (ensures clean state on failure)
        self._remove_file_from_index(rel_path)

        if not self._should_index(path):
            return None

        chunks = self._chunk_file(path)
        if not chunks:
            # Update metadata for empty files
            self._update_file_metadata(path, 0)
            return rel_path

        # Embed asynchronously
        texts = [c["text"] for c in chunks]
        embeddings = await self._embed_async(texts, "RETRIEVAL_DOCUMENT", semaphore)

        # Add to collection (sync - chromadb doesn't have async API)
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )

        self._update_file_metadata(path, len(chunks))
        return rel_path

    # Type alias for progress callbacks (sync or async)
    ProgressCallback = Union[
        Callable[[str, int, int], Awaitable[None]],  # async with (message, current, total)
        Callable[[str], None],  # sync with just message
    ]

    async def _notify_progress(
        self,
        callback: ProgressCallback | None,
        message: str,
        current: int = 0,
        total: int = 0,
    ) -> None:
        """
        Call progress callback, handling both sync and async variants.

        Introspects the callback signature to determine how many arguments it accepts.

        Args:
            callback: The progress callback (sync or async), or None.
            message: Progress message.
            current: Current item number (for 3-arg callbacks).
            total: Total items (for 3-arg callbacks).
        """
        if callback is None:
            return

        # Introspect signature to determine argument count
        sig = inspect.signature(callback)
        params_count = len(sig.parameters)

        if asyncio.iscoroutinefunction(callback):
            if params_count >= 3:
                await callback(message, current, total)
            else:
                await callback(message)  # type: ignore[call-arg]
        else:
            if params_count >= 3:
                callback(message, current, total)  # type: ignore[call-arg]
            else:
                callback(message)

    async def rebuild_async(
        self,
        force: bool = False,
        dry_run: bool = False,
        max_files: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        """
        Rebuild the index asynchronously with concurrent embedding.

        Args:
            force: If True, clear everything and rebuild from scratch.
            dry_run: If True, count files/chunks without making API calls.
            max_files: Maximum number of files to index (for testing/budget control).
            progress_callback: Optional callback for progress updates.
                Can be sync (message: str) -> None or
                async (message: str, current: int, total: int) -> Awaitable[None].

        Returns:
            Dict with counts: {"indexed": N, "skipped": M, "removed": K, "chunks": C}
            In dry_run mode, "indexed" shows what would be indexed.
        """
        # Check for embedding dimension mismatch (model change)
        if not force and not dry_run and not self._check_embedding_dimensions():
            await self._notify_progress(progress_callback, "Embedding model changed, forcing full rebuild...")
            force = True

        if force and not dry_run:
            # Full rebuild: clear everything
            self.chroma.delete_collection("code")
            self.collection = self.chroma.create_collection(
                name="code",
                metadata={"hnsw:space": "cosine"},
            )
            self.chroma.delete_collection("file_metadata")
            self.meta_collection = self.chroma.create_collection(
                name="file_metadata",
            )

        # Collect all indexable files
        files_to_index: list[Path] = []
        current_files: set[str] = set()
        total_chunks = 0

        for path in self.root.rglob("*"):
            if not path.is_file() or not self._should_index(path):
                continue

            try:
                rel_path = str(path.relative_to(self.root))
            except ValueError:
                continue

            current_files.add(rel_path)

            # Check if needs reindex
            if force or self._file_needs_reindex(path):
                files_to_index.append(path)

        skipped = len(current_files) - len(files_to_index)

        # Apply max_files limit
        if max_files is not None and len(files_to_index) > max_files:
            files_to_index = files_to_index[:max_files]

        # Dry run: count chunks without API calls
        if dry_run:
            for path in files_to_index:
                chunks = self._chunk_file(path)
                total_chunks += len(chunks)
            await self._notify_progress(
                progress_callback,
                f"Dry run: {len(files_to_index)} files, {total_chunks} chunks",
            )
            return {
                "indexed": len(files_to_index),
                "skipped": skipped,
                "removed": 0,
                "chunks": total_chunks,
                "dry_run": True,
            }

        total_files = len(files_to_index)
        await self._notify_progress(
            progress_callback, f"Indexing {total_files} files...", 0, total_files
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDS)

        # Index files concurrently
        tasks = [self.index_file_async(p, semaphore) for p in files_to_index]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        indexed = sum(1 for r in results if r is not None and not isinstance(r, Exception))

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                await self._notify_progress(
                    progress_callback,
                    f"Error indexing {files_to_index[i]}: {result}",
                    i + 1,
                    total_files,
                )

        # Remove stale files
        removed = self._remove_stale_files(current_files)

        await self._notify_progress(
            progress_callback,
            f"Done: {indexed} indexed, {skipped} skipped, {removed} removed",
            total_files,
            total_files,
        )

        return {"indexed": indexed, "skipped": skipped, "removed": removed}

    def _check_embedding_dimensions(self) -> bool:
        """
        Check if the existing index has compatible embedding dimensions.

        Returns True if compatible or empty, False if dimensions mismatch.
        """
        try:
            # Try to get one item to check dimensions
            result = self.collection.get(limit=1, include=["embeddings"])
            if not result["ids"]:
                return True  # Empty collection, compatible

            embeddings = result.get("embeddings")
            if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                # Get a test embedding to compare dimensions
                test_embed = self._embed(["test"], task_type="RETRIEVAL_DOCUMENT")
                if test_embed:
                    existing_dim = len(embeddings[0])
                    new_dim = len(test_embed[0])
                    if existing_dim != new_dim:
                        return False
            return True
        except Exception:
            return True  # On error, assume compatible

    def index_file(self, path: Path) -> int:
        """
        Index or re-index a single file.

        Removes any existing chunks and metadata before adding new ones.
        This ensures partial failures leave the file in a re-indexable state.

        Returns:
            Number of chunks indexed (0 if skipped).
        """
        try:
            rel_path = str(path.relative_to(self.root))
        except ValueError:
            return 0

        # Delete old chunks AND metadata first
        # This ensures failures leave the file in a state where _file_needs_reindex
        # returns True (missing metadata = needs reindex)
        self._remove_file_from_index(rel_path)

        if not self._should_index(path):
            return 0

        chunks = self._chunk_file(path)
        if not chunks:
            # Still update metadata for empty files so we don't re-check them
            self._update_file_metadata(path, 0)
            return 0

        # Batch embed all chunks
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )

        # Update metadata for incremental indexing
        self._update_file_metadata(path, len(chunks))

        return len(chunks)

    def rebuild(
        self,
        force: bool = False,
        dry_run: bool = False,
        max_files: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Rebuild the index, optionally incrementally.

        Uses async concurrency internally for faster embedding.
        By default, only re-indexes files that have changed (mtime/size).
        Use force=True for a complete rebuild.

        Args:
            force: If True, clear everything and rebuild from scratch.
            dry_run: If True, count files/chunks without making API calls.
            max_files: Maximum number of files to index (for testing/budget).
            progress_callback: Optional callback for progress updates.

        Returns:
            Dict with counts: {"indexed": N, "skipped": M, "removed": K}
            In dry_run mode, also includes "chunks" count and "dry_run": True.
        """
        # Check if we're already in an event loop
        try:
            asyncio.get_running_loop()
            # We're in an async context - can't use asyncio.run()
            # Fall back to sync implementation
            return self._rebuild_sync(force, dry_run, max_files, progress_callback)
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(
                self.rebuild_async(force, dry_run, max_files, progress_callback)
            )

    def _rebuild_sync(
        self,
        force: bool = False,
        dry_run: bool = False,
        max_files: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Synchronous rebuild fallback when already in an async context.

        Args:
            force: If True, clear everything and rebuild from scratch.
            dry_run: If True, count files/chunks without making API calls.
            max_files: Maximum number of files to index.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dict with counts: {"indexed": N, "skipped": M, "removed": K}
        """
        # Check for embedding dimension mismatch (model change)
        if not force and not dry_run and not self._check_embedding_dimensions():
            if progress_callback:
                progress_callback("Embedding model changed, forcing full rebuild...")
            force = True

        if force and not dry_run:
            # Full rebuild: clear everything
            self.chroma.delete_collection("code")
            self.collection = self.chroma.create_collection(
                name="code",
                metadata={"hnsw:space": "cosine"},
            )
            self.chroma.delete_collection("file_metadata")
            self.meta_collection = self.chroma.create_collection(
                name="file_metadata",
            )

        # Collect all indexable files
        files_to_index: list[Path] = []
        current_files: set[str] = set()

        for path in self.root.rglob("*"):
            if not path.is_file() or not self._should_index(path):
                continue

            try:
                rel_path = str(path.relative_to(self.root))
            except ValueError:
                continue

            current_files.add(rel_path)

            # Check if needs reindex
            if force or self._file_needs_reindex(path):
                files_to_index.append(path)

        skipped = len(current_files) - len(files_to_index)

        # Apply max_files limit
        if max_files is not None and len(files_to_index) > max_files:
            files_to_index = files_to_index[:max_files]

        # Dry run mode
        if dry_run:
            total_chunks = 0
            for path in files_to_index:
                chunks = self._chunk_file(path)
                total_chunks += len(chunks)
            if progress_callback:
                progress_callback(
                    f"Dry run: {len(files_to_index)} files, {total_chunks} chunks"
                )
            return {
                "indexed": len(files_to_index),
                "skipped": skipped,
                "removed": 0,
                "chunks": total_chunks,
                "dry_run": True,
            }

        indexed = 0
        for path in files_to_index:
            try:
                rel_path = str(path.relative_to(self.root))
            except ValueError:
                continue

            if progress_callback:
                progress_callback(f"Indexing {rel_path}...")

            self.index_file(path)
            indexed += 1

        # Remove stale files (not in dry run)
        removed = self._remove_stale_files(current_files)

        if progress_callback:
            progress_callback(f"Done: {indexed} indexed, {skipped} skipped, {removed} removed")

        return {"indexed": indexed, "skipped": skipped, "removed": removed}

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search for code matching a natural language query.

        Args:
            query: Natural language description of what to find.
            limit: Maximum number of results to return.

        Returns:
            List of matches with file, lines, score, and snippet.
        """
        # Embed query with RETRIEVAL_QUERY task type
        embeddings = self._embed([query], task_type="RETRIEVAL_QUERY")
        if not embeddings:
            return []

        query_embedding = embeddings[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )

        # Format results
        matches = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None

            # Cosine distance → similarity score
            score = round(1 - dist, 3) if dist is not None else None

            # Truncate snippet for display
            snippet = doc[:200] + "..." if len(doc) > 200 else doc

            matches.append({
                "file": meta.get("file", "unknown"),
                "lines": f"{meta.get('start_line', '?')}-{meta.get('end_line', '?')}",
                "score": score,
                "snippet": snippet,
            })

        return matches
