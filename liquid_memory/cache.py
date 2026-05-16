"""Stateful compression cache.

The proxy compresses every incoming request from scratch today. For
two important workload families that is wasteful:

  1. RAG: the same N documents get queried hundreds of times per day
     against different user questions. Compressing the document portion
     once and reusing the fact pack across queries cuts the local
     extractor's CPU/GPU bill linearly with cache hit rate.

  2. Agent loops: a long-lived agent thread reuses the same system
     prompt + accumulated tool-call history on every turn. Re-extracting
     that prefix on every tool invocation is the same waste at higher
     frequency.

Both cases boil down to: a stable byte-string keyed cache lookup before
we burn extractor cycles. This module provides that lookup with two
swappable backends and a single async-friendly API.

USAGE (intended integration in proxy.py, not wired yet, see TODO at
the bottom of this file and ENGINE_INTEGRATION_TODO.md):

    cache = build_cache_from_env()
    pack = await cache.get(doc_text)
    if pack is None:
        pack = run_local_extractor(doc_text)
        await cache.put(doc_text, pack)

KEYING:
    The cache key is SHA-256 of (document_bytes + b'::' + ext_model_id).
    Including the extractor model id in the key means rolling out a new
    extractor invalidates everything automatically; nobody serves a
    stale pack produced by a previous model version.

BACKENDS:
    InMemoryCache    Process-local LRU with TTL. Zero deps, perfect for
                     single-replica deployments and unit tests.
    SqliteCache      File-backed. Survives restarts. Good fit for the
                     pip-installed single-host install path.
    RedisCache       Multi-replica deployments. Optional install via
                     `pip install liquid-memory[cache]`.

THREAD/ASYNC SAFETY:
    InMemoryCache uses a threading.Lock; safe to call from multiple
    threads and from FastAPI's threadpool. SqliteCache opens one
    connection per call (sqlite3 module's own per-connection locking
    handles concurrency). RedisCache is thread-safe per redis-py docs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Default extractor identifier baked into the key so that bumping the
# extractor model invalidates every cached pack automatically. Override
# by passing extractor_id explicitly to make_key / .get / .put.
DEFAULT_EXTRACTOR_ID = "mistral-7b-instruct-v0.3"

# Default TTL: 24 hours. Long enough that the typical RAG workload sees
# 95%+ hit rate on hot documents; short enough that a misconfigured
# extractor (or a customer doc that should not have been cached) ages
# out without a manual flush.
DEFAULT_TTL_SECONDS = 24 * 60 * 60


def make_key(doc_bytes: bytes, extractor_id: str = DEFAULT_EXTRACTOR_ID) -> str:
    """Stable cache key derived from the document and the extractor id.

    Returns a hex-encoded SHA-256 digest (64 chars). Same input is
    guaranteed to produce the same key; different extractors produce
    different keys for the same document.
    """
    h = hashlib.sha256()
    h.update(doc_bytes)
    h.update(b"::")
    h.update(extractor_id.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """Wrapper around the cached fact pack + when it was written."""
    pack: Dict[str, Any]
    created_at: float

    def is_expired(self, now: float, ttl_seconds: int) -> bool:
        return (now - self.created_at) > ttl_seconds


class CacheBackend(ABC):
    """Common interface so the rest of the proxy is backend-agnostic."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Return the cached entry or None. Implementations MUST NOT
        raise on cache miss; misses are normal traffic."""

    @abstractmethod
    def put(self, key: str, pack: Dict[str, Any]) -> None:
        """Store the fact pack. Implementations SHOULD silently drop
        entries that exceed any backend-specific size limit rather
        than raise (cache writes are best-effort)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Explicit invalidation. Used by the admin /cache/flush
        endpoint (not implemented yet, see TODO)."""

    @abstractmethod
    def clear(self) -> None:
        """Drop everything. Used by tests and by the manual flush
        admin command."""


# ----------------------------------------------------------------------
# Backend 1: in-memory LRU. Default for single-process deployments and
# unit tests. Bounded by `max_entries`; oldest entries get evicted.
# ----------------------------------------------------------------------
class InMemoryCache(CacheBackend):
    """Process-local LRU. Zero external deps. Thread-safe."""

    def __init__(self, max_entries: int = 256):
        self._max = max_entries
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            # Promote to most-recently-used.
            self._store.move_to_end(key)
            return entry

    def put(self, key: str, pack: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = CacheEntry(pack=pack, created_at=time.time())
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # drop oldest

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ----------------------------------------------------------------------
# Backend 2: SQLite. Survives restarts, single file, no external service.
# Good fit for the pip-installed single-host install where adding Redis
# would be an unwelcome dependency.
# ----------------------------------------------------------------------
class SqliteCache(CacheBackend):
    """File-backed cache, one row per (key, pack). Uses sqlite3's
    built-in per-connection locking via short-lived connections so it
    plays well with FastAPI's threadpool."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS lm_cache (
          key TEXT PRIMARY KEY,
          pack TEXT NOT NULL,
          created_at REAL NOT NULL
        );
    """

    def __init__(self, path: str):
        self._path = path
        # Eager schema setup so first .get/.put doesn't pay it.
        with self._conn() as conn:
            conn.execute(self._SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        # check_same_thread=False because FastAPI may dispatch the cache
        # call from a worker thread different from the one that created
        # the connection. We hold the connection for the duration of one
        # operation only (via the with-statement), so cross-thread reuse
        # is bounded and safe.
        return sqlite3.connect(self._path, check_same_thread=False, timeout=5.0)

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pack, created_at FROM lm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(pack=json.loads(row[0]), created_at=float(row[1]))

    def put(self, key: str, pack: Dict[str, Any]) -> None:
        payload = json.dumps(pack, separators=(",", ":"))
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lm_cache (key, pack, created_at) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM lm_cache WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM lm_cache")
            conn.commit()


# ----------------------------------------------------------------------
# Backend 3: Redis. For multi-replica deployments where every replica
# should see the same cache (e.g., a load-balanced Kubernetes deploy).
# Imported lazily so installing without [cache] extras still works.
# ----------------------------------------------------------------------
class RedisCache(CacheBackend):
    """Multi-replica cache via Redis. Optional install:
        pip install liquid-memory[cache]
    """

    def __init__(self, url: str, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "redis-py is not installed. Install with: "
                "pip install 'liquid-memory[cache]'"
            ) from exc
        self._r = redis.Redis.from_url(url)
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[CacheEntry]:
        raw = self._r.get(self._k(key))
        if raw is None:
            return None
        obj = json.loads(raw)
        return CacheEntry(pack=obj["pack"], created_at=float(obj["created_at"]))

    def put(self, key: str, pack: Dict[str, Any]) -> None:
        payload = json.dumps(
            {"pack": pack, "created_at": time.time()},
            separators=(",", ":"),
        )
        # SETEX so Redis itself enforces TTL; we don't have to clean up.
        self._r.setex(self._k(key), self._ttl, payload)

    def delete(self, key: str) -> None:
        self._r.delete(self._k(key))

    def clear(self) -> None:
        # SCAN + DEL with our namespace prefix only. We never call
        # FLUSHDB because someone might be sharing the Redis instance.
        for k in self._r.scan_iter(match="lm:cache:*", count=500):
            self._r.delete(k)

    @staticmethod
    def _k(key: str) -> str:
        return f"lm:cache:{key}"


# ----------------------------------------------------------------------
# Public facade. Wraps a backend with TTL enforcement and offers
# nice helpers so the proxy doesn't have to know about backend types.
# ----------------------------------------------------------------------
class CompressionCache:
    """The thing the proxy actually holds onto. Wraps a CacheBackend
    with TTL enforcement, extractor-id keying, and a built-in stats
    counter so the admin can see hit rate over time."""

    def __init__(
        self,
        backend: CacheBackend,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        extractor_id: str = DEFAULT_EXTRACTOR_ID,
    ):
        self._backend = backend
        self._ttl = ttl_seconds
        self._extractor_id = extractor_id
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._lock = threading.Lock()

    def get_pack(self, doc_bytes: bytes) -> Optional[Dict[str, Any]]:
        key = make_key(doc_bytes, self._extractor_id)
        entry = self._backend.get(key)
        if entry is None:
            with self._lock:
                self._misses += 1
            return None
        if entry.is_expired(time.time(), self._ttl):
            # Treat expired entries as misses, and proactively remove
            # so we don't repeatedly fetch + reject.
            self._backend.delete(key)
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
            self._hits += 1
        return entry.pack

    def put_pack(self, doc_bytes: bytes, pack: Dict[str, Any]) -> None:
        key = make_key(doc_bytes, self._extractor_id)
        self._backend.put(key, pack)
        with self._lock:
            self._writes += 1

    def invalidate(self, doc_bytes: bytes) -> None:
        self._backend.delete(make_key(doc_bytes, self._extractor_id))

    def clear(self) -> None:
        self._backend.clear()
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._writes = 0

    def stats(self) -> Dict[str, float]:
        """Snapshot of hit-rate counters. Surface this from a /cache/stats
        admin endpoint (not implemented yet)."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": float(self._hits),
                "misses": float(self._misses),
                "writes": float(self._writes),
                "hit_rate": float(self._hits) / total if total else 0.0,
                "ttl_seconds": float(self._ttl),
            }


def build_cache_from_env() -> CompressionCache:
    """Build the cache instance based on env vars. Called once during
    proxy startup. Env vars:

        LM_CACHE_BACKEND        memory | sqlite | redis | none    (default: memory)
        LM_CACHE_TTL_SECONDS    integer                          (default: 86400)
        LM_CACHE_SQLITE_PATH    path to .db file                 (default: ./lm_cache.db)
        LM_CACHE_REDIS_URL      redis://host:port/db             (default: redis://localhost:6379/0)
        LM_CACHE_MAX_ENTRIES    integer (memory backend only)    (default: 256)
        LM_EXTRACTOR_ID         override the extractor id used   (default: mistral-7b-instruct-v0.3)
                                in the cache key
    """
    backend_kind = os.environ.get("LM_CACHE_BACKEND", "memory").lower()
    ttl = int(os.environ.get("LM_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
    extractor_id = os.environ.get("LM_EXTRACTOR_ID", DEFAULT_EXTRACTOR_ID)

    if backend_kind == "memory":
        max_entries = int(os.environ.get("LM_CACHE_MAX_ENTRIES", "256"))
        backend: CacheBackend = InMemoryCache(max_entries=max_entries)
    elif backend_kind == "sqlite":
        path = os.environ.get("LM_CACHE_SQLITE_PATH", "lm_cache.db")
        backend = SqliteCache(path=path)
    elif backend_kind == "redis":
        url = os.environ.get("LM_CACHE_REDIS_URL", "redis://localhost:6379/0")
        backend = RedisCache(url=url, ttl_seconds=ttl)
    elif backend_kind == "none":
        # No-op cache: every call is a miss. Useful for benchmarking the
        # extractor itself without cache effects polluting the numbers.
        backend = _NoopBackend()
    else:
        raise ValueError(
            f"Unknown LM_CACHE_BACKEND={backend_kind!r}. "
            "Valid: memory, sqlite, redis, none."
        )
    return CompressionCache(backend=backend, ttl_seconds=ttl, extractor_id=extractor_id)


class _NoopBackend(CacheBackend):
    """Cache that does nothing. Useful for `LM_CACHE_BACKEND=none`
    benchmarking runs that want pure-extractor latency without cache
    side effects."""

    def get(self, key: str) -> Optional[CacheEntry]:
        return None

    def put(self, key: str, pack: Dict[str, Any]) -> None:  # noqa: ARG002
        return None

    def delete(self, key: str) -> None:  # noqa: ARG002
        return None

    def clear(self) -> None:
        return None


# ----------------------------------------------------------------------
# Integration TODO (for engine team review before wiring this into
# proxy.py's request handler).
# ----------------------------------------------------------------------
# The intended integration point is at the top of
# HybridProxyService.handle_request(), immediately after the request
# has been parsed and the `large_document` portion has been isolated:
#
#     # New, added by the cache rollout:
#     cached = self.cache.get_pack(large_document.encode("utf-8"))
#     if cached is not None:
#         aggregated_context = cached
#     else:
#         aggregated_context = self.local_extractor.run(large_document)
#         self.cache.put_pack(large_document.encode("utf-8"), aggregated_context)
#     # ...then continue with synthesis as today.
#
# Questions for the engine team before flipping this on:
#
#   1. Is `aggregated_context` actually a JSON-serialisable dict at the
#      point we want to cache it, or is it a richer Python object that
#      would need a __dict__ pickling step?
#   2. The proxy already does extraction-task customisation per request.
#      Should the cache key include extraction_task hash too, or is it
#      stable enough across requests to ignore?
#   3. What's the right size cap per cached pack? In-memory backend is
#      bounded by entry count but not byte size.
#   4. Cache poisoning: should we sign the cached value with HMAC so a
#      compromised SQLite file can't inject a malicious "compressed
#      context" that smuggles instructions into the cloud LLM call?
#
# Until those are answered the cache is implemented and tested in
# isolation but NOT wired into the request hot path. Importing
# liquid_memory.cache from outside the package works today.
# ----------------------------------------------------------------------
