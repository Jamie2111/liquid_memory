"""Unit tests for liquid_memory.cache.

Run with:
    python -m pytest tests/test_cache.py -v

These tests cover the backend contract (memory, sqlite, noop), the
CompressionCache facade's TTL + stats behaviour, key derivation, and
the env-var factory. RedisCache is NOT tested here because it would
require a live Redis instance; we leave that to integration CI.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Make the package importable when running pytest from any CWD.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from liquid_memory.cache import (  # noqa: E402
    CompressionCache,
    InMemoryCache,
    SqliteCache,
    _NoopBackend,
    build_cache_from_env,
    make_key,
)


# ---------------------------------------------------------------------
# make_key: stable, deterministic, extractor-sensitive
# ---------------------------------------------------------------------

def test_make_key_is_deterministic():
    a = make_key(b"hello world")
    b = make_key(b"hello world")
    assert a == b


def test_make_key_changes_with_document():
    a = make_key(b"document one")
    b = make_key(b"document two")
    assert a != b


def test_make_key_changes_with_extractor_id():
    a = make_key(b"same doc", extractor_id="ext-v1")
    b = make_key(b"same doc", extractor_id="ext-v2")
    assert a != b


def test_make_key_is_hex_sha256():
    k = make_key(b"x")
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------
# InMemoryCache: LRU eviction + thread-safety
# ---------------------------------------------------------------------

def test_in_memory_roundtrip():
    c = InMemoryCache(max_entries=4)
    c.put("k1", {"value": 1})
    entry = c.get("k1")
    assert entry is not None
    assert entry.pack == {"value": 1}


def test_in_memory_miss_returns_none():
    c = InMemoryCache()
    assert c.get("nope") is None


def test_in_memory_lru_evicts_oldest():
    c = InMemoryCache(max_entries=2)
    c.put("a", {"v": 1})
    c.put("b", {"v": 2})
    # `a` should still be there
    assert c.get("a") is not None
    # access `a` to bump it to most-recent
    c.put("c", {"v": 3})  # this should evict `b` (least recently used)
    assert c.get("a") is not None
    assert c.get("b") is None
    assert c.get("c") is not None


def test_in_memory_delete_and_clear():
    c = InMemoryCache()
    c.put("k", {"v": 1})
    c.delete("k")
    assert c.get("k") is None
    c.put("k", {"v": 1})
    c.clear()
    assert c.get("k") is None


# ---------------------------------------------------------------------
# SqliteCache: persistence across instances + same contract
# ---------------------------------------------------------------------

def test_sqlite_roundtrip(tmp_path):
    db = tmp_path / "cache.db"
    c1 = SqliteCache(path=str(db))
    c1.put("k1", {"value": 42})
    # New instance over the same file: should still see the row.
    c2 = SqliteCache(path=str(db))
    entry = c2.get("k1")
    assert entry is not None
    assert entry.pack == {"value": 42}


def test_sqlite_miss(tmp_path):
    c = SqliteCache(path=str(tmp_path / "x.db"))
    assert c.get("nope") is None


def test_sqlite_put_replaces(tmp_path):
    c = SqliteCache(path=str(tmp_path / "x.db"))
    c.put("k", {"v": 1})
    c.put("k", {"v": 2})  # overwrite
    entry = c.get("k")
    assert entry.pack == {"v": 2}


def test_sqlite_delete_and_clear(tmp_path):
    c = SqliteCache(path=str(tmp_path / "x.db"))
    c.put("k", {"v": 1})
    c.delete("k")
    assert c.get("k") is None
    c.put("k", {"v": 1})
    c.clear()
    assert c.get("k") is None


# ---------------------------------------------------------------------
# CompressionCache facade: TTL enforcement + stats
# ---------------------------------------------------------------------

def test_facade_caches_and_serves():
    backend = InMemoryCache()
    cache = CompressionCache(backend=backend, ttl_seconds=60)
    doc = b"a long document"

    assert cache.get_pack(doc) is None
    cache.put_pack(doc, {"facts": ["x", "y"]})
    assert cache.get_pack(doc) == {"facts": ["x", "y"]}


def test_facade_ttl_expires(monkeypatch):
    backend = InMemoryCache()
    cache = CompressionCache(backend=backend, ttl_seconds=10)
    doc = b"some doc"

    # Patch time.time as seen inside the cache module so the entry
    # appears to have been written 100s ago.
    cache.put_pack(doc, {"v": 1})
    import liquid_memory.cache as cache_mod
    real_time = cache_mod.time.time
    monkeypatch.setattr(cache_mod.time, "time", lambda: real_time() + 100)

    assert cache.get_pack(doc) is None


def test_facade_stats_tracks_hits_misses_writes():
    cache = CompressionCache(backend=InMemoryCache(), ttl_seconds=60)
    doc = b"doc"

    cache.get_pack(doc)        # miss
    cache.put_pack(doc, {"v": 1})  # write
    cache.get_pack(doc)        # hit
    cache.get_pack(doc)        # hit

    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["writes"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_facade_invalidate():
    cache = CompressionCache(backend=InMemoryCache(), ttl_seconds=60)
    doc = b"doc"
    cache.put_pack(doc, {"v": 1})
    cache.invalidate(doc)
    assert cache.get_pack(doc) is None


def test_facade_clear_resets_stats():
    cache = CompressionCache(backend=InMemoryCache(), ttl_seconds=60)
    doc = b"doc"
    cache.put_pack(doc, {"v": 1})
    cache.get_pack(doc)
    cache.clear()
    s = cache.stats()
    assert s["hits"] == 0 and s["misses"] == 0 and s["writes"] == 0


# ---------------------------------------------------------------------
# build_cache_from_env: each backend kind + bad value error
# ---------------------------------------------------------------------

def test_env_factory_default_is_memory(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LM_CACHE_") or k == "LM_EXTRACTOR_ID":
            monkeypatch.delenv(k, raising=False)
    cache = build_cache_from_env()
    assert isinstance(cache._backend, InMemoryCache)


def test_env_factory_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("LM_CACHE_BACKEND", "sqlite")
    monkeypatch.setenv("LM_CACHE_SQLITE_PATH", str(tmp_path / "lm.db"))
    cache = build_cache_from_env()
    assert isinstance(cache._backend, SqliteCache)


def test_env_factory_noop(monkeypatch):
    monkeypatch.setenv("LM_CACHE_BACKEND", "none")
    cache = build_cache_from_env()
    assert isinstance(cache._backend, _NoopBackend)
    # Noop never returns a hit
    cache.put_pack(b"doc", {"v": 1})
    assert cache.get_pack(b"doc") is None


def test_env_factory_unknown_raises(monkeypatch):
    monkeypatch.setenv("LM_CACHE_BACKEND", "rocksdb-please-no")
    with pytest.raises(ValueError, match="Unknown LM_CACHE_BACKEND"):
        build_cache_from_env()


def test_env_factory_respects_extractor_id(monkeypatch):
    monkeypatch.setenv("LM_EXTRACTOR_ID", "custom-extractor-v9")
    cache = build_cache_from_env()
    # The same doc with default extractor would key differently
    k_custom = make_key(b"doc", "custom-extractor-v9")
    k_default = make_key(b"doc")
    assert k_custom != k_default
    # And the cache uses the env one internally
    cache.put_pack(b"doc", {"v": 1})
    # Look up via the same env extractor: hit
    assert cache.get_pack(b"doc") == {"v": 1}
