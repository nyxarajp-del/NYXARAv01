"""Tests for the scalable vector index + thread-safe memory store."""

from __future__ import annotations

import threading

from nyxara.memory.store import (FaissVectorIndex, MemoryStore, MemoryType, VectorIndex,
                                 make_vector_index)


def test_factory_falls_back_to_numpy_when_faiss_absent():
    idx = make_vector_index(64)
    # faiss is opt-in via config + install; default config selects the exact numpy index
    assert isinstance(idx, (VectorIndex, FaissVectorIndex))
    if not FaissVectorIndex.available():
        assert isinstance(idx, VectorIndex)


def test_faiss_availability_is_honest():
    assert isinstance(FaissVectorIndex.available(), bool)


def test_faiss_backend_when_selected():
    from nyxara.kernel.config import NyxaraSettings, VectorBackend
    s = NyxaraSettings()
    s.memory.vector_backend = VectorBackend.FAISS
    idx = make_vector_index(64, s)
    if FaissVectorIndex.available():
        assert isinstance(idx, FaissVectorIndex)
    else:
        assert isinstance(idx, VectorIndex)


def test_concurrent_remember_and_recall_is_safe():
    ms = MemoryStore()

    def worker(n: int) -> None:
        for i in range(40):
            ms.remember(f"fact {n}-{i} about security and networks",
                        mem_type=MemoryType.SEMANTIC)
            ms.recall("security network", k=3)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ms) == 6 * 40
    # the index stayed consistent with the kv store
    assert len(ms._index) == len(ms)


def test_recall_still_correct_under_lock():
    ms = MemoryStore()
    ms.remember("the owner is Jaypal Khoja, JP", mem_type=MemoryType.SEMANTIC, importance=1.0)
    ms.remember("an unrelated note about weather", mem_type=MemoryType.SEMANTIC)
    hits = ms.recall("who is my master", k=1)
    assert hits and "Jaypal" in hits[0][0].text()
