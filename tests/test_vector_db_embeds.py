import sys
import types
import math
from unittest import mock

import pytest

from app import vector_db


def test_text_to_embedding_hash_fallback_empty():
    # empty text -> zero vector
    v = vector_db._text_to_embedding("", dim=16)
    assert isinstance(v, list)
    assert len(v) == 16
    assert all(x == 0.0 for x in v)


def test_text_to_embedding_hash_fallback_basic():
    v1 = vector_db._text_to_embedding("hello world", dim=16)
    v2 = vector_db._text_to_embedding("hello world", dim=16)
    # deterministic
    assert v1 == v2
    assert math.isclose(sum(x * x for x in v1) ** 0.5, 1.0, rel_tol=1e-6)


def test_text_to_embedding_with_sentence_transformers(monkeypatch):
    # Mock a minimal SentenceTransformer-like object
    class DummyST:
        def __init__(self, name):
            self._dim = 8

        def get_sentence_embedding_dimension(self):
            return self._dim

        def encode(self, texts, show_progress_bar=False):
            # return reproducible embeddings: length _dim filled with incremental values
            return [[float(i + 1) for i in range(self._dim)] for _ in texts]

    # Ensure any previously cached model is cleared
    vector_db.__dict__.pop("_st_model", None)
    vector_db.__dict__.pop("_st_encoder_dim", None)

    monkeypatch.setenv("ST_MODEL", "dummy")

    # Inject a fake sentence_transformers module into sys.modules
    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = DummyST
    sys.modules["sentence_transformers"] = fake_mod
    try:
        emb = vector_db._text_to_embedding("the quick brown fox", dim=8)
        assert isinstance(emb, list)
        assert len(emb) == 8
        # normalized vector: all positive values
        assert all(x >= 0 for x in emb)
    finally:
        # cleanup
        sys.modules.pop("sentence_transformers", None)
        vector_db.__dict__.pop("_st_model", None)
        vector_db.__dict__.pop("_st_encoder_dim", None)


def test_memory_upsert_and_query():
    # force memory backend
    vector_db._backend = "memory"
    vector_db._ensure_memory_client()
    vector_db._client = {"vectors": [], "metadatas": [], "ids": []}

    ids = ["a", "b"]
    embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    metas = [{"id": "a", "model_name": "m1"}, {"id": "b", "model_name": "m2"}]

    ok = vector_db.upsert(ids, embs, metas)
    assert ok is True

    # query near first vector
    res = vector_db.query([1.0, 0.0, 0.0], k=1)
    assert len(res) == 1
    md, score = res[0]
    assert md.get("id") == "a"
    assert score > 0.9

    # query when no vectors
    # clear
    vector_db._client = {"vectors": [], "metadatas": [], "ids": []}
    res = vector_db.query([1.0, 0.0, 0.0], k=1)
    assert res == []
