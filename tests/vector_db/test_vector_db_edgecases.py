import sys
import types
import os
import json
import math
import numpy as np
import pytest

from app import vector_db


def test_text_to_embedding_encode_raises(monkeypatch):
    # simulate sentence_transformers present but encode raises
    class BadST:
        def __init__(self, name):
            pass
        def get_sentence_embedding_dimension(self):
            return 4
        def encode(self, texts, show_progress_bar=False):
            raise RuntimeError('encode failed')

    mod = types.ModuleType('sentence_transformers')
    mod.SentenceTransformer = BadST
    sys.modules['sentence_transformers'] = mod
    try:
        vector_db.__dict__.pop('_st_model', None)
        vector_db.__dict__.pop('_st_encoder_dim', None)
        v = vector_db._text_to_embedding('some text', dim=8)
        # fallback should produce a normalized vector
        assert isinstance(v, list)
        assert len(v) == 8
        assert math.isclose(sum(x*x for x in v) ** 0.5, 1.0, rel_tol=1e-6)
    finally:
        sys.modules.pop('sentence_transformers', None)
        vector_db.__dict__.pop('_st_model', None)
        vector_db.__dict__.pop('_st_encoder_dim', None)


def test_text_to_embedding_non_array_return(monkeypatch):
    # simulate ST returns a python list (not numpy), but with zero norm
    class ZeroST:
        def __init__(self, name):
            pass
        def get_sentence_embedding_dimension(self):
            return 4
        def encode(self, texts, show_progress_bar=False):
            return [[0.0, 0.0, 0.0, 0.0]]

    mod = types.ModuleType('sentence_transformers')
    mod.SentenceTransformer = ZeroST
    sys.modules['sentence_transformers'] = mod
    try:
        vector_db.__dict__.pop('_st_model', None)
        vector_db.__dict__.pop('_st_encoder_dim', None)
        v = vector_db._text_to_embedding('empty', dim=4)
        # should fall back to list-handling branch and return normalized (zero -> zeros)
        assert isinstance(v, list)
        assert len(v) == 4
        assert all(x == 0.0 for x in v)
    finally:
        sys.modules.pop('sentence_transformers', None)
        vector_db.__dict__.pop('_st_model', None)
        vector_db.__dict__.pop('_st_encoder_dim', None)


def test_faiss_autosave_flag_triggers_save(monkeypatch, tmp_path):
    # Mock faiss with index that supports write_index/read_index and test autosave path
    class FakeIndex:
        def __init__(self, d):
            self.d = d
            self.data = []
        def add(self, vecs):
            self.data.extend(vecs.tolist())
        def search(self, q, k):
            import numpy as _np
            return _np.array([[1.0]]), _np.array([[0]])

    fake_faiss = types.ModuleType('faiss')
    fake_faiss.IndexFlatIP = FakeIndex
    def fake_write_index(idx, path):
        with open(path, 'wb') as f:
            f.write(b'IDX')
    fake_faiss.write_index = fake_write_index
    sys.modules['faiss'] = fake_faiss

    try:
        vector_db._backend = 'faiss'
        vector_db._client = {'index': None, 'vectors': [], 'metadatas': []}
        # configure module-level autosave/path so upsert will call save_faiss
        vector_db._faiss_index_path = str(tmp_path / 'idx')
        vector_db._faiss_autosave = True
        # set env as well for completeness
        monkeypatch.setenv('FAISS_AUTOSAVE', '1')
        monkeypatch.setenv('FAISS_PATH', str(tmp_path / 'idx'))

        # call upsert to trigger autosave path (index will be created and save_faiss invoked)
        ids = ['z']
        embs = [[1.0, 0.0, 0.0]]
        metas = [{'id':'z'}]
        assert vector_db.upsert(ids, embs, metas) is True
        # save file should exist
        assert (tmp_path / 'idx.index').exists()
    finally:
        sys.modules.pop('faiss', None)
        vector_db._backend = None
        vector_db._client = None
        os.environ.pop('FAISS_AUTOSAVE', None)
        os.environ.pop('FAISS_PATH', None)


def test_faiss_save_error_raises_when_no_index():
    # when backend != faiss or index None, save_faiss should raise
    vector_db._backend = 'memory'
    with pytest.raises(RuntimeError):
        vector_db.save_faiss('nope')
    vector_db._backend = 'faiss'
    vector_db._client = None
    with pytest.raises(RuntimeError):
        vector_db.save_faiss('nope')
    vector_db._backend = None
    vector_db._client = None
