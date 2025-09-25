import os
import sys
import json
import types
import tempfile
import numpy as np

import pytest

from app import vector_db


def test_init_chroma_and_seed(monkeypatch, tmp_path):
    # create fake chromadb module with Client and collection behavior
    class FakeCollection:
        def __init__(self):
            self.upserted = []
            self.persisted = False

        def upsert(self, ids, embeddings, metadatas):
            self.upserted.append((ids, embeddings, metadatas))

        def persist(self):
            self.persisted = True

    class FakeClient:
        def __init__(self):
            self._col = FakeCollection()

        def get_collection(self, name=None):
            return self._col

        def create_collection(self, name=None):
            self._col = FakeCollection()
            return self._col

        def persist(self):
            # simulate client-level persist
            if hasattr(self._col, 'persist'):
                self._col.persist()

    fake_chroma = types.ModuleType('chromadb')
    fake_chroma.Client = FakeClient
    sys.modules['chromadb'] = fake_chroma

    try:
        # force reinitialize
        vector_db._backend = None
        vector_db._client = None
        vector_db._collection = None

        res = vector_db.initialize()
        assert res is True
        assert vector_db._backend == 'chroma'
        # initialize does not auto-seed chroma; call seed explicitly
        assert hasattr(vector_db._collection, 'upsert')
        # seed_default_models should upsert into the (possibly re-created) collection
        assert vector_db.seed_default_models() is True
        col = vector_db._collection
        assert hasattr(col, 'upserted')
        assert col.upserted  # not empty
        # test persist attempted
        assert col.persisted is True
    finally:
        sys.modules.pop('chromadb', None)
        # reset backend to avoid leaking into other tests
        vector_db._backend = None
        vector_db._client = None
        vector_db._collection = None


def test_faiss_upsert_query_and_save_load(monkeypatch, tmp_path):
    # Create a fake faiss module with IndexFlatIP implementation
    class FakeIndex:
        def __init__(self, d):
            self.d = d
            self.data = []

        def add(self, vecs):
            # accept numpy arrays
            self.data.extend(vecs.tolist())

        def search(self, q, k):
            # naive: compute dot product with stored data
            import numpy as _np
            qv = _np.array(q).reshape(1, -1)
            D = []
            I = []
            for stored in self.data:
                D.append(float(sum(a*b for a, b in zip(stored, qv.ravel()))))
                I.append(len(I))
            # return top k
            if not D:
                return _np.array([[]], dtype=float), _np.array([[]], dtype=int)
            import numpy as _np
            arr = _np.array(D)
            idxs = arr.argsort()[::-1][:k]
            return arr[idxs].reshape(1, -1), idxs.reshape(1, -1)

    fake_faiss = types.ModuleType('faiss')
    fake_faiss.IndexFlatIP = FakeIndex
    def fake_write_index(idx, path):
        # write a small representation
        with open(path, 'wb') as f:
            f.write(b'FAKEIDX')
    def fake_read_index(path):
        class Idx:
            pass
        return Idx()
    fake_faiss.write_index = fake_write_index
    fake_faiss.read_index = fake_read_index

    sys.modules['faiss'] = fake_faiss

    try:
        # set backend to faiss via init
        vector_db._backend = None
        vector_db._client = None
        vector_db._faiss_index_path = None
        vector_db._faiss_autosave = False
        assert vector_db._init_faiss() is True
        assert vector_db._backend == 'faiss'
        # upsert two vectors
        ids = ['i1', 'i2']
        embs = [[1.0, 0.0], [0.0, 1.0]]
        metas = [{'id':'i1'},{'id':'i2'}]
        assert vector_db.upsert(ids, embs, metas) is True
        # query first vector
        res = vector_db.query([1.0, 0.0], k=1)
        assert res and isinstance(res, list)
        md, score = res[0]
        assert isinstance(md, dict)
        # test save_faiss writes files
        p = tmp_path / 'faiss_test'
        idx_path, meta_path = vector_db.save_faiss(str(p))
        assert idx_path.endswith('.index')
        assert meta_path.endswith('.meta.json')
        # ensure metadata file exists
        assert (p.with_suffix('.meta.json')).exists()
        # test load_faiss raises FileNotFound when not present (cleanup then test)
        # remove files and expect FileNotFoundError
        (p.with_suffix('.index')).unlink()
        (p.with_suffix('.meta.json')).unlink()
        with pytest.raises(FileNotFoundError):
            vector_db.load_faiss(str(p))
    finally:
        sys.modules.pop('faiss', None)
        vector_db._backend = None
        vector_db._client = None
        vector_db._faiss_index_path = None
        vector_db._faiss_autosave = False


def test_search_by_text_and_query_scores():
    # ensure memory backend and simple search works
    vector_db._backend = 'memory'
    vector_db._client = {"vectors": [], "metadatas": [], "ids": []}
    # upsert two orthogonal vectors
    ids = ['a','b']
    embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    metas = [{'id':'a'}, {'id':'b'}]
    assert vector_db.upsert(ids, embs, metas) is True
    res = vector_db.search_by_text('foo bar baz', k=2)
    # should return 2 results (may be low similarity but deterministic)
    assert isinstance(res, list)
    assert len(res) == 2

