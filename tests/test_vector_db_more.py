import sys
import types
import json
import numpy as np
import tempfile
import os
import pytest

from app import vector_db


def test_faiss_save_and_load_roundtrip(tmp_path):
    # Fake faiss with read/write index that writes files
    class FakeIndexObj:
        pass

    fake_faiss = types.ModuleType('faiss')
    def fake_write_index(idx, path):
        with open(path, 'wb') as f:
            f.write(b'IDX')
    def fake_read_index(path):
        return FakeIndexObj()
    fake_faiss.write_index = fake_write_index
    fake_faiss.read_index = fake_read_index
    sys.modules['faiss'] = fake_faiss

    try:
        vector_db._backend = 'faiss'
        vector_db._client = {'index': FakeIndexObj(), 'vectors': [], 'metadatas': []}
        p = tmp_path / 'idx'
        idx_path, meta_path = vector_db.save_faiss(str(p))
        assert os.path.exists(idx_path)
        # create meta.json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        # now load via load_faiss
        assert vector_db.load_faiss(str(p)) is True
        assert vector_db._backend == 'faiss'
    finally:
        sys.modules.pop('faiss', None)
        vector_db._backend = None
        vector_db._client = None


def test_chroma_query_distance_mapping():
    # simulate chroma query output shape
    fake_res = {'metadatas': [[{'id':'x'},{'id':'y'}]], 'distances': [[0.2, 0.8]]}
    class FakeCollection:
        def query(self, query_embeddings, n_results):
            return fake_res
    vector_db._backend = 'chroma'
    vector_db._collection = FakeCollection()
    out = vector_db.query([0.1, 0.2], k=2)
    assert isinstance(out, list)
    # distances are mapped to scores as 1 - d
    assert out[0][1] == pytest.approx(0.8)
    assert out[1][1] == pytest.approx(0.2)
