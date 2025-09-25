import importlib
import sys
import types
import numpy as np


def test_text_to_embedding_hash_fallback():
    vdb = importlib.import_module('app.vector_db')
    # ensure no sentence-transformers model present
    if '_st_model' in vdb.__dict__:
        vdb._st_model = None
    # deterministic embedding for given text
    emb1 = vdb._text_to_embedding('alpha beta gamma', dim=16)
    emb2 = vdb._text_to_embedding('alpha beta gamma', dim=16)
    assert isinstance(emb1, list)
    assert len(emb1) == 16
    # deterministic same text -> same embedding
    assert np.allclose(np.array(emb1), np.array(emb2))


def test_text_to_embedding_with_mocked_sentence_transformers(monkeypatch):
    vdb = importlib.import_module('app.vector_db')

    class FakeModel:
        def __init__(self, dim=8):
            self._dim = dim

        def encode(self, texts, show_progress_bar=False):
            # return simple numeric embeddings
            out = []
            for t in texts:
                vec = np.array([float(len(t) + i) for i in range(self._dim)], dtype=float)
                out.append(vec)
            return np.array(out)

    # monkeypatch the lazy loader globals
    monkeypatch.setitem(vdb.__dict__, '_st_model', FakeModel(dim=8))
    monkeypatch.setitem(vdb.__dict__, '_st_encoder_dim', 8)

    emb = vdb._text_to_embedding('hello world', dim=8)
    assert isinstance(emb, list)
    assert len(emb) == 8
    # check normalization (norm <= 1 + small eps)
    assert abs(np.linalg.norm(np.array(emb)) - 1.0) < 1e-6


def test_call_transformers_extracts_json_marked_and_balanced(monkeypatch):
    # Use the extractor directly to validate JSON extraction logic without mocking transformers
    llm = importlib.import_module('app.llm')

    # Case 1: marked JSON
    text1 = 'prefix <<JSON>>{"library":"marked"}<</JSON>> suffix'
    out1 = llm.extract_json_from_text(text1)
    assert isinstance(out1, dict)
    assert out1.get('library') == 'marked'

    # Case 2: balanced JSON extraction
    text2 = 'some text {"library":"bal","model_choice":"z"} trailing'
    out2 = llm.extract_json_from_text(text2)
    assert isinstance(out2, dict)
    assert out2.get('library') == 'bal'
