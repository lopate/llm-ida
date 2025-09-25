import pytest
from unittest.mock import Mock, patch

import app.llm as llm
import app.runner_service as runner_service
import app.model_selector as model_selector
from app.vector_db import VectorDB


def test_select_model_rag_uses_passed_db():
    """When a VectorDB instance is passed to select_model_rag, it must call
    that instance's search_by_text method (not a module-level DB).
    """
    db = VectorDB(backend='memory')
    # replace the instance method with a mock to avoid backend details
    db.search_by_text = Mock(return_value=[({'model_name': 'AutoARIMA', 'library': 'sktime', 'description': 'desc', 'id': 'm1'}, 0.9)])

    res = llm.select_model_rag('some dataset', 'forecast', db=db, top_k=1, hf_model_name='')

    # ensure we called the provided db's search_by_text
    assert db.search_by_text.called
    called_args, called_kwargs = db.search_by_text.call_args
    assert 'some dataset' in called_args[0]
    # select_model_rag passes a string query and may pass k as kwarg or not; accept both
    assert (called_kwargs.get('k', 1) == 1) or (len(called_args) > 1 and called_args[1] == 1)

    # result should be a dict (selector output or fallback) and should reflect a selection
    assert isinstance(res, dict)


def test_run_service_forwards_db_to_llm(monkeypatch):
    """run_service should forward the db argument into llm.select_model_rag.
    We'll patch select_model_rag to capture the db passed to it.
    """
    mock_db = Mock()

    captured = {}

    def fake_select_model_rag(dataset_desc, task, db=None, top_k=4, hf_model_name=llm.HF_MODEL_DEFAULT):
        captured['db'] = db
        # return a minimal valid choice dict
        return {'library': 'sktime', 'model_name': 'AutoARIMA'}

    monkeypatch.setattr(llm, 'select_model_rag', fake_select_model_rag)

    # run service with our mock_db
    import numpy as np
    data = np.zeros(10)
    out = runner_service.run_service('ds', data, db=mock_db, task='forecast', horizon=2, hf_model_name='')

    assert 'db' in captured and captured['db'] is mock_db
    assert isinstance(out, dict)
    assert 'choice' in out and 'result' in out


def test_model_selector_init_uses_module_vector_db(monkeypatch):
    """init_vector_db should call the module-level VECTOR_DB.initialize and upsert.
    We monkeypatch VECTOR_DB to a mock to ensure it's used instead of creating a new instance.
    """
    mock_db = Mock()
    monkeypatch.setattr(model_selector, 'VECTOR_DB', mock_db)

    # Ensure embeddings are deterministic and small so upsert args are predictable
    monkeypatch.setattr(model_selector, '_embed_texts', lambda texts: [[0.0] * 64 for _ in texts])

    model_selector.init_vector_db()

    mock_db.initialize.assert_called_once()
    mock_db.upsert.assert_called_once()
    ids_arg, embs_arg, metas_arg = mock_db.upsert.call_args[0]
    assert isinstance(ids_arg, list)
    assert len(ids_arg) == len(model_selector.MODEL_DB)
    assert isinstance(embs_arg, list)
    assert isinstance(metas_arg, list)
