import os
from pathlib import Path
import pytest
import numpy as np

from app.vector_db import VectorDB
from app import llm
from app import model_selector
from app.runner_service import run_service


def test_integration_run_service_with_shared_db(tmp_path):
    # shared DB instance used by both selector and run_service
    db = VectorDB(backend='memory')
    db.initialize()
    # seed a custom doc so selector can find a match
    ids = ['int_test_m1']
    embs = [[0.0] * 128]
    metas = [{'model_name': 'AutoARIMA', 'library': 'sktime', 'description': 'univariate daily seasonality', 'id': 'int_test_m1'}]
    assert db.upsert(ids, embs, metas)

    # pass db into select_model_rag and ensure returned dict mentions sktime
    sel = llm.select_model_rag('daily seasonality', 'forecast', db=db, hf_model_name='')
    assert isinstance(sel, dict)
    assert 'library' in sel and 'sktime' in str(sel['library']).lower()

    # run_service should accept same db and produce a result structure
    uni = np.arange(100) * 0.0
    out = run_service('daily seasonality', uni, db=db, task='forecast', horizon=5, hf_model_name='')
    assert 'choice' in out and 'result' in out and 'logs' in out


def test_model_selector_init_uses_explicit_db(monkeypatch):
    # Create a fresh VectorDB and monkeypatch the module-level VECTOR_DB so init_vector_db uses ours
    db = VectorDB(backend='memory')
    db.initialize()
    monkeypatch.setattr(model_selector, 'VECTOR_DB', db)

    # ensure init_vector_db doesn't raise and upserts into our db
    model_selector.init_vector_db()
    # verify memory store has content
    assert len(db._memory_ids) > 0


def test_select_model_rag_handles_non_vector_db_objects():
    # If a non-VectorDB object is passed, select_model_rag should fall back to an internal transient DB
    class Dummy:
        pass

    res = llm.select_model_rag('something', 'forecast', db=Dummy(), hf_model_name='')
    assert isinstance(res, dict)
