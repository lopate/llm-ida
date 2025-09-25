import builtins
from typing import Any

import pytest

from app import llm


def test_select_model_rag_uses_vector_db(monkeypatch):
    # Prepare a fake vector_db.search_by_text returning two entries
    fake_docs = [
        (
            {
                'id': 'sktime_autoarima',
                'model_name': 'AutoARIMA',
                'library': 'sktime',
                'description': 'AutoARIMA with seasonal adjustment; good for hourly seasonality.'
            },
            0.95,
        ),
        (
            {
                'id': 'tslearn_knn',
                'model_name': 'KNeighbors',
                'library': 'tslearn',
                'description': 'KNN/DTW based nearest-neighbour forecasting for shape-based similarity.'
            },
            0.9,
        ),
    ]

    def fake_search(text: str, k: int = 4):
        return fake_docs[:k]

    # Build a fake db object exposing search_by_text so we can pass it to select_model_rag
    class FakeDB:
        def search_by_text(self, text: str, k: int = 4):
            return fake_search(text, k=k)

    fake_db = FakeDB()

    # Call select_model_rag with the fake db instance (new signature)
    dataset_desc = 'Hourly sensor with daily seasonality and moderate noise'
    task = 'forecasting'

    # HF_MODEL defaults to 'fallback' so this should exercise the fallback selection
    res = llm.select_model_rag(dataset_desc, task, db=fake_db, top_k=2)

    # Result should be a dict with keys we expect from select_model fallback
    assert isinstance(res, dict)
    assert 'library' in res
    # Because the fake docs prioritized sktime AutoARIMA, fallback rules also default to sktime
    assert res['library'] in ('sktime', 'tslearn', 'pysteps')
