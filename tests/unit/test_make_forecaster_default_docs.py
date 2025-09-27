import importlib
import numpy as np
import pytest

from app.vector_db import DEFAULT_DOCS
from app.model_runner import make_forecaster


def test_make_forecaster_for_default_docs(monkeypatch):
    """Ensure make_forecaster does not raise for any model_name in DEFAULT_DOCS.

    We mock out `importlib.import_module('sktime')` to raise ImportError so the
    function uses its safe fallback implementation. The test then calls fit
    and predict on a short univariate series to ensure the returned object
    behaves like a forecaster.
    """
    # Force make_forecaster to take the lightweight fallback path
    monkeypatch.setattr(importlib, 'import_module', lambda name: (_ for _ in ()).throw(ImportError()))

    # small univariate series
    y = np.arange(10, dtype=np.float32)
    fh = np.arange(1, 4)

    for doc in DEFAULT_DOCS:
        model_name = doc.get('model_name')
        # Should not raise
        forecaster = make_forecaster(model_name)
        # forecaster should have fit and predict
        assert hasattr(forecaster, 'fit') and hasattr(forecaster, 'predict')
        # fit and predict should work on simple data
        fitted = forecaster.fit(y)
        assert fitted is not None
        pred = forecaster.predict(fh)
        # predictions should be array-like and length matches fh
        arr = np.asarray(pred)
        assert arr.ndim >= 1
        assert arr.shape[0] == len(fh)
