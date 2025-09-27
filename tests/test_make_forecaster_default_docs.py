import numpy as np
import pytest

from app.vector_db import DEFAULT_DOCS
import app.model_runner as mr


def test_make_forecaster_for_default_docs(monkeypatch):
    """For each DEFAULT_DOCS entry, ensure make_forecaster(model_name)
    returns an object with fit/predict and that predict produces an array
    of the requested horizon length.

    We monkeypatch sktime import to raise ImportError so the factory
    uses the local lightweight fallbacks; this keeps the test fast and
    avoids heavy third-party dependencies.
    """
    # Force the internal importlib.import_module('sktime') to raise ImportError
    monkeypatch.setattr(mr.importlib, 'import_module', lambda name: (_ for _ in ()).throw(ImportError()))

    fh = np.arange(1, 4)
    for doc in DEFAULT_DOCS:
        name = doc.get('model_name')
        assert name is not None
        # Should not raise
        forecaster = mr.make_forecaster(name)
        assert hasattr(forecaster, 'fit'), f"forecaster for {name} has no fit"
        assert hasattr(forecaster, 'predict'), f"forecaster for {name} has no predict"

        # Call fit/predict with numpy arrays/iterables
        # Some forecasters accept pandas Series; we use numpy arrays for simplicity
        try:
            forecaster.fit(np.arange(10))
        except Exception as e:
            pytest.fail(f"fit failed for forecaster {name}: {e}")

        try:
            pred = forecaster.predict(fh)
        except Exception as e:
            pytest.fail(f"predict failed for forecaster {name}: {e}")

        arr = np.asarray(pred)
        assert arr.ndim >= 1
        assert arr.shape[0] == len(fh)
