import numpy as np
import types
import sys
import pytest

from app import model_runner


def test_run_sktime_autoarima_monkeypatched_make_forecaster():
    # prepare 1D array
    arr = np.array([1.0, 2.0, 3.0, 4.0])

    class DummyForecaster:
        def fit(self, y):
            self._y = np.asarray(y)
            return self

        def predict(self, fh):
            # return constant predictions matching fh length
            import numpy as _np
            return _np.full(len(fh), 42.0, dtype=_np.float32)

    # monkeypatch make_forecaster to observe arguments
    captured = {}

    def fake_make_forecaster(name, args=None):
        captured['name'] = name
        captured['args'] = args
        return DummyForecaster()

    orig = model_runner.make_forecaster
    try:
        model_runner.make_forecaster = fake_make_forecaster
        res = model_runner.run_model_from_choice('sktime', arr, horizon=3, model_name='AutoARIMA')
        assert res['library'] == 'sktime'
        y = res['y_pred']
        assert y.ravel().shape[0] == 3
        assert all(float(v) == 42.0 for v in y.ravel())
        assert captured['name'].lower().startswith('auto')
    finally:
        model_runner.make_forecaster = orig


def test_run_sktime_forest_multinode_with_make_forecaster():
    # 3D array T,N,1
    T, N = 6, 3
    arr = np.zeros((T, N, 1), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            arr[t, n, 0] = t + n

    class NodeForecaster:
        def fit(self, y):
            self._y = np.asarray(y)
            return self

        def predict(self, fh):
            import numpy as _np
            # produce sequence 1..len(fh)
            return _np.arange(1, len(fh) + 1, dtype=_np.float32)

    seen = {}

    def fake_make_forecaster(name, args=None):
        seen['name'] = name
        seen['args'] = args
        return NodeForecaster()

    orig = model_runner.make_forecaster
    try:
        model_runner.make_forecaster = fake_make_forecaster
        res = model_runner.run_model_from_choice('sktime', arr, horizon=4, model_name='forest')
        assert res['library'] == 'sktime'
        y = res['y_pred']
        # shape should be (horizon, N, 1) as numpy from parser
        assert hasattr(y, 'shape')
        # ensure predictions are present for horizon*N entries when flattened
        assert y.ravel().shape[0] >= 4 * N
        assert seen['name'] is not None
    finally:
        model_runner.make_forecaster = orig
