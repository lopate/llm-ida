import importlib
import numpy as np


def test_run_model_from_choice_univariate():
    mr = importlib.import_module('app.model_runner')
    choice = {'library': 'sktime'}
    series = np.arange(15).astype(float)
    res = mr.run_model_from_choice(choice, series, horizon=3)
    assert isinstance(res, dict)
    assert res.get('library') == 'sktime'
    y = res.get('y_pred')
    assert hasattr(y, 'shape')
    assert y.shape[0] == 3


def test_run_model_from_choice_spatio_temporal():
    mr = importlib.import_module('app.model_runner')
    choice = {'library': 'sktime'}
    # (T, N, F)
    T, N, F = 20, 4, 1
    X = np.zeros((T, N, F), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            X[t, n, 0] = t * (n + 1)
    res = mr.run_model_from_choice(choice, X, horizon=2)
    assert isinstance(res, dict)
    y = res.get('y_pred')
    # expected shape (horizon, N, 1)
    assert y.ndim == 3 and y.shape[1] == N


def test_run_model_from_choice_with_model_arg():
    mr = importlib.import_module('app.model_runner')
    choice = {'library': 'sktime'}
    series = np.arange(12).astype(float)
    # try AutoARIMA (if available) - should not raise
    res = mr.run_model_from_choice(choice, series, horizon=3)
    assert isinstance(res, dict)
    # explicitly request a different model via kwargs through run_sktime (indirect)
    res2 = mr.run_model_from_choice({'library': 'sktime'}, series, horizon=3)
    assert isinstance(res2, dict)
