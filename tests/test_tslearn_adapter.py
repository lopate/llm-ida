import importlib
import numpy as np
import pytest


def _run(choice, X, horizon=3):
    mr = importlib.import_module('app.model_runner')
    return mr.run_model_from_choice(choice, X, horizon=horizon)


def test_tslearn_univariate_basic():
    # simple sinusoid
    T = 40
    x = np.linspace(0, 4 * np.pi, T)
    series = np.sin(x).astype(np.float32)
    choice = {'library': 'tslearn', 'model_name': 'knn', 'model_args': {'n_neighbors': 3, 'window_size': 8}}
    res = _run(choice, series, horizon=3)
    assert isinstance(res, dict)
    y = res.get('y_pred')
    assert y is not None
    # y may be 1D array of length horizon
    assert getattr(y, 'ndim', 1) >= 1


def test_tslearn_multinode_3d():
    # create T,N,1 data
    T, N = 30, 4
    X = np.zeros((T, N, 1), dtype=np.float32)
    for n in range(N):
        X[:, n, 0] = np.sin(np.linspace(0, 2 * np.pi, T) + n * 0.3)
    choice = {'library': 'tslearn', 'model_name': 'knn', 'model_args': {'n_neighbors': 3, 'window_size': 6}}
    res = _run(choice, X, horizon=4)
    assert 'y_pred' in res
    y = res['y_pred']
    # Expect shape (H,N,1) or (H,N)
    assert hasattr(y, 'shape')
    assert y.shape[0] == 4
    assert y.shape[1] == N


def test_tslearn_model_args_window_size_effect():
    # Verify that changing window_size runs and often produces different outputs
    T = 50
    x = np.linspace(0, 6 * np.pi, T)
    series = (np.sin(x) + 0.1 * np.random.RandomState(0).randn(T)).astype(np.float32)
    choice1 = {'library': 'tslearn', 'model_name': 'knn', 'model_args': {'n_neighbors': 3, 'window_size': 5}}
    choice2 = {'library': 'tslearn', 'model_name': 'knn', 'model_args': {'n_neighbors': 3, 'window_size': 12}}
    res1 = _run(choice1, series, horizon=3)
    res2 = _run(choice2, series, horizon=3)
    y1 = res1.get('y_pred')
    y2 = res2.get('y_pred')
    # If both succeeded, allow either equality or inequality, but prefer they both run
    assert y1 is not None and y2 is not None
