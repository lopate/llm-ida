import numpy as np

from app.model_runner import run_model_from_choice, run_model


def test_run_model_from_choice_sktime_naive():
    arr = np.array([1.0, 2.0, 3.0])
    res = run_model_from_choice('sktime', arr, horizon=2)
    assert isinstance(res, dict)
    assert res['library'] == 'sktime'
    y = res['y_pred']
    assert hasattr(y, 'shape')
    assert y.ravel().shape[0] == 2
    # naive forecaster repeats last value
    assert np.allclose(y.ravel(), np.array([3.0, 3.0]))


def test_run_tslearn_fallback_knn_small_series():
    arr = np.array([1.0, 2.0])
    res = run_model_from_choice('tslearn', arr, horizon=3)
    y = res['y_pred']
    assert y.ravel().shape[0] == 3
    # with too-short series, fallback repeats last value
    assert np.allclose(y.ravel(), np.array([2.0, 2.0, 2.0]))


def test_run_torch_geometric_fallback_extrapolation():
    # provide CSV string input to ensure parsing takes the 1D branch
    csv = "t,value\n0,1\n1,3\n"
    res = run_model_from_choice('torch_geometric', csv, horizon=3)
    y = res['y_pred']
    # fallback linear extrapolation: last=3, delta=2 -> predictions: 5,7,9
    assert y.ravel().shape[0] == 3
    assert np.allclose(y.ravel(), np.array([5.0, 7.0, 9.0]))
