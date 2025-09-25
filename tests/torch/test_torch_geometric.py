import numpy as np

from app.model_runner import run_model_from_choice


def test_run_torch_geometric_fallback_extrapolation():
    # provide CSV string input to ensure parsing takes the 1D branch
    csv = "t,value\n0,1\n1,3\n"
    res = run_model_from_choice('torch_geometric', csv, horizon=3)
    y = res['y_pred']
    # fallback linear extrapolation: last=3, delta=2 -> predictions: 5,7,9
    assert y.ravel().shape[0] == 3
    assert np.allclose(y.ravel(), np.array([5.0, 7.0, 9.0]))
