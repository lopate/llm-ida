import numpy as np
from app.model_runner import run_model


def _assert_no_nan_in_csv(csv_text: str):
    # simple check: ensure 'nan' substring not present and numeric parse yields no NaN
    assert 'nan' not in csv_text.lower()


def test_sktime_no_nan_on_univariate():
    series = np.linspace(0, 1, 50).astype(float)
    code, csv = run_model('sktime', series, horizon=5)
    _assert_no_nan_in_csv(csv)


def test_tslearn_no_nan_on_multivariate():
    # create T x N array
    T, N = 60, 5
    arr = np.vstack([np.sin(np.linspace(0, 2 * np.pi, T)) + 0.1 * np.random.randn(T) for _ in range(N)]).T
    code, csv = run_model('tslearn', arr, horizon=4, model_args={'n_neighbors': 2, 'window_size': 6})
    _assert_no_nan_in_csv(csv)


def test_torch_geometric_no_nan_on_sensor_network():
    # small sensor network T x S
    T, S = 40, 6
    rng = np.random.default_rng(0)
    sensors = rng.normal(size=(T, S)).astype(float)
    # build simple ring edge_index
    edges = []
    for i in range(S):
        edges.append((i, (i+1) % S))
        edges.append(((i+1) % S, i))
    edge_index = np.array(edges).T.astype(int)
    code, csv = run_model('torch_geometric', sensors, horizon=3, edge_index=edge_index)
    _assert_no_nan_in_csv(csv)
