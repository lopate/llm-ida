import sys
import types
import numpy as np
from app.model_runner import run_model_from_choice


def test_tslearn_knn_with_mocked_tslearn(monkeypatch, tmp_path):
    """Test tslearn branch that uses KNeighborsTimeSeriesRegressor when h5py is present."""
    # Create a fake tslearn.neighbors module with KNeighborsTimeSeriesRegressor
    tslearn_mod = types.SimpleNamespace()

    class FakeKNN:
        def __init__(self, n_neighbors=1):
            self.n_neighbors = n_neighbors
            self._fit_X = None
            self._fit_Y = None

        def fit(self, X, Y):
            # store training data; X shape (n_samples, window, 1), Y shape (n_samples, horizon)
            self._fit_X = np.asarray(X)
            self._fit_Y = np.asarray(Y)
            return self

        def predict(self, X):
            # simple behavior: return mean of Y training targets for each sample
            if self._fit_Y is None:
                return np.zeros((np.asarray(X).shape[0], self._fit_Y.shape[1] if self._fit_Y is not None else 1))
            mean = np.nanmean(self._fit_Y, axis=0)
            return np.tile(mean.reshape(1, -1), (np.asarray(X).shape[0], 1))

    tslearn_neighbors = types.SimpleNamespace(KNeighborsTimeSeriesRegressor=FakeKNN)
    sys.modules['tslearn'] = types.ModuleType('tslearn')
    sys.modules['tslearn.neighbors'] = types.ModuleType('tslearn.neighbors')
    setattr(sys.modules['tslearn.neighbors'], 'KNeighborsTimeSeriesRegressor', FakeKNN)

    # ensure importlib_util.find_spec('h5py') returns a non-None to trigger tslearn path
    # monkeypatch the find_spec used in model_runner via importlib.util
    import importlib.util as importlib_util
    monkeypatch.setattr(importlib_util, 'find_spec', lambda name: True if name == 'h5py' else None)

    # build a simple time series 1D array with enough length for window+ horizon
    T = 30
    series = np.arange(T, dtype=np.float32)
    # construct 1D input series -> run_model_from_choice should route to tslearn
    choice = {'library': 'tslearn'}
    res = run_model_from_choice(choice, series, horizon=3, model_args={'n_neighbors':2, 'window_size':5})

    assert res['library'] == 'tslearn'
    # predictions should be a numpy array with shape (horizon,) for 1D input
    y = res['y_pred']
    assert hasattr(y, 'shape')
    assert y.shape[0] == 3

    # cleanup
    sys.modules.pop('tslearn.neighbors', None)
    sys.modules.pop('tslearn', None)

