import sys
import types
import importlib
import numpy as np


def _make_fake_torch():
    fake = types.SimpleNamespace()

    class Tensor(np.ndarray):
        pass

    def tensor(x, dtype=None):
        return np.asarray(x)

    def empty(shape, dtype=None):
        return np.zeros(shape, dtype=int)

    fake.tensor = tensor
    fake.empty = empty
    fake.no_grad = lambda: (lambda *a, **k: None)
    fake.long = int
    return fake


def _make_fake_tg():
    tg = types.SimpleNamespace()

    class GCNConv:
        def __init__(self, in_ch, out_ch):
            pass

        def __call__(self, x, edge_index):
            # x: numpy array (N,1) -> return same shape transform
            import numpy as _np
            return _np.asarray(x) * 1.0

    nn = types.SimpleNamespace(GCNConv=GCNConv)
    tg.nn = nn
    return tg


def test_gcn_branch_with_mocks(monkeypatch):
    # create fake torch and torch_geometric
    fake_torch = _make_fake_torch()
    fake_tg = _make_fake_tg()

    # insert into sys.modules
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'torch_geometric', types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'torch_geometric.nn', fake_tg.nn)

    # reload module to pick fake libs
    import app.model_runner as mr
    importlib.reload(mr)

    # build small test array
    T, N = 6, 4
    arr = np.zeros((T, N, 1), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            arr[t, n, 0] = t + n * 0.1

    # run and assert CSV returned
    code, out = mr.run_model('torch_geometric', arr, edge_index=[[0,1],[1,2]], horizon=2)
    assert isinstance(code, str)
    assert isinstance(out, str)
    assert '\n' in out
