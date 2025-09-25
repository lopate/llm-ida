import sys
import types
import numpy as np
from app.model_runner import run_torch_geometric, run_model_from_choice


class DummyTensor(np.ndarray):
    # minimal wrapper to mimic torch tensor interface used in model_runner
    def numpy(self):
        return np.asarray(self)


def make_torch_module():
    torch = types.SimpleNamespace()

    def tensor(x, dtype=None):
        arr = np.asarray(x)
        # cast to numpy array; return object with .numpy()
        return arr

    class nn:
        class Module:
            def __init__(self):
                pass

        class Linear:
            def __init__(self, in_f, out_f):
                pass

            def __call__(self, x):
                return x

        class MSELoss:
            def __init__(self):
                pass

            def __call__(self, a, b):
                return np.mean((a - b) ** 2)

    class optim:
        class Adam:
            def __init__(self, params, lr=0.01):
                pass

            def zero_grad(self):
                pass

            def step(self):
                pass

    torch.tensor = tensor
    torch.nn = nn
    torch.optim = optim
    torch.no_grad = lambda : _NoGrad()

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    # minimal torch functions used
    torch.relu = lambda x: np.maximum(0, x)

    return torch


class FakeGCNConv:
    def __init__(self, in_f, out_f):
        self.in_f = in_f
        self.out_f = out_f

    def __call__(self, x, edge_index=None):
        # simulate a simple linear aggregation: sum neighbors if edge_index provided
        arr = np.asarray(x)
        if edge_index is None:
            return arr
        try:
            ei = np.asarray(edge_index)
            if ei.ndim == 2 and ei.shape[0] == 2:
                # aggregate by simple mean for each node
                N = arr.shape[0]
                out = np.zeros((N, self.out_f)) if arr.ndim == 2 else np.zeros_like(arr)
                # return arr itself simplified
                return arr
        except Exception:
            return arr
        return arr


def test_torch_geometric_inference(monkeypatch):
    # Insert fake torch and torch_geometric modules
    torch_mod = make_torch_module()
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch'].tensor = torch_mod.tensor
    sys.modules['torch'].nn = torch_mod.nn
    sys.modules['torch'].optim = torch_mod.optim
    sys.modules['torch'].no_grad = torch_mod.no_grad
    sys.modules['torch'].relu = torch_mod.relu

    # fake torch_geometric.nn.GCNConv
    tg_nn = types.ModuleType('torch_geometric.nn')
    tg_nn.GCNConv = FakeGCNConv
    sys.modules['torch_geometric'] = types.ModuleType('torch_geometric')
    sys.modules['torch_geometric.nn'] = tg_nn

    # construct a small 3D array (T, N, F)
    T, N = 4, 3
    arr = np.arange(T * N, dtype=np.float32).reshape(T, N, 1)
    # run inference path (no train flag)
    res = run_model_from_choice('torch_geometric', arr, horizon=2)
    assert res['library'] == 'torch_geometric'
    y = res['y_pred']
    # expectation: returned array should be shape (horizon, N, 1)
    assert y.shape[0] == 2
    assert y.shape[1] == N

    # cleanup
    sys.modules.pop('torch_geometric.nn', None)
    sys.modules.pop('torch_geometric', None)
    sys.modules.pop('torch', None)


def test_torch_geometric_train(monkeypatch):
    # Similar to inference but request training via model_args
    torch_mod = make_torch_module()
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch'].tensor = torch_mod.tensor
    sys.modules['torch'].nn = torch_mod.nn
    sys.modules['torch'].optim = torch_mod.optim
    sys.modules['torch'].no_grad = torch_mod.no_grad
    sys.modules['torch'].relu = torch_mod.relu

    tg_nn = types.ModuleType('torch_geometric.nn')
    tg_nn.GCNConv = FakeGCNConv
    sys.modules['torch_geometric'] = types.ModuleType('torch_geometric')
    sys.modules['torch_geometric.nn'] = tg_nn

    # create array with T>=2
    T, N = 5, 2
    arr = np.arange(T * N, dtype=np.float32).reshape(T, N, 1)
    res = run_model_from_choice('torch_geometric', arr, horizon=1, model_args={'train': True, 'train_epochs': 1})
    assert res['library'] == 'torch_geometric'
    y = res['y_pred']
    assert y.shape[0] == 1

    sys.modules.pop('torch_geometric.nn', None)
    sys.modules.pop('torch_geometric', None)
    sys.modules.pop('torch', None)
