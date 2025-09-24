import pytest
import numpy as np

try:
    import torch
    from torch_geometric.nn import GCNConv  # type: ignore
    TORCH_OK = True
except Exception:
    TORCH_OK = False

import app.model_runner as mr


@pytest.mark.skipif(not TORCH_OK, reason='torch or torch_geometric not installed')
def test_torch_geometric_gcn_with_edge_index():
    # construct small (T,N,1) array
    T, N = 6, 4
    arr = np.zeros((T, N, 1), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            arr[t, n, 0] = t + n * 0.1

    # construct a simple ring edge_index
    edges = []
    for i in range(N):
        edges.append([i, (i + 1) % N])
        edges.append([(i + 1) % N, i])

    # run via run_model with edge_index kwarg
    code, csv_out = mr.run_model('torch_geometric', arr, edge_index=edges, horizon=3)
    assert isinstance(code, str)
    assert isinstance(csv_out, str)
    # should produce at least one row
    assert '\n' in csv_out
