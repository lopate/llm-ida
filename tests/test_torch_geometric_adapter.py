import csv
from io import StringIO
import numpy as np
import app.model_runner as mr


def _parse_csv(s: str):
    f = StringIO(s)
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
    return header, rows


def test_torch_geometric_node_output():
    # Create a small (T,N,1) array and pass directly to run_model helper
    T, N = 6, 3
    arr = np.zeros((T, N, 1), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            arr[t, n, 0] = t + n * 0.1

    code, csv_out = mr.run_model('torch_geometric', arr)
    hdr, rows = _parse_csv(csv_out)
    # Expect header to contain 'horizon' and either 'node' or 'forecast'
    assert any('horizon' in h for h in hdr)
    assert len(rows) >= 1


def test_torch_geometric_fallback_scalar():
    # When provided with simple scalar CSV, it should still produce horizon outputs
    csv_text = "time,value\n0,1.0\n1,2.0\n2,3.0"
    out = mr.run_torch_geometric(csv_text, horizon=4)
    hdr, rows = _parse_csv(out)
    assert len(rows) >= 4
