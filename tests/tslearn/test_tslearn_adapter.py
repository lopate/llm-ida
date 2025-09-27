import csv
from io import StringIO
import app.model_runner as mr


def _parse_csv(s: str):
    f = StringIO(s)
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)
    return header, rows


def test_pysteps_basic_multi_node():
    # build simple CSV with t,node,value for T=5, N=2
    lines = [["t", "node", "value"]]
    for t in range(5):
        for n in range(2):
            lines.append([str(t), str(n), f"{t*10 + n}.0"])
    csv_text = "\n".join([",".join(r) for r in lines])

    # Provide array input and expect array output (H, N)
    import numpy as _np
    # create T,N,1 array
    arr = _np.zeros((5, 2, 1), dtype=_np.float32)
    for t in range(5):
        arr[t, 0, 0] = t * 10 + 0.0
        arr[t, 1, 0] = t * 10 + 1.0

    out = mr.run_pysteps(arr, horizon=3)
    assert isinstance(out, _np.ndarray)
    assert out.shape[0] >= 3


def test_pysteps_fallback_scalar():
    # simple two-line CSV without node column
    import numpy as _np
    arr = _np.array([1.0, 2.0, 3.0], dtype=_np.float32)
    out = mr.run_pysteps(arr, horizon=4)
    assert isinstance(out, _np.ndarray)
    assert out.shape[0] >= 4
