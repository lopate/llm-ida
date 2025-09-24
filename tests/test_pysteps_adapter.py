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

    out = mr.run_pysteps(csv_text, horizon=3)
    hdr, rows = _parse_csv(out)
    # should contain 'forecast' column or 'value' depending on branch
    assert any('forecast' in h or 'value' in h for h in hdr)
    # expect at least horizon rows in output
    assert len(rows) >= 3


def test_pysteps_fallback_scalar():
    # simple two-line CSV without node column
    csv_text = "time,value\n0,1.0\n1,2.0\n2,3.0"
    out = mr.run_pysteps(csv_text, horizon=4)
    hdr, rows = _parse_csv(out)
    assert any('forecast' in h or 'value' in h for h in hdr)
    assert len(rows) >= 4
