import csv
import io
from typing import Tuple, List


def _to_csv(header: List[str], rows: List[List]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def run_pysteps(csv_text: str):
    # mock: repeat last row as forecast
    rows = [r for r in csv_text.splitlines() if r.strip()]
    if len(rows) <= 1:
        return _to_csv(["t","value"], [[0,0]])
    last = rows[-1].split(",")
    header = rows[0].split(",")
    # output 3-step persistence
    out_rows = []
    for i in range(3):
        out_rows.append([f"t+{i+1}"] + last[1:])
    return _to_csv(["time"] + header[1:], out_rows)


def run_sktime(csv_text: str):
    # mock: forecast mean of last column
    lines = [l for l in csv_text.splitlines() if l.strip()]
    header = lines[0].split(",")
    vals = []
    for l in lines[1:]:
        parts = l.split(",")
        try:
            vals.append(float(parts[-1]))
        except Exception:
            pass
    mean = sum(vals)/len(vals) if vals else 0.0
    rows = [[i+1, mean] for i in range(3)]
    return _to_csv(["horizon","forecast"], rows)


def run_tslearn(csv_text: str):
    # mock: return median
    lines = [l for l in csv_text.splitlines() if l.strip()]
    vals = []
    for l in lines[1:]:
        try:
            vals.append(float(l.split(",")[-1]))
        except Exception:
            pass
    vals.sort()
    if not vals:
        med = 0.0
    else:
        mid = len(vals)//2
        med = vals[mid]
    rows = [[i+1, med] for i in range(3)]
    return _to_csv(["horizon","forecast"], rows)


def run_torch_geometric(csv_text: str):
    # mock: linear extrapolation based on last two points
    lines = [l for l in csv_text.splitlines() if l.strip()]
    vals = []
    for l in lines[1:]:
        try:
            vals.append(float(l.split(",")[-1]))
        except Exception:
            pass
    if len(vals) >= 2:
        delta = vals[-1] - vals[-2]
    else:
        delta = 0.0
    rows = [[i+1, (vals[-1] if vals else 0.0) + (i+1)*delta] for i in range(3)]
    return _to_csv(["horizon","forecast"], rows)


RUNNERS = {
    "pysteps": run_pysteps,
    "sktime": run_sktime,
    "tslearn": run_tslearn,
    "torch_geometric": run_torch_geometric,
}


def run_model(lib_name: str, csv_text: str) -> Tuple[str, str]:
    """Run mock model and return (code_py, predictions_csv)"""
    func = RUNNERS.get(lib_name)
    if not func:
        # fallback to sktime
        func = run_sktime

    predictions = func(csv_text)

    code = f"# Mock runner for {lib_name}\n" + "def predict(csv_text):\n    # implement model loading and prediction here\n    return '" + predictions.replace("\n", "\\n") + "'\n"
    return code, predictions
