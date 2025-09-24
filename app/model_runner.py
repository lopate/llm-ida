import csv
import io
from typing import Tuple, List, Any
import numpy as np


def make_forecaster(name: str, args: dict = None):
    """Module-level factory for creating sktime forecasters.

    This is a thin wrapper used by `run_sktime` and is convenient to patch in tests.
    """
    args = args or {}
    name_l = (name or 'naive').lower()
    try:
        # try importing sktime modules but we will avoid using NaiveForecaster
        # directly because it emits a FutureWarning during concatenation in some versions.
        import importlib
        _sktime = importlib.import_module('sktime')
    except Exception:
        # sktime not available -> return a simple lambda-like object
        class Simple:
            def fit(self, y):
                return self

            def predict(self, fh):
                import numpy as _np
                return _np.zeros(len(fh))

        return Simple()

    if name_l in ('naive', 'last'):
        # use a tiny, stable local forecaster to avoid sktime internal warnings
        class Simple:
            def fit(self, y):
                # accept pandas Series or numpy array
                import numpy as _np
                import pandas as _pd
                if isinstance(y, _pd.Series):
                    self._y = y.dropna().to_numpy()
                else:
                    self._y = _np.asarray(y)
                return self

            def predict(self, fh):
                import numpy as _np
                # simple last-value persistence
                if getattr(self, '_y', None) is None or len(self._y) == 0:
                    return _np.zeros(len(fh))
                last = float(self._y[-1])
                return _np.array([last for _ in range(len(fh))], dtype=_np.float32)

        return Simple()

    if name_l in ('autoarima', 'arima', 'auto_arima'):
        try:
            from sktime.forecasting.arima import AutoARIMA
            return AutoARIMA(**args) if args else AutoARIMA()
        except Exception:
            return NaiveForecaster(strategy='last')

    if name_l in ('autoets', 'ets', 'auto_ets'):
        try:
            from sktime.forecasting.ets import AutoETS
            return AutoETS(**args) if args else AutoETS()
        except Exception:
            return NaiveForecaster(strategy='last')

    if name_l in ('forest', 'randomforest', 'rf'):
        try:
            from sktime.forecasting.compose import ReducedRegressionForecaster
            from sklearn.ensemble import RandomForestRegressor
            n_est = int(args.get('n_estimators', 50)) if isinstance(args, dict) else 50
            reg = RandomForestRegressor(n_estimators=n_est)
            return ReducedRegressionForecaster(regressor=reg)
        except Exception:
            return NaiveForecaster(strategy='last')

    # default: return a simple stable forecaster
    class SimpleDefault:
        def fit(self, y):
            import numpy as _np
            import pandas as _pd
            if isinstance(y, _pd.Series):
                self._y = y.dropna().to_numpy()
            else:
                self._y = _np.asarray(y)
            return self

        def predict(self, fh):
            import numpy as _np
            if getattr(self, '_y', None) is None or len(self._y) == 0:
                return _np.zeros(len(fh))
            last = float(self._y[-1])
            return _np.array([last for _ in range(len(fh))], dtype=_np.float32)

    return SimpleDefault()


def _to_csv(header: List[str], rows: List[List]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def run_pysteps(csv_text: str, **kwargs):
    """Adapter for PySTEPS-like nowcasting.

    Attempt to use `pysteps` if installed to run a simple extrapolation (persistence
    or optical-flow) on the last frame. If `pysteps` is not available, fall back to a
    persistence-based mock: repeat the last timestep for the forecast horizon.
    """
    horizon = int(kwargs.get('horizon', 3))
    # try to parse CSV into a 2D array (T, N) or 3D (T,N,F)
    try:
        arr = _csv_to_array(csv_text)
    except Exception:
        arr = None

    # If pysteps is available, try a very small pipeline: use persistence or simple extrap
    try:
        import numpy as _np
        # lightweight use of pysteps nowcasting: try `pysteps.motion` if present
        try:
            from pysteps import motion
            # We will use a trivial persistence: copy last frame for each horizon step
            if isinstance(arr, _np.ndarray) and arr.ndim >= 2:
                last = arr[-1]
                out_rows = []
                for i in range(horizon):
                    # flatten per-node values if needed
                    if last.ndim == 1:
                        vals = last.tolist()
                    else:
                        # take first feature channel
                        vals = (last[:, 0] if last.ndim >= 2 else last.ravel()).tolist()
                    # produce one row per node
                    for n, v in enumerate(vals):
                        out_rows.append([i + 1, n, float(v)])
                return _to_csv(["horizon", "node", "forecast"], out_rows)
        except Exception:
            # if pysteps import fails or motion API missing, proceed to fallback below
            pass
    except Exception:
        # if any import-time error, fall back
        pass

    # final fallback: persistence / repeat last seen value
    lines = [r for r in csv_text.splitlines() if r.strip()]
    header = lines[0].split(",") if lines else ["t", "value"]
    vals = []
    for l in lines[1:]:
        parts = l.split(",")
        try:
            vals.append(float(parts[-1]))
        except Exception:
            pass
    last = vals[-1] if vals else 0.0
    out_rows = []
    for i in range(horizon):
        out_rows.append([i + 1, last])
    # choose header depending on whether node information exists
    if len(header) >= 3 and header[1].lower() in ('node', 'station'):
        # cannot recover node mapping from the flattened fallback; provide generic rows
        rows = []
        for i in range(horizon):
            rows.append([i + 1, 0, float(last)])
        return _to_csv(["horizon", "node", "forecast"], rows)
    return _to_csv(["horizon", "forecast"], out_rows)


def run_sktime(csv_text: str, *args, **kwargs):
    """Real sktime adapter: fits a simple forecaster per-node and returns CSV.

    Accepts `csv_text` produced by `_to_csv_for_runner` (or raw CSV). If `sktime`
    is available, uses `NaiveForecaster` with a relative forecasting horizon.
    Falls back to a simple mock if any import/prediction error occurs.
    """
    # determine horizon and requested model if passed
    horizon = 3
    model_name = kwargs.get('model', kwargs.get('model_name', 'naive'))
    model_args = kwargs.get('model_args', {}) or {}
    if args:
        try:
            horizon = int(args[0])
        except Exception:
            pass
    if 'horizon' in kwargs:
        try:
            horizon = int(kwargs['horizon'])
        except Exception:
            pass

    # parse CSV into numpy array if possible
    try:
        arr = _csv_to_array(csv_text)
    except Exception:
        arr = None

    # try to use sktime if available, but avoid importing sktime internals unconditionally
    try:
        import numpy as _np
        import pandas as pd

        # delegate forecaster construction to module-level factory so it can be patched in tests
        # make_forecaster will provide a lightweight local forecaster for 'naive' strategy
        def _apply_forecaster(series, forecaster):
            forecaster.fit(series)
            # try numeric fh first (many simple forecasters accept array-like)
            fh_try = _np.arange(1, horizon + 1)
            try:
                pred = forecaster.predict(fh_try)
                # if returns pandas/np array-like
                try:
                    return _np.asarray(pred).ravel()
                except Exception:
                    return _np.array(pred)
            except Exception:
                # fallback: import ForecastingHorizon if available and try again
                try:
                    from sktime.forecasting.base import ForecastingHorizon
                    fh = ForecastingHorizon(_np.arange(1, horizon + 1), is_relative=True)
                    pred = forecaster.predict(fh)
                    try:
                        return _np.asarray(pred).ravel()
                    except Exception:
                        return _np.array(pred)
                except Exception:
                    # cannot use this forecaster
                    raise

        # if arr is 1D (univariate series)
        if isinstance(arr, _np.ndarray) and arr.ndim == 1:
            y = pd.Series(arr).dropna()
            # if series is empty after dropping NA, fall back to mock
            if y.empty:
                raise ValueError("empty series")
            forecaster = make_forecaster(model_name, model_args)
            pred = _apply_forecaster(y, forecaster)
            rows = [[i + 1, float(pred[i])] for i in range(len(pred))]
            return _to_csv(["horizon", "forecast"], rows)

        # if arr is 2D treat as (T, features)
        if isinstance(arr, _np.ndarray) and arr.ndim == 2:
            T, N = arr.shape
            out = _np.zeros((horizon, N), dtype=_np.float32)
            for n in range(N):
                series = pd.Series(arr[:, n]).dropna()
                if series.empty:
                    # fallback to last-value repetition for this node
                    out[:, n] = _np.full((horizon,), 0.0, dtype=_np.float32)
                    continue
                forecaster = make_forecaster(model_name, model_args)
                pred = _apply_forecaster(series, forecaster)
                out[:, n] = pred
            rows = []
            for h in range(horizon):
                for n in range(N):
                    rows.append([h + 1, n, float(out[h, n])])
            return _to_csv(["horizon", "node", "forecast"], rows)

        # 3D case (T, N, F)
        if isinstance(arr, _np.ndarray) and arr.ndim >= 3:
            T, N, F = arr.shape[0], arr.shape[1], (arr.shape[2] if arr.ndim >= 3 else 1)
            out = _np.zeros((horizon, N, 1), dtype=_np.float32)
            for n in range(N):
                series = pd.Series(arr[:, n, 0]).dropna()
                if series.empty:
                    out[:, n, 0] = _np.full((horizon,), 0.0, dtype=_np.float32)
                    continue
                forecaster = make_forecaster(model_name, model_args)
                pred = _apply_forecaster(series, forecaster)
                out[:, n, 0] = pred
            rows = []
            for h in range(horizon):
                for n in range(N):
                    rows.append([h + 1, n, float(out[h, n, 0])])
            return _to_csv(["horizon", "node", "forecast"], rows)

    except Exception:
        # fall back to mock behavior below
        pass

    # if sktime not available or parsing failed, use simple mock behavior
    lines = [l for l in csv_text.splitlines() if l.strip()]
    header = lines[0].split(",") if lines else ["horizon", "forecast"]
    vals = []
    for l in lines[1:]:
        parts = l.split(",")
        try:
            vals.append(float(parts[-1]))
        except Exception:
            pass
    mean = sum(vals)/len(vals) if vals else 0.0
    rows = [[i+1, mean] for i in range(horizon)]
    return _to_csv(["horizon","forecast"], rows)


def run_tslearn(csv_text: str, **kwargs):
    """tslearn adapter with sliding-window KNN fallback.

    Attempts to use `tslearn` neighbor regressor if available; otherwise
    uses `sklearn.neighbors.KNeighborsRegressor` with a sliding-window
    supervised framing. Supports `model_args`:
      - n_neighbors (int)
      - window_size (int)
      - metric (str, optional)

    Returns CSV with header ['horizon','node','forecast'] for multi-node
    input or ['horizon','forecast'] for univariate input.
    """
    # extract kwargs
    horizon = int(kwargs.get('horizon', 3))
    model_name = kwargs.get('model_name') or ''
    model_args = kwargs.get('model_args') or {}
    n_neighbors = int(model_args.get('n_neighbors', 5))
    window_size = int(model_args.get('window_size', 10))

    # parse input into numpy array if possible
    try:
        arr = _csv_to_array(csv_text)
    except Exception:
        arr = None

    import numpy as _np

    def _predict_series_knn(series: _np.ndarray):
        """Train a sliding-window KNN regressor to predict `horizon` values."""
        series = _np.asarray(series, dtype=_np.float32)
        T = series.shape[0]
        if T < 1:
            return _np.zeros((horizon,), dtype=_np.float32)

        # If not enough data to make a supervised matrix, fallback to repeating last value
        if T < window_size + 1:
            return _np.full((horizon,), float(series[-1]) if T > 0 else 0.0, dtype=_np.float32)

        X = []
        Y = []
        for i in range(0, T - window_size - horizon + 1):
            x = series[i:i + window_size]
            y = series[i + window_size: i + window_size + horizon]
            if len(y) < horizon:
                continue
            X.append(x)
            Y.append(y)

        if not X:
            return _np.full((horizon,), float(series[-1]), dtype=_np.float32)

        X = _np.vstack(X)
        Y = _np.vstack(Y)

        # Try tslearn regressor first (if it provides a convenient API)
        try:
            # avoid importing tslearn if h5py is not installed (tslearn warns about this)
            import importlib
            has_h5py = importlib.util.find_spec('h5py') is not None
            if has_h5py:
                from tslearn.neighbors import KNeighborsTimeSeriesRegressor
                # KNeighborsTimeSeriesRegressor expects shape (n_ts, sz, d) for time series
                # convert X to required shape and wrap Y accordingly
                X_ts = X.reshape((X.shape[0], X.shape[1], 1))
                knn = KNeighborsTimeSeriesRegressor(n_neighbors=n_neighbors)
                knn.fit(X_ts, Y)
                last = series[-window_size:]
                last_ts = last.reshape((1, last.shape[0], 1))
                pred = knn.predict(last_ts)
                pred = _np.asarray(pred).reshape(-1)[:horizon]
                return pred
        except Exception:
            # fall back to sklearn or other fallback below
            pass

        # Fallback to sklearn KNeighborsRegressor with multi-output
        try:
            from sklearn.neighbors import KNeighborsRegressor
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            knn.fit(X, Y)
            last = series[-window_size:]
            pred = knn.predict(last.reshape(1, -1))
            pred = _np.asarray(pred).reshape(-1)[:horizon]
            return pred
        except Exception:
            # as ultimate fallback, return last value repeated or simple linear extrapolation
            if T >= 2:
                delta = float(series[-1]) - float(series[-2])
            else:
                delta = 0.0
            return _np.array([float(series[-1]) + (i + 1) * delta for i in range(horizon)], dtype=_np.float32)

    # Now build outputs depending on arr shape
    try:
        # 1D series
        if isinstance(arr, _np.ndarray) and arr.ndim == 1:
            pred = _predict_series_knn(arr)
            rows = [[i + 1, float(pred[i])] for i in range(len(pred))]
            return _to_csv(["horizon", "forecast"], rows)

        # 2D: treat as (T, N) -> nodes in columns
        if isinstance(arr, _np.ndarray) and arr.ndim == 2:
            T, N = arr.shape
            out = _np.zeros((horizon, N), dtype=_np.float32)
            for n in range(N):
                series = arr[:, n]
                out[:, n] = _predict_series_knn(series)
            rows = []
            for h in range(horizon):
                for n in range(N):
                    rows.append([h + 1, n, float(out[h, n])])
            return _to_csv(["horizon", "node", "forecast"], rows)

        # 3D: (T, N, F)
        if isinstance(arr, _np.ndarray) and arr.ndim >= 3:
            T, N = arr.shape[0], arr.shape[1]
            out = _np.zeros((horizon, N, 1), dtype=_np.float32)
            for n in range(N):
                series = arr[:, n, 0]
                out[:, n, 0] = _predict_series_knn(series)
            rows = []
            for h in range(horizon):
                for n in range(N):
                    rows.append([h + 1, n, float(out[h, n, 0])])
            return _to_csv(["horizon", "node", "forecast"], rows)
    except Exception:
        # fall through to simple fallback
        pass

    # final fallback: median-based simple output similar to previous mock
    lines = [l for l in csv_text.splitlines() if l.strip()]
    vals = []
    for l in lines[1:]:
        try:
            vals.append(float(l.split(",")[-1]))
        except Exception:
            pass
    vals.sort()
    med = vals[len(vals)//2] if vals else 0.0
    rows = [[i + 1, med] for i in range(horizon)]
    return _to_csv(["horizon", "forecast"], rows)


def run_torch_geometric(csv_text: str, **kwargs):
    """Adapter for graph-based models (torch_geometric).

    Attempt to construct a simple graph-based prediction if `torch_geometric` is
    installed: build a small PyTorch model that averages neighbor values (if edge
    information is provided), or else fall back to a simple linear extrapolation
    per-node using the last two timesteps.
    
    The function expects CSV text produced by `_to_csv_from_array` (t,node,value)
    or similar. If no node/edge info is available, it falls back to per-node
    linear extrapolation.
    """
    horizon = int(kwargs.get('horizon', 3))
    try:
        arr = _csv_to_array(csv_text)
    except Exception:
        arr = None

    import numpy as _np

    # allow passing model-specific args via kwargs (e.g. train flags, lr)
    model_args = kwargs.get('model_args') or {}

    # If torch_geometric and torch are available, try a minimal neighbor-averaging model.
    # allow caller to provide explicit edge_index via kwargs (list/array of pairs)
    edge_index_kw = kwargs.get('edge_index', None)

    try:
        import torch
        try:
            # Try a small GCN-based inference if torch_geometric is present
            from torch_geometric.nn import GCNConv

            if isinstance(arr, _np.ndarray) and arr.ndim >= 2:
                T = arr.shape[0]
                if arr.ndim == 3:
                    N = arr.shape[1]
                    last = arr[-1, :, 0]
                    prev = arr[-2, :, 0] if T >= 2 else last
                elif arr.ndim == 2:
                    N = arr.shape[1]
                    last = arr[-1, :]
                    prev = arr[-2, :] if T >= 2 else last
                else:
                    last = arr[-1]
                    prev = arr[-2] if T >= 2 else last

                if isinstance(last, _np.ndarray) and last.size > 1:
                    # Use provided edge_index if given, otherwise build a simple fully-connected graph
                    if edge_index_kw is not None:
                        try:
                            ei = edge_index_kw
                            import numpy as _np_local
                            if isinstance(ei, _np_local.ndarray):
                                ei_arr = ei
                            else:
                                ei_arr = _np_local.asarray(ei)
                            # expect shape (2, E) or (E, 2)
                            if ei_arr.ndim == 2 and ei_arr.shape[0] == 2:
                                edge_index = torch.tensor(ei_arr, dtype=torch.long)
                            elif ei_arr.ndim == 2 and ei_arr.shape[1] == 2:
                                edge_index = torch.tensor(ei_arr.T, dtype=torch.long)
                            else:
                                edge_index = None
                        except Exception:
                            edge_index = None
                    else:
                        row = []
                        col = []
                        for i in range(N):
                            for j in range(N):
                                if i != j:
                                    row.append(i)
                                    col.append(j)
                        edge_index = torch.tensor([row, col], dtype=torch.long) if row else torch.empty((2,0), dtype=torch.long)

                    x = torch.tensor(last.reshape(-1, 1), dtype=torch.float32)
                    # if requested, perform a small CPU-only training loop
                    do_train = bool(kwargs.get('train', False) or (isinstance(model_args, dict) and model_args.get('train', False)))
                    if do_train:
                        # training hyperparameters
                        epochs = int(model_args.get('train_epochs', 5)) if isinstance(model_args, dict) else 5
                        lr = float(model_args.get('lr', 0.01)) if isinstance(model_args, dict) else 0.01
                        window = int(model_args.get('window_size', 1)) if isinstance(model_args, dict) else 1

                        # prepare simple dataset: for each t in [window, T-1) use last value as feature and next value as target
                        X_samples = []
                        Y_samples = []
                        for t_idx in range(window, T):
                            # feature: previous value at t_idx-1
                            feat = arr[t_idx - 1, :, 0]
                            tgt = arr[t_idx, :, 0]
                            X_samples.append(feat.reshape(-1, 1))
                            Y_samples.append(tgt)

                        if X_samples:
                            import torch.nn as nn
                            conv = GCNConv(1, 16)
                            readout = nn.Linear(16, 1)
                            params = list(conv.parameters()) + list(readout.parameters())
                            opt = torch.optim.Adam(params, lr=lr)
                            loss_fn = nn.MSELoss()
                            # training loop
                            for ep in range(epochs):
                                total_loss = 0.0
                                for xi, yi in zip(X_samples, Y_samples):
                                    x_in = torch.tensor(xi, dtype=torch.float32)
                                    y_true = torch.tensor(yi, dtype=torch.float32)
                                    opt.zero_grad()
                                    try:
                                        out = conv(x_in, edge_index) if edge_index is not None else conv(x_in, torch.empty((2,0), dtype=torch.long))
                                    except Exception:
                                        out = conv(x_in, torch.empty((2,0), dtype=torch.long))
                                    h = torch.relu(out)
                                    pred = readout(h).squeeze()
                                    loss = loss_fn(pred, y_true)
                                    loss.backward()
                                    opt.step()
                                    total_loss += float(loss.item())
                                # optional: print progress small-scale
                                # print(f"epoch {ep+1}/{epochs} loss={total_loss:.6f}")

                            # after training, produce prediction using last observed features
                            x_last = torch.tensor(last.reshape(-1, 1), dtype=torch.float32)
                            try:
                                out = conv(x_last, edge_index) if edge_index is not None else conv(x_last, torch.empty((2,0), dtype=torch.long))
                            except Exception:
                                out = conv(x_last, torch.empty((2,0), dtype=torch.long))
                            h = torch.relu(out)
                            pred_out = readout(h).squeeze().detach().numpy()
                            # build multi-horizon by simple extrapolation using delta and GCN output
                            delta = (last - prev)
                            preds = []
                            for h_idx in range(horizon):
                                step = pred_out + (h_idx) * delta
                                for n in range(N):
                                    preds.append([h_idx + 1, n, float(step[n])])
                            return _to_csv(["horizon", "node", "forecast"], preds)

                    # inference-only path (no training requested)
                    conv = GCNConv(1, 1)
                    with torch.no_grad():
                        try:
                            out = conv(x, edge_index) if edge_index is not None else conv(x, torch.empty((2,0), dtype=torch.long))
                        except Exception:
                            out = None
                    neigh = out.squeeze().numpy() if out is not None else last
                    delta = (last - prev)
                    preds = []
                    for h in range(horizon):
                        step = last + (h + 1) * delta + 0.05 * neigh
                        for n in range(N):
                            preds.append([h + 1, n, float(step[n])])
                    return _to_csv(["horizon", "node", "forecast"], preds)
        except Exception:
            # torch_geometric/GCNConv not available or failed: fall through
            pass
    except Exception:
        # torch not installed
        pass
    except Exception:
        # torch not installed
        pass

    # fallback: simple linear extrapolation per-node using last two values
    lines = [l for l in csv_text.splitlines() if l.strip()]
    vals_by_node = {}
    for l in lines[1:]:
        parts = l.split(',')
        if len(parts) >= 3:
            try:
                t = parts[0]
                node = int(parts[1])
                v = float(parts[-1])
            except Exception:
                continue
            vals_by_node.setdefault(node, []).append(v)

    rows = []
    for h in range(horizon):
        for node, vals in sorted(vals_by_node.items()):
            if len(vals) >= 2:
                delta = vals[-1] - vals[-2]
            else:
                delta = 0.0
            pred = (vals[-1] if vals else 0.0) + (h + 1) * delta
            rows.append([h + 1, node, float(pred)])
    if rows:
        return _to_csv(["horizon", "node", "forecast"], rows)

    # ultimate fallback: scalar extrapolation
    vals = []
    for l in lines[1:]:
        try:
            vals.append(float(l.split(',')[-1]))
        except Exception:
            pass
    if len(vals) >= 2:
        delta = vals[-1] - vals[-2]
    else:
        delta = 0.0
    rows = [[i + 1, (vals[-1] if vals else 0.0) + (i + 1) * delta] for i in range(horizon)]
    return _to_csv(["horizon", "forecast"], rows)


RUNNERS = {
    "pysteps": run_pysteps,
    "sktime": run_sktime,
    "tslearn": run_tslearn,
    "torch_geometric": run_torch_geometric,
}


def run_model(lib_name: str, csv_text: str, **kwargs) -> Tuple[str, str]:
    """Run mock model and return (code_py, predictions_csv).

    Any additional keyword arguments are forwarded to the underlying runner
    (for example `horizon` or `edge_index`).
    """
    func = RUNNERS.get(lib_name)
    if not func:
        # fallback to sktime
        func = run_sktime

    # normalize input: accept CSV text or numpy ndarray (T,N,F) or similar
    def _to_csv_from_array(arr: Any) -> str:
        if isinstance(arr, str):
            return arr
        # try numpy-like
        try:
            a = np.asarray(arr)
        except Exception:
            return str(arr)
        # 3D array: (T, N, F) -> t,node,value
        if a.ndim >= 3:
            T_s, N_s = a.shape[0], a.shape[1]
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(['t', 'node', 'value'])
            for t in range(T_s):
                for n in range(N_s):
                    try:
                        v = float(a[t, n, 0])
                    except Exception:
                        v = float(a[t, n]) if a.ndim == 2 else 0.0
                    writer.writerow([t, n, v])
            return buf.getvalue()
        # 2D array: (T, N) -> t,node,value
        if a.ndim == 2:
            T_s, N_s = a.shape
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(['t', 'node', 'value'])
            for t in range(T_s):
                for n in range(N_s):
                    try:
                        v = float(a[t, n])
                    except Exception:
                        v = 0.0
                    writer.writerow([t, n, v])
            return buf.getvalue()

        # 1D array: (T,) -> t,value
        if a.ndim == 1:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(['t', 'value'])
            for i, v in enumerate(a.tolist()):
                try:
                    vv = float(v)
                except Exception:
                    vv = 0.0
                writer.writerow([i, vv])
            return buf.getvalue()

        # fallback: represent as string
        return str(arr)

    csv_input = _to_csv_from_array(csv_text)
    predictions = func(csv_input, **kwargs)

    code = f"# Mock runner for {lib_name}\n" + "def predict(csv_text):\n    # implement model loading and prediction here\n    return '" + predictions.replace("\n", "\\n") + "'\n"
    return code, predictions


def _csv_to_array(csv_text: str):
    """Parse CSV with header (t,node,value) into numpy array of shape (T,N,1).
    If CSV has a different layout, try to return a 2D array (T, features).
    """
    import csv
    from collections import defaultdict

    lines = [l for l in csv_text.splitlines() if l.strip()]
    if not lines:
        return np.zeros((0, 0, 1), dtype=np.float32)
    reader = csv.reader(lines)
    header = next(reader)
    # try to detect (t,node,value) format
    h0 = header[0].lower() if header else ''
    h1 = header[1].lower() if len(header) > 1 else ''
    hlast = header[-1].lower() if header else ''

    if len(header) >= 3 and h0 in ('t', 'time') and h1 in ('node', 'n'):
        data = defaultdict(dict)
        max_t = -1
        max_n = -1
        for row in reader:
            try:
                t = int(float(row[0]))
                n = int(float(row[1]))
                v = float(row[2])
            except Exception:
                continue
            data[t][n] = v
            if t > max_t:
                max_t = t
            if n > max_n:
                max_n = n
        T = max_t + 1
        N = max_n + 1
        out = np.zeros((T, N, 1), dtype=np.float32)
        for t, cols in data.items():
            for n, v in cols.items():
                out[t, n, 0] = v
        return out

    # detect (horizon,node,forecast) style output from multi-node forecasters
    if len(header) >= 3 and h0 in ('horizon', 'h') and h1 in ('node', 'n') and hlast in ('forecast', 'value', 'pred', 'prediction'):
        data = defaultdict(dict)
        max_h = -1
        max_n = -1
        for row in reader:
            try:
                h = int(float(row[0]))
                n = int(float(row[1]))
                v = float(row[2])
            except Exception:
                continue
            data[h][n] = v
            if h > max_h:
                max_h = h
            if n > max_n:
                max_n = n
        H = max_h
        N = max_n + 1
        out = np.zeros((H, N, 1), dtype=np.float32)
        # horizons indexed 1..H in CSV -> map to 0..H-1
        for h, cols in data.items():
            for n, v in cols.items():
                out[h-1, n, 0] = v
        return out

    # fallback: try single-column time series
    vals = []
    for row in reader:
        try:
            vals.append(float(row[-1]))
        except Exception:
            pass
    return np.array(vals, dtype=np.float32)


def run_model_from_choice(choice: dict, input_data, horizon: int = 3, **kwargs):
    """Run a selected model given the LLM choice dict (or library name string).

    Parameters:
    - choice: either a dict as returned by `select_model` or a string with library name.
    - input_data: can be a numpy array (T,N,F), a path to .npz/.npy file, or CSV text.
    - horizon: number of future steps to predict.

    Returns:
    - dict with keys: library, y_pred (np.ndarray), csv (text), meta
    """
    lib = None
    if isinstance(choice, dict):
        lib = choice.get('library') or choice.get('model_choice')
    elif isinstance(choice, str):
        lib = choice
    else:
        raise ValueError('choice must be dict or str')

    # normalize lib name
    lib = (lib or 'sktime').lower()

    # load input_data
    arr = None
    if isinstance(input_data, str):
        # try file
        p = input_data
        try:
            if p.endswith('.npz') or p.endswith('.npz'):
                d = np.load(p, allow_pickle=True)
                if 'X' in d:
                    arr = d['X']
                elif 'y' in d:
                    arr = d['y']
                else:
                    # take first array
                    keys = list(d.keys())
                    arr = d[keys[0]]
            elif p.endswith('.npy'):
                arr = np.load(p)
            else:
                # treat as CSV text
                arr = _csv_to_array(p)
        except Exception:
            # treat as CSV text
            arr = _csv_to_array(p)
    else:
        # array-like or file-like
        try:
            arr = np.asarray(input_data)
        except Exception:
            arr = input_data

    # If arr is multi-dim, convert to CSV for the mock runners
    def _to_csv_for_runner(a):
        if isinstance(a, str):
            return a
        try:
            a = np.asarray(a)
        except Exception:
            return str(a)
        if a.ndim >= 3:
            T_s, N_s = a.shape[0], a.shape[1]
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(['t', 'node', 'value'])
            for t in range(T_s):
                for n in range(N_s):
                    try:
                        v = float(a[t, n, 0])
                    except Exception:
                        v = float(a[t, n]) if a.ndim == 2 else 0.0
                    writer.writerow([t, n, v])
            return buf.getvalue()
        # 1D array -> simple CSV
        if a.ndim == 1:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(['t', 'value'])
            for i, v in enumerate(a.tolist()):
                writer.writerow([i, float(v)])
            return buf.getvalue()
        return str(a)

    csv_text = _to_csv_for_runner(arr)

    # extract model_name and args from choice if present and also from kwargs
    model_name = None
    model_args = {}
    if isinstance(choice, dict):
        model_name = choice.get('model_name') or choice.get('model') or choice.get('model_choice')
        model_args = choice.get('model_args') or {}

    # also allow callers to pass model_name/model_args directly via kwargs
    # merge such info, preferring explicit kwargs
    if 'model_name' in kwargs and kwargs.get('model_name') is not None:
        model_name = kwargs.get('model_name')
    if 'model_args' in kwargs and kwargs.get('model_args'):
        try:
            extra_args = kwargs.get('model_args') or {}
            if isinstance(extra_args, dict):
                merged = {}
                merged.update(model_args or {})
                merged.update(extra_args)
                model_args = merged
        except Exception:
            pass

    # choose runner function
    func = RUNNERS.get(lib, run_sktime)
    # prepare forwarded kwargs for adapter: include horizon, model_name, model_args and any other user kwargs
    forward_kwargs = dict(kwargs)  # copy
    # ensure horizon/model_name/model_args present and normalized
    forward_kwargs['horizon'] = horizon
    forward_kwargs['model_name'] = model_name
    forward_kwargs['model_args'] = model_args

    # call runner to get CSV output
    try:
        predictions_csv = func(csv_text, **forward_kwargs)
    except TypeError:
        # adapter didn't accept kwargs; try calling with minimal signature
        try:
            predictions_csv = func(csv_text, model_name=model_name, model_args=model_args, horizon=horizon)
        except Exception:
            predictions_csv = func(csv_text)
    # try to parse CSV into numpy array
    # predictions_csv may have header like ['horizon','forecast'] or ['time', 'node', ...]
    pred_arr = _csv_to_array(predictions_csv)

    result = {
        'library': lib,
        'y_pred': pred_arr,
        'csv': predictions_csv,
        'meta': {'horizon': horizon}
    }
    return result
