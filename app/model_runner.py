import csv
import io
import importlib
import importlib.util as importlib_util
from typing import Tuple, List, Any, Dict, Optional, TypedDict, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray


def make_forecaster(name: str, args: Optional[Dict[str, Any]] = None) -> Any:
    """Module-level factory for creating sktime forecasters.

    This is a thin wrapper used by `run_sktime` and is convenient to patch in tests.
    """
    import re
    args = args or {}
    # normalize name: lowercase and strip non-alphanumeric so values like
    # 'AutoARIMA', 'auto_arima', 'auto-arima' all map to the same key
    raw = (name or 'naive')
    name_l = re.sub(r'[^a-z0-9]', '', str(raw).lower())
    try:
        # try importing sktime modules but we will avoid using NaiveForecaster
        # directly because it emits a FutureWarning during concatenation in some versions.
        import importlib
        _sktime = importlib.import_module('sktime')
    except Exception:
        # sktime not available -> return a simple lambda-like object
        class Simple:
            def fit(self, y: Any) -> Any:
                return self

            def predict(self, fh: Any) -> Any:
                import numpy as _np
                return _np.zeros(len(fh))

        return Simple()

    # map persistence -> naive forecaster (used by DEFAULT_DOCS)
    if name_l in ('naive', 'last', 'persistence'):
        # use a tiny, stable local forecaster to avoid sktime internal warnings
        class Simple:
            def fit(self, y: Any) -> Any:
                # accept pandas Series or numpy array
                import numpy as _np
                import pandas as _pd
                if isinstance(y, _pd.Series):
                    self._y = y.dropna().to_numpy()
                else:
                    self._y = _np.asarray(y)
                return self

            def predict(self, fh: Any) -> Any:
                import numpy as _np
                # simple last-value persistence
                if getattr(self, '_y', None) is None or len(self._y) == 0:
                    return _np.zeros(len(fh))
                last = float(self._y[-1])
                return _np.array([last for _ in range(len(fh))], dtype=_np.float32)

        return Simple()
    
    if name_l in ('autoarima', 'arima', 'statsforecast_autoarima', 'statsforecastautoarima'):
        from sktime.forecasting.statsforecast import StatsForecastAutoARIMA  # type: ignore
        return StatsForecastAutoARIMA(**args) if args and len(args) > 0 else StatsForecastAutoARIMA()


    if name_l in ('autoets', 'ets'):
        from sktime.forecasting.ets import AutoETS  # type: ignore
        return AutoETS(**args) if args and len(args) > 0 else AutoETS()

    # accept 'forest' as well as potential DEFAULT_DOCS value 'forest'
    if name_l in ('forest', 'randomforest', 'rf'):
        from sktime.forecasting.compose import ReducedRegressionForecaster
        from sklearn.ensemble import RandomForestRegressor
        n_est = int(args.get('n_estimators', 50)) if isinstance(args, dict) else 50
        reg = RandomForestRegressor(n_estimators=n_est)
        return ReducedRegressionForecaster(regressor=reg)

    # default: return a simple stable forecaster
    class SimpleDefault:
        def fit(self, y: Any) -> Any:
            import numpy as _np
            import pandas as _pd
            if isinstance(y, _pd.Series):
                self._y = y.dropna().to_numpy()
            else:
                self._y = _np.asarray(y)
            return self

        def predict(self, fh: Any) -> Any:
            import numpy as _np
            if getattr(self, '_y', None) is None or len(self._y) == 0:
                return _np.zeros(len(fh))
            last = float(self._y[-1])
            return _np.array([last for _ in range(len(fh))], dtype=_np.float32)

    return SimpleDefault()


class ModelChoice(TypedDict, total=False):
    library: str
    model_name: str
    model_args: Dict[str, Any]


class ModelCandidate(TypedDict, total=False):
    """TypedDict representing a candidate entry used by the selector and RAG.

    Fields:
    - library: required library identifier (e.g. 'sktime')
    - model_name: optional model name within the library (e.g. 'AutoARIMA')
    - description: optional textual description used by RAG prompts
    - id: optional stable id for the candidate
    - score: optional retrieval score (float)
    - model_args: optional model args dict
    """
    library: str
    model_name: str
    description: str
    id: str
    score: float
    model_args: Dict[str, Any]


# Numpy ndarray alias for typing
ArrayND = NDArray[Any]

if TYPE_CHECKING:
    # Help static checker know optional third-party modules (no runtime import here)
    try:
        from sktime.forecasting.base import ForecastingHorizon  # type: ignore
    except Exception:
        ForecastingHorizon = Any  # type: ignore
    try:
        from tslearn.neighbors import KNeighborsTimeSeriesRegressor  # type: ignore
    except Exception:
        KNeighborsTimeSeriesRegressor = Any  # type: ignore
    try:
        from torch_geometric.nn import GCNConv  # type: ignore
    except Exception:
        GCNConv = Any  # type: ignore
    # Common sktime symbols used at runtime but may lack stubs
    try:
        from sktime.forecasting.arima import AutoARIMA  # type: ignore
    except Exception:
        AutoARIMA = Any  # type: ignore
    try:
        from sktime.forecasting.ets import AutoETS  # type: ignore
    except Exception:
        AutoETS = Any  # type: ignore
    try:
            from sktime.forecasting.compose import ReducedRegressionForecaster  # type: ignore
    except Exception:
        ReducedRegressionForecaster = Any  # type: ignore


def _to_csv(header: List[str], rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def run_pysteps(data: Any, **kwargs: Any) -> ArrayND:
    """Adapter for PySTEPS-like nowcasting.

    Attempt to use `pysteps` if installed to run a simple extrapolation (persistence
    or optical-flow) on the last frame. If `pysteps` is not available, fall back to a
    persistence-based mock: repeat the last timestep for the forecast horizon.
    """
    horizon = int(kwargs.get('horizon', 3))
    # accept either CSV text (legacy) or numpy-like array
    arr: Optional[ArrayND]
    if isinstance(data, str):
        try:
            arr = _csv_to_array(data)  # type: ignore[arg-type]
        except Exception:
            arr = None
    else:
        try:
            arr = np.asarray(data)
        except Exception:
            arr = None

    # If pysteps is available, try a very small pipeline: use persistence or simple extrap
    try:
        import numpy as _np
        # lightweight use of pysteps nowcasting: try `pysteps.motion` if present
        try:
            # check for presence of pysteps without importing symbols to avoid linter complaints
            if importlib_util.find_spec('pysteps') is not None:
                # We will use a trivial persistence: copy last frame for each horizon step
                if isinstance(arr, _np.ndarray) and arr.ndim >= 2:
                    last = arr[-1]
                    if last.ndim == 1:
                        vals = np.asarray(last, dtype=np.float32)
                    else:
                        vals = np.asarray(last[:, 0] if last.ndim >= 2 else last.ravel(), dtype=np.float32)
                    N = int(vals.shape[0])
                    out = _np.zeros((horizon, N), dtype=_np.float32)
                    for i in range(horizon):
                        out[i, :] = vals
                    return out
        except Exception:
            # if pysteps import fails or motion API missing, proceed to fallback below
            pass
    except Exception:
        # if any import-time error, fall back
        pass

    # final fallback: try to use any numeric values from data and repeat last
    try:
        if isinstance(arr, _np.ndarray) and arr.size > 0:
            # take last value(s)
            if arr.ndim == 1:
                last = float(arr[-1])
                return _np.full((horizon,), last, dtype=_np.float32)
            elif arr.ndim >= 2:
                last_vals = arr.reshape((arr.shape[0], -1))[-1]
                vals = np.asarray(last_vals, dtype=_np.float32)
                N = int(vals.shape[0])
                out = _np.zeros((horizon, N), dtype=_np.float32)
                for i in range(horizon):
                    out[i, :] = vals
                return out
    except Exception:
        pass
    # ultimate fallback scalar
    return _np.zeros((horizon,), dtype=_np.float32)


def run_sktime(data: Any, *args: Any, **kwargs: Any) -> ArrayND:
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

    # accept CSV text (legacy) or numpy-like array
    try:
        if isinstance(data, str):
            arr = _csv_to_array(data)  # type: ignore[arg-type]
        else:
            arr = np.asarray(data)
    except Exception:
        arr = None

    # try to use sktime if available, but avoid importing sktime internals unconditionally

    import numpy as _np
    import pandas as pd

    # delegate forecaster construction to module-level factory so it can be patched in tests
    # make_forecaster will provide a lightweight local forecaster for 'naive' strategy
    def _apply_forecaster(series: Any, forecaster: Any) -> ArrayND:
        """Fit forecaster and return 1D numpy predictions."""
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
        return np.asarray(pred, dtype=np.float32)

    # 3D case (T, N, F): treat as spatio-temporal with nodes in axis 1
    if isinstance(arr, _np.ndarray) and arr.ndim >= 3:
        _T, N = int(arr.shape[0]), int(arr.shape[1])
        out: ArrayND = _np.zeros((horizon, N), dtype=_np.float32)
        for n in range(N):
            series = pd.Series(arr[:, n, 0]).dropna()
            if series.empty:
                out[:, n] = _np.full((horizon,), 0.0, dtype=_np.float32)
                continue
            forecaster = make_forecaster(model_name, model_args)
            pred = _apply_forecaster(series, forecaster)
            out[:, n] = pred
        return out

    # if arr is 2D treat as (T, features)
    if isinstance(arr, _np.ndarray) and arr.ndim == 2:
        _T, N = int(arr.shape[0]), int(arr.shape[1])
        out: ArrayND = _np.zeros((horizon, N), dtype=_np.float32)
        for n in range(N):
            series = pd.Series(arr[:, n]).dropna()
            if series.empty:
                # fallback to last-value repetition for this node
                out[:, n] = _np.full((horizon,), 0.0, dtype=_np.float32)
                continue
            forecaster = make_forecaster(model_name, model_args)
            pred = _apply_forecaster(series, forecaster)
            out[:, n] = pred
        return out



def run_tslearn(data: Any, **kwargs: Any) -> ArrayND:
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
    model_args = kwargs.get('model_args') or {}
    n_neighbors = int(model_args.get('n_neighbors', 5))
    window_size = int(model_args.get('window_size', 10))

    # accept CSV text (legacy) or numpy array
    try:
        if isinstance(data, str):
            arr = _csv_to_array(data)  # type: ignore[arg-type]
        else:
            arr = np.asarray(data)
    except Exception:
        arr = None

    import numpy as _np

    def _predict_series_knn(series: _np.ndarray) -> ArrayND:
        """Train a sliding-window KNN regressor to predict `horizon` values."""
        series = _np.asarray(series, dtype=_np.float32)
        T = int(series.shape[0])
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
            has_h5py = importlib_util.find_spec('h5py') is not None
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
        # 1D series -> return (H,) array
        if isinstance(arr, _np.ndarray) and arr.ndim == 1:
            pred = _predict_series_knn(arr)
            return np.asarray(pred, dtype=np.float32)

        # 2D: treat as (T, N) -> nodes in columns
        if isinstance(arr, _np.ndarray) and arr.ndim == 2:
            _T, N = int(arr.shape[0]), int(arr.shape[1])
            out: ArrayND = _np.zeros((horizon, N), dtype=_np.float32)
            for n in range(N):
                series = arr[:, n]
                out[:, n] = _predict_series_knn(series)
            return out

        # 3D: (T, N, F)
        if isinstance(arr, _np.ndarray) and arr.ndim >= 3:
            N = int(arr.shape[1])
            out: ArrayND = _np.zeros((horizon, N), dtype=_np.float32)
            for n in range(N):
                series = arr[:, n, 0]
                out[:, n] = _predict_series_knn(series)
            return out
    except Exception:
        # fall through to simple fallback
        pass

    # final fallback: try to use numeric content of arr
    try:
        if isinstance(arr, _np.ndarray) and arr.size > 0:
            if arr.ndim == 1:
                last = float(arr[-1])
                return _np.full((horizon,), last, dtype=_np.float32)
            else:
                last_vals = arr.reshape((arr.shape[0], -1))[-1]
                vals = np.asarray(last_vals, dtype=_np.float32)
                N = int(vals.shape[0])
                out = _np.zeros((horizon, N), dtype=_np.float32)
                for i in range(horizon):
                    out[i, :] = vals
                return out
    except Exception:
        pass
    return _np.zeros((horizon,), dtype=_np.float32)


def run_torch_geometric(data: Any, **kwargs: Any) -> ArrayND:
    """Array-first adapter for simple graph-based forecasts.

    Returns numpy arrays of shape (H,) or (H, N). Accepts either a numpy-like
    array (preferred) or CSV text (legacy). Attempts a tiny GCN-based inference
    if torch and torch_geometric are available; otherwise falls back to simple
    per-node linear extrapolation using the last two timesteps.
    """
    horizon = int(kwargs.get('horizon', 3))

    # accept CSV text (legacy) or array
    try:
        if isinstance(data, str):
            arr = _csv_to_array(data)
        else:
            arr = np.asarray(data)
    except Exception:
        arr = None

    import numpy as _np
    model_args = kwargs.get('model_args') or {}
    edge_index_kw = kwargs.get('edge_index', None)

    # Try torch_geometric path
    try:
        import torch
        from torch_geometric.nn import GCNConv
        if isinstance(arr, _np.ndarray) and arr.size > 0:
            T = arr.shape[0]
            # normalize to last and prev vectors
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

            last = np.asarray(last, dtype=_np.float32)
            prev = np.asarray(prev, dtype=_np.float32)

            if last.size > 1:
                # build edge_index
                edge_index = None
                if edge_index_kw is not None:
                    try:
                        ei = edge_index_kw
                        ei_arr = np.asarray(ei)
                        if ei_arr.ndim == 2 and ei_arr.shape[0] == 2:
                            edge_index = torch.tensor(ei_arr, dtype=torch.long)
                        elif ei_arr.ndim == 2 and ei_arr.shape[1] == 2:
                            edge_index = torch.tensor(ei_arr.T, dtype=torch.long)
                    except Exception:
                        edge_index = None
                if edge_index is None:
                    rows = []
                    cols = []
                    for i in range(int(last.size)):
                        for j in range(int(last.size)):
                            if i != j:
                                rows.append(i)
                                cols.append(j)
                    edge_index = torch.tensor([rows, cols], dtype=torch.long) if rows else torch.empty((2, 0), dtype=torch.long)

                x = torch.tensor(last.reshape(-1, 1), dtype=torch.float32)

                do_train = bool(kwargs.get('train', False) or (isinstance(model_args, dict) and model_args.get('train', False)))
                if do_train:
                    epochs = int(model_args.get('train_epochs', 5)) if isinstance(model_args, dict) else 5
                    lr = float(model_args.get('lr', 0.01)) if isinstance(model_args, dict) else 0.01
                    window = int(model_args.get('window_size', 1)) if isinstance(model_args, dict) else 1

                    X_samples = []
                    Y_samples = []
                    for t_idx in range(window, T):
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
                        for ep in range(epochs):
                            for xi, yi in zip(X_samples, Y_samples):
                                x_in = torch.tensor(xi, dtype=torch.float32)
                                y_true = torch.tensor(yi, dtype=torch.float32)
                                opt.zero_grad()
                                try:
                                    out = conv(x_in, edge_index)
                                except Exception:
                                    out = conv(x_in, torch.empty((2, 0), dtype=torch.long))
                                h = torch.relu(out)
                                pred = readout(h).squeeze()
                                loss = loss_fn(pred, y_true)
                                loss.backward()
                                opt.step()

                        x_last = torch.tensor(last.reshape(-1, 1), dtype=torch.float32)
                        try:
                            out = conv(x_last, edge_index)
                        except Exception:
                            out = conv(x_last, torch.empty((2, 0), dtype=torch.long))
                        h = torch.relu(out)
                        pred_out = readout(h).squeeze().detach().numpy()
                        delta = (last - prev)
                        pred_arr = _np.zeros((horizon, int(last.size)), dtype=_np.float32)
                        for h_idx in range(horizon):
                            step = pred_out + (h_idx) * delta
                            pred_arr[h_idx, :] = step
                        return pred_arr

                # inference-only
                conv = GCNConv(1, 1)
                with torch.no_grad():
                    try:
                        out = conv(x, edge_index)
                    except Exception:
                        out = None
                neigh = out.squeeze().numpy() if out is not None else last
                delta = (last - prev)
                pred_arr = _np.zeros((horizon, int(last.size)), dtype=_np.float32)
                for h_idx in range(horizon):
                    step = last + (h_idx + 1) * delta + 0.05 * neigh
                    pred_arr[h_idx, :] = step
                return pred_arr
    except Exception:
        # torch or torch_geometric not available or failed
        pass

    # fallback: linear extrapolation per-node using last two values
    try:
        if isinstance(arr, _np.ndarray) and arr.size > 0:
            if arr.ndim == 1:
                vals = arr.tolist()
                delta = float(vals[-1]) - float(vals[-2]) if len(vals) >= 2 else 0.0
                out = _np.array([float(vals[-1] if vals else 0.0) + (i + 1) * delta for i in range(horizon)], dtype=_np.float32)
                return out
            else:
                last = arr.reshape((arr.shape[0], -1))[-1]
                prev = arr.reshape((arr.shape[0], -1))[-2] if arr.shape[0] >= 2 else last
                last = np.asarray(last, dtype=_np.float32)
                prev = np.asarray(prev, dtype=_np.float32)
                delta = last - prev
                N = int(last.shape[0])
                out = _np.zeros((horizon, N), dtype=_np.float32)
                for h_idx in range(horizon):
                    out[h_idx, :] = last + (h_idx + 1) * delta
                return out
    except Exception:
        pass

    # ultimate scalar fallback
    return _np.zeros((horizon,), dtype=_np.float32)



def _standardize_pred(pred: Any, horizon: Optional[int] = None) -> ArrayND:
    """Coerce runner output to numpy array with shape (H, N, F).

    Rules:
    - 0D scalar -> (1,1,1)
    - 1D (H,) -> (H,1,1)
    - 2D (H,N) -> (H,N,1)
    - 3D (H,N,F) -> (H,N,F) (no-op)
    - If horizon is provided and mismatches, keep returned H when possible.
    """
    try:
        a = np.asarray(pred)
    except Exception:
        # non-array-like -> return empty 3D array
        return np.zeros((0, 0, 1), dtype=np.float32)

    # ensure float32 for consistency with existing runners
    try:
        a = a.astype(np.float32)
    except Exception:
        a = np.asarray(a, dtype=np.float32)

    if a.ndim == 0:
        return a.reshape((1, 1, 1))
    if a.ndim == 1:
        H = a.shape[0]
        return a.reshape((H, 1, 1))
    if a.ndim == 2:
        H, N = a.shape
        return a.reshape((H, N, 1))
    # ndim >= 3: assume (H, N, F) or compatible
    if a.ndim >= 3:
        # if extra leading dims exist, try to preserve first three axes
        return a.reshape((a.shape[0], a.shape[1], a.shape[2]))
    # fallback
    return a.reshape((a.shape[0], -1, 1))


def _wrap_runner(fn):
    """Return a wrapper that calls fn and coerces its output to (H,N,F).

    The wrapper accepts the same (data, **kwargs) signature as runners.
    """
    def wrapped(data: Any, **kwargs: Any) -> ArrayND:
        horizon = int(kwargs.get('horizon', 3)) if kwargs is not None else 3
        out = fn(data, **kwargs)
        # if runner returned CSV text, parse it first
        if isinstance(out, str):
            try:
                out_arr = _csv_to_array(out)
            except Exception:
                out_arr = np.asarray(out)
        else:
            out_arr = np.asarray(out)
        return _standardize_pred(out_arr, horizon=horizon)

    return wrapped


# Wrap runners so every runner returns a standardized 3D numpy array (H, N, F)
RUNNERS = {
    "pysteps": _wrap_runner(run_pysteps),
    "sktime": _wrap_runner(run_sktime),
    "tslearn": _wrap_runner(run_tslearn),
    "torch_geometric": _wrap_runner(run_torch_geometric),
}


def _array_to_csv(arr: Any) -> str:
    """Convert numpy array (H,) or (H,N) or (H,N,1) to CSV text with headers.

    - (H,) -> header ['horizon','forecast'] rows (1..H, value)
    - (H,N) -> header ['horizon','node','forecast'] rows repeated per node
    - scalar-like or other -> fallback to string
    """
    try:
        a = np.asarray(arr)
    except Exception:
        return str(arr)
    import io as _io
    import csv as _csv
    buf = _io.StringIO()
    writer = _csv.writer(buf)
    if a.ndim == 1:
        writer.writerow(['horizon', 'forecast'])
        for i in range(a.shape[0]):
            writer.writerow([i + 1, float(a[i])])
        return buf.getvalue()
    if a.ndim == 2:
        H, N = a.shape[0], a.shape[1]
        writer.writerow(['horizon', 'node', 'forecast'])
        for h in range(H):
            for n in range(N):
                writer.writerow([h + 1, n, float(a[h, n])])
        return buf.getvalue()
    if a.ndim >= 3:
        # take first feature channel
        H = a.shape[0]
        N = a.shape[1] if a.ndim >= 2 else 1
        writer.writerow(['horizon', 'node', 'forecast'])
        for h in range(H):
            for n in range(N):
                try:
                    v = float(a[h, n, 0])
                except Exception:
                    try:
                        v = float(a[h, n])
                    except Exception:
                        v = 0.0
                writer.writerow([h + 1, n, v])
        return buf.getvalue()
    return str(arr)


def run_model(lib_name: str, csv_text: str, **kwargs: Any) -> Tuple[str, str]:
    """Run mock model and return (code_py, predictions_csv).

    Any additional keyword arguments are forwarded to the underlying runner
    (for example `horizon` or `edge_index`).
    """
    func = RUNNERS.get(lib_name)
    if not func:
        func = run_sktime

    # accept array or CSV: callers might pass CSV; convert to array first if needed
    if isinstance(csv_text, str):
        try:
            arr_in = _csv_to_array(csv_text)
        except Exception:
            arr_in = csv_text
    else:
        arr_in = csv_text

    # run runner which now returns numpy arrays
    try:
        pred_arr = func(arr_in, **kwargs)
    except Exception:
        # try legacy CSV-based call
        try:
            csv_in = csv_text if isinstance(csv_text, str) else _array_to_csv(csv_text)
            out_csv = func(csv_in, **kwargs)
            pred_arr = _csv_to_array(out_csv)
        except Exception:
            pred_arr = np.zeros((0,), dtype=np.float32)

    # convert predictions to CSV for backwards compatibility
    pred_csv = _array_to_csv(pred_arr)

    code = f"# Mock runner for {lib_name}\n" + "def predict(csv_text):\n    # implement model loading and prediction here\n    return '" + pred_csv.replace("\n", "\\n") + "'\n"
    return code, pred_csv


def _csv_to_array(csv_text: str) -> Union[ArrayND, List[float]]:
    """Parse CSV with header (t,node,value) into numpy array of shape (T,N,1).
    If CSV has a different layout, try to return a 2D array (T, features).
    """
    import csv
    from collections import defaultdict

    lines = [line for line in csv_text.splitlines() if line.strip()]
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
        vals.append(float(row[-1]))
    return np.array(vals, dtype=np.float32)


def run_model_from_choice(choice: Union[ModelChoice, str], input_data: Any, horizon: int = 3, **kwargs: Any) -> Dict[str, Any]:
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

    # arr is already loaded above; pass numpy array directly to runners
    csv_text = None

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

    # call runner with array input; try array-first API

    csv_in = _array_to_csv(arr)
    out = func(csv_in, **forward_kwargs)
    # out may be CSV text or array
    if isinstance(out, str):
        pred_arr = _csv_to_array(out)
    else:
        pred_arr = np.asarray(out)


    predictions_csv = _array_to_csv(pred_arr)

    result = {
        'library': lib,
        'y_pred': pred_arr,
        'csv': predictions_csv,
        'meta': {'horizon': horizon}
    }
    return result
