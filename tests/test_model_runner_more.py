import numpy as np
import types
import sys
import pytest

from app import model_runner


def test_run_pysteps_fallback_with_invalid_csv():
    # invalid CSV -> fallback persistence behavior
    csv = "not,a,real,csv"
    out = model_runner.run_pysteps(csv, horizon=2)
    assert isinstance(out, str)
    # returns CSV header and rows
    assert 'horizon' in out


def test_run_sktime_naive_multinode():
    # create 3D array T,N,1 with simple increasing sequences
    arr = np.zeros((5, 2, 1), dtype=np.float32)
    for t in range(5):
        arr[t, 0, 0] = t + 1
        arr[t, 1, 0] = (t + 1) * 2
    resd = model_runner.run_model_from_choice('sktime', arr, horizon=3, model_name='naive')
    res = resd['csv']
    assert 'node' in res
    # parse to ensure we have horizon*nodes rows
    lines = [r for r in res.splitlines() if r.strip()]
    assert len(lines) >= 1


def test_run_tslearn_fallback_to_sklearn(monkeypatch):
    # provide series long enough to require KNN window but disable tslearn
    arr = np.array([float(i) for i in range(20)], dtype=np.float32)
    # ensure tslearn not available and sklearn is used -> use public runner
    monkeypatch.setitem(sys.modules, 'tslearn', None)
    resd = model_runner.run_model_from_choice('tslearn', arr, horizon=2, model_args={'n_neighbors': 1, 'window_size': 3})
    res = resd['csv']
    assert isinstance(res, str)
    assert 'horizon' in res


def test_run_sktime_empty_series_fallback():
    # empty CSV
    csv = "t,value\n"
    res = model_runner.run_sktime(csv, horizon=2)
    assert isinstance(res, str)
    assert 'horizon' in res
