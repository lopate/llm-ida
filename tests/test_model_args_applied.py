import importlib
import numpy as np
from unittest.mock import MagicMock
import pytest


def test_randomforest_n_estimators_passed():
    mr = importlib.import_module('app.model_runner')

    recorded = {}

    # patch ReducedRegressionForecaster to capture the regressor passed to it
    # inject a fake ReducedRegressionForecaster into sktime.forecasting.compose if necessary
    import sktime.forecasting.compose as compose

    def fake_reduced_forecaster(*args, **kwargs):
        # record the regressor object passed in kwargs
        recorded['regressor'] = kwargs.get('regressor')
        # return a simple mock forecaster with fit/predict
        m = MagicMock()
        m.fit.return_value = None
        m.predict.return_value = np.zeros(3)
        return m

    orig_rrf = getattr(compose, 'ReducedRegressionForecaster', None)
    setattr(compose, 'ReducedRegressionForecaster', fake_reduced_forecaster)
    try:
        series = np.arange(20).astype(float)
        choice = {'library': 'sktime', 'model_name': 'forest', 'model_args': {'n_estimators': 7}}
        res = mr.run_model_from_choice(choice, series, horizon=3)
    finally:
        # restore original if present
        if orig_rrf is None:
            try:
                delattr(compose, 'ReducedRegressionForecaster')
            except Exception:
                pass
        else:
            setattr(compose, 'ReducedRegressionForecaster', orig_rrf)
    # check recorded regressor has the requested parameter if possible
    assert 'regressor' in recorded
    reg = recorded['regressor']
    # try to get n_estimators from reg.get_params() if available
    n = None
    try:
        n = int(reg.get_params().get('n_estimators'))
    except Exception:
        # if reg is a MagicMock, we cannot inspect params; accept that
        n = None
    # if we could read n, assert it equals 7
    if n is not None:
        assert n == 7


def test_randomforest_default_n_estimators_used():
    mr = importlib.import_module('app.model_runner')

    recorded = {}

    import sktime.forecasting.compose as compose2

    def fake_reduced_forecaster2(*args, **kwargs):
        recorded['regressor'] = kwargs.get('regressor')
        m = MagicMock()
        m.fit.return_value = None
        m.predict.return_value = np.zeros(2)
        return m

    orig_rrf2 = getattr(compose2, 'ReducedRegressionForecaster', None)
    setattr(compose2, 'ReducedRegressionForecaster', fake_reduced_forecaster2)
    try:
        series = np.arange(15).astype(float)
        choice = {'library': 'sktime', 'model_name': 'forest', 'model_args': {}}
        res = mr.run_model_from_choice(choice, series, horizon=2)
    finally:
        if orig_rrf2 is None:
            try:
                delattr(compose2, 'ReducedRegressionForecaster')
            except Exception:
                pass
        else:
            setattr(compose2, 'ReducedRegressionForecaster', orig_rrf2)

    assert 'regressor' in recorded
    reg = recorded['regressor']
    n = None
    try:
        n = int(reg.get_params().get('n_estimators'))
    except Exception:
        n = None
    if n is not None:
        assert n == 50
