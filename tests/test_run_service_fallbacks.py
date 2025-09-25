import importlib


def test_run_service_fallback_univariate(monkeypatch):
    rs = importlib.import_module('app.runner_service')
    llm = importlib.import_module('app.llm')

    # make selector raise to force fallback
    def fail_selector(*args, **kwargs):
        raise RuntimeError('selector fail')

    monkeypatch.setattr(llm, 'select_model_rag', fail_selector)

    data = [1.0, 2.0, 3.0]
    out = rs.run_service('desc', data, db=None, task='forecast', horizon=2)
    # fallback heuristic for 1D -> sktime
    assert out['choice'].get('library') == 'sktime'


def test_run_service_fallback_sensors(monkeypatch):
    rs = importlib.import_module('app.runner_service')
    llm = importlib.import_module('app.llm')

    def fail_selector(*args, **kwargs):
        raise RuntimeError('selector fail')

    monkeypatch.setattr(llm, 'select_model_rag', fail_selector)

    import numpy as np
    X = np.zeros((20, 5))  # 2D -> sensors -> tslearn
    out = rs.run_service('desc', X, db=None, task='forecast', horizon=2)
    assert out['choice'].get('library') == 'tslearn'
