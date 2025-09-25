import importlib


def test_run_service_uses_selector_and_runs_model(monkeypatch):
    rs = importlib.import_module('app.runner_service')
    llm = importlib.import_module('app.llm')

    # fake selector returns a choice dict
    def fake_select_model_rag(desc, task, db=None, top_k=4, hf_model_name=None):
        return {'library': 'sktime', 'model_name': 'AutoARIMA', 'model_args': {}}

    monkeypatch.setattr(llm, 'select_model_rag', fake_select_model_rag)

    # fake runner to avoid heavy model execution
    def fake_run_model_from_choice(choice, input_data, horizon=3, **kwargs):
        return {'library': choice.get('library'), 'y_pred': [0] * horizon, 'meta': {'horizon': horizon}}

    mr = importlib.import_module('app.model_runner')
    # runner_service imported run_model_from_choice at module import time, so patch there
    rs_mod = importlib.import_module('app.runner_service')
    monkeypatch.setattr(rs_mod, 'run_model_from_choice', fake_run_model_from_choice)

    # prepare input and call service
    data = [1.0, 2.0, 3.0, 4.0]
    out = rs.run_service('demo dataset', data, db=None, task='forecast', horizon=2)
    assert isinstance(out, dict)
    assert 'choice' in out and 'result' in out
    # y_pred may be numpy array or list; normalize to list for assertion
    y = out['result']['y_pred']
    try:
        y_list = [float(x) for x in list(y)]
    except Exception:
        try:
            y_list = [float(y)]
        except Exception:
            y_list = [y]
    assert y_list == [0.0, 0.0]
