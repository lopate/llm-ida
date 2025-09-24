import numpy as np
from app import model_runner


def test_forward_model_args_from_choice(monkeypatch):
    captured = {}

    def dummy(csv_text, **kwargs):
        # capture forwarded kwargs
        captured.update(kwargs)
        return "horizon,forecast\n1,0.1\n2,0.2\n"

    monkeypatch.setitem(model_runner.RUNNERS, 'sktime', dummy)

    choice = {'library': 'sktime', 'model_name': 'my_model', 'model_args': {'a': 1}}
    res = model_runner.run_model_from_choice(choice, np.array([1, 2, 3]), horizon=2)

    assert 'model_args' in captured
    assert captured['model_args'] == {'a': 1}
    assert captured.get('model_name') == 'my_model'
    assert captured.get('horizon') == 2


def test_forward_model_args_override(monkeypatch):
    captured = {}

    def dummy(csv_text, **kwargs):
        captured.update(kwargs)
        return "horizon,forecast\n1,0.1\n2,0.2\n"

    monkeypatch.setitem(model_runner.RUNNERS, 'sktime', dummy)

    choice = {'library': 'sktime', 'model_name': 'my_model', 'model_args': {'a': 1}}
    # pass model_args via kwargs to override/extend
    res = model_runner.run_model_from_choice(choice, np.array([1, 2, 3]), horizon=2, model_args={'b': 2})

    assert 'model_args' in captured
    # merged dict should contain both keys a and b, kwargs wins on conflicts
    assert captured['model_args'] == {'a': 1, 'b': 2}
    assert captured.get('model_name') == 'my_model'
    assert captured.get('horizon') == 2
