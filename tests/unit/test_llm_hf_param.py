import importlib


def test_select_model_fallback_when_transformers_fail(monkeypatch):
    llm = importlib.import_module('app.llm')

    # monkeypatch call_transformers to raise
    def fake_call_transformers(prompt, model_name):
        raise RuntimeError('transformers not available')

    monkeypatch.setattr(llm, 'call_transformers', fake_call_transformers)

    # should fall back to rules-based selection
    res = llm.select_model('Radar reflectivity', 'nowcasting', ['pysteps', 'sktime'], hf_model_name='nonexistent-model')
    assert isinstance(res, dict)
    assert res.get('library') == 'pysteps'


def test_select_model_uses_call_transformers_when_available(monkeypatch):
    llm = importlib.import_module('app.llm')

    def fake_call_transformers(prompt, model_name):
        return {'library': 'fake_lib', 'model_choice': 'fake', 'model_name': 'fake_model'}

    monkeypatch.setattr(llm, 'call_transformers', fake_call_transformers)

    res = llm.select_model('some dataset', 'forecast', ['sktime'], hf_model_name='any-model')
    assert isinstance(res, dict)
    assert res.get('library') == 'fake_lib'
