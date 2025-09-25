import importlib


def test_seed_default_models_upserts(monkeypatch):
    vdb = importlib.import_module('app.vector_db')

    called = {'upsert': False}

    def fake_upsert(ids, embs, metas):
        called['upsert'] = True
        # ensure lists non-empty and lengths match
        assert isinstance(ids, list) and isinstance(embs, list) and isinstance(metas, list)
        assert len(ids) == len(embs) == len(metas)
        return True

    monkeypatch.setattr(vdb, 'upsert', fake_upsert)

    ok = vdb.seed_default_models()
    assert ok is True
    assert called['upsert'] is True
