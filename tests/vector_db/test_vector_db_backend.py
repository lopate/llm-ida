from app import vector_db


def test_backend_smoke():
    assert hasattr(vector_db, 'upsert')
