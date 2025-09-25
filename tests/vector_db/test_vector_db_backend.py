from app.vector_db import VectorDB


def test_backend_smoke():
    db = VectorDB()
    assert hasattr(db, 'upsert')
