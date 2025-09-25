from app.vector_db import VectorDB


def test_more_smoke():
    # trivial smoke to ensure VectorDB class is importable
    v = VectorDB()
    assert v is not None
