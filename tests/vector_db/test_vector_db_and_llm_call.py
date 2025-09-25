from app.vector_db import VectorDB


def test_vector_db_and_llm_call_smoke():
    # ensure VectorDB class is importable
    assert hasattr(VectorDB, 'text_to_embedding')
