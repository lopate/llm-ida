from app import vector_db


def test_vector_db_and_llm_call_smoke():
    # simple smoke to ensure module import
    assert hasattr(vector_db, '_text_to_embedding')
