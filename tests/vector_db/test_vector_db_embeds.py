from app import vector_db


def test_embeds_smoke():
    v = vector_db._text_to_embedding('x', dim=4)
    assert isinstance(v, list)
