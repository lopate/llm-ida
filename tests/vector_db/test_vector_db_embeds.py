from app.vector_db import VectorDB


def test_embeds_smoke():
    db = VectorDB(backend='memory')
    v = db.text_to_embedding('x', dim=4)
    assert isinstance(v, list)
