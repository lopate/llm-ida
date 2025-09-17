from typing import List, Dict, Optional
import os
import math

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

from app import vector_db


# Local model catalogue
MODEL_DB = [
    {"id": "nowcast_pysteps", "name": "nowcast_pysteps", "library": "pysteps", "desc": "radar nowcasting using optical flow"},
    {"id": "forecast_sktime", "name": "forecast_sktime", "library": "sktime", "desc": "classical forecasting models"},
    {"id": "shape_tslearn", "name": "shape_tslearn", "library": "tslearn", "desc": "shape-based clustering and DTW"},
    {"id": "graph_tg", "name": "graph_tg", "library": "torch_geometric", "desc": "graph neural networks for sensor networks"},
]


def _embed_texts(texts: List[str]):
    model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if SentenceTransformer is not None:
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True)
    # fallback: simple char-level hashing
    embeddings = []
    for t in texts:
        vec = [0.0] * 64
        for i, ch in enumerate(t[:256]):
            vec[i % 64] += ord(ch)
        embeddings.append(vec)
    return embeddings


def init_vector_db():
    # initialize vector DB and upsert model descriptions
    vector_db.initialize()
    texts = [m["desc"] for m in MODEL_DB]
    ids = [m["id"] for m in MODEL_DB]
    embs = _embed_texts(texts)
    metadatas = [{"name": m["name"], "library": m["library"], "desc": m["desc"]} for m in MODEL_DB]
    vector_db.upsert(ids, embs, metadatas)


def find_best_model(query_text: str) -> Dict:
    # ensure DB initialized
    if not vector_db.available():
        init_vector_db()

    q_emb = _embed_texts([query_text])[0]
    res = vector_db.query(q_emb, k=1)
    if not res:
        # fallback to simple heuristic
        # reuse previous logic
        for m in MODEL_DB:
            if "graph" in query_text and m["library"] == "torch_geometric":
                return {"selected": m, "score": 0.9}
        for m in MODEL_DB:
            if "radar" in query_text and m["library"] == "pysteps":
                return {"selected": m, "score": 0.8}
        return {"selected": MODEL_DB[1], "score": 0.5}

    md, score = res[0]
    selected = {"id": md.get("name"), "name": md.get("name"), "library": md.get("library"), "desc": md.get("desc")}
    return {"selected": selected, "score": float(score)}

