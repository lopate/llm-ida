"""Vector DB wrapper with optional Chromadb or FAISS backends.
If chromadb/faiss missing, fall back to in-memory store.
Provides robust initialization and safe operations for memory backend.
Supports FAISS on-disk save/load via env var `FAISS_PATH` and autosave via `FAISS_AUTOSAVE`.
"""
from typing import List, Dict, Optional
import re
import threading
import os
import math

_backend: Optional[str] = None
_client = None
_collection = None
_lock = threading.Lock()
_faiss_index_path: Optional[str] = None
_faiss_autosave: bool = False
_seeded: bool = False

# Default docs used for seeding the vector DB. Kept module-level so we can
# reindex deterministically on-demand.
DEFAULT_DOCS = [
    {
        'id': 'pysteps_persistence',
        'model_name': 'persistence',
        'library': 'pysteps',
        'description': 'Persistence and optical-flow based nowcasting for radar/gridded fields; fast, low-latency classical nowcasting. Example usage: load radar grid, apply advection/optical-flow, extrapolate frames for short horizons. Good for sub-hour nowcasts where motion dominates.'
    },
    {
        'id': 'sktime_autoarima',
        'model_name': 'AutoARIMA',
        'library': 'sktime',
        'description': 'Automatic ARIMA model selection and forecasting for univariate time series. Usage: fit AutoARIMA on each sensor series, forecast with seasonal adjustments; reliable baseline for hourly/daily seasonal series.'
    },
    {
        'id': 'sktime_forest',
        'model_name': 'forest',
        'library': 'sktime',
        'description': 'ReducedRegressionForecaster using RandomForestRegressor. Usage: reduce multivariate/time-window into feature vectors per timestep and train RandomForest; handles nonlinearities and is robust to missing values.'
    },
    {
        'id': 'tslearn_knn',
        'model_name': 'KNeighbors',
        'library': 'tslearn',
        'description': 'Distance-based nearest-neighbour time-series regression (sliding-window KNN). Example: use DTW/Euclidean window distances to find nearest historical patterns and average their next-step values for prediction.'
    },
    {
        'id': 'torchgeom_gat',
        'model_name': 'GAT',
        'library': 'torch_geometric',
        'description': 'Graph-based GAT/GCN models for sensor networks; captures spatial relations via graph convolutions. Usage: construct graph from sensor adjacency, create temporal windows per node, train GAT for multi-step forecasting with attention over neighbors.'
    },
]


def _init_chroma() -> bool:
    try:
        import chromadb
    except Exception:
        return False

    client = chromadb.Client()
    try:
        col = client.get_collection(name="models")
    except Exception:
        col = client.create_collection(name="models")

    global _backend, _client, _collection
    _backend = "chroma"
    _client = client
    _collection = col
    return True


def _init_faiss() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False

    global _backend, _client, _faiss_index_path, _faiss_autosave
    _backend = "faiss"
    # initialize empty structure
    _client = {"index": None, "vectors": [], "metadatas": []}

    # read env vars
    path = os.environ.get("FAISS_PATH")
    autosave = os.environ.get("FAISS_AUTOSAVE")
    _faiss_index_path = path if path else None
    _faiss_autosave = bool(autosave and autosave.lower() in ("1", "true", "yes"))

    # if path provided, try load
    if _faiss_index_path:
        try:
            load_faiss(_faiss_index_path)
        except Exception:
            # ignore load errors
            pass

    return True


def _ensure_memory_client():
    global _client
    if _client is None or not isinstance(_client, dict):
        _client = {"vectors": [], "metadatas": [], "ids": []}


def initialize() -> bool:
    global _backend
    # prefer chroma, then faiss, else memory
    if _init_chroma():
        return True
    if _init_faiss():
        return True
    _backend = "memory"
    _ensure_memory_client()
    # seed default model descriptions for in-memory DB on first init
    try:
        global _seeded
        if not _seeded:
            seed_default_models()
            _seeded = True
    except Exception:
        # seeding is best-effort; don't fail initialization
        pass
    return True


def available() -> bool:
    return _backend is not None


def upsert(ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> bool:
    """Insert or update vectors into the backend."""
    with _lock:
        if _backend == "chroma":
            _collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            return True

        if _backend == "faiss":
            try:
                import numpy as np
                import faiss
            except Exception:
                raise RuntimeError("faiss and numpy required for faiss backend")

            vecs = np.array(embeddings).astype('float32')
            client = _client
            if client.get("index") is None:
                d = vecs.shape[1]
                index = faiss.IndexFlatIP(d)
                client["index"] = index
            else:
                index = client["index"]
            index.add(vecs)
            client["vectors"].extend(vecs.tolist())
            client["metadatas"].extend(metadatas)

            # autosave if configured
            if _faiss_autosave and _faiss_index_path:
                try:
                    save_faiss(_faiss_index_path)
                except Exception:
                    pass

            return True

        # memory backend
        _ensure_memory_client()
        client = _client
        client["vectors"].extend(embeddings)
        client["metadatas"].extend(metadatas)
        client["ids"].extend(ids)
        return True


def query(embedding: List[float], k: int = 1):
    """Query nearest k; returns list of (metadata, score)"""
    with _lock:
        if _backend == "chroma":
            res = _collection.query(query_embeddings=[embedding], n_results=k)
            metadatas = res.get('metadatas', [[]])[0]
            distances = res.get('distances', [[]])[0]
            out = []
            for m, d in zip(metadatas, distances):
                out.append((m, float(1 - d)))
            return out

        if _backend == "faiss":
            try:
                import numpy as np
            except Exception:
                raise RuntimeError("numpy required for faiss query")
            client = _client
            if not client or client.get("index") is None:
                return []
            vec = np.array(embedding).astype('float32').reshape(1, -1)
            D, indices = client["index"].search(vec, k)
            out = []
            for idx, dist in zip(indices[0], D[0]):
                md = client["metadatas"][idx] if idx < len(client["metadatas"]) else {}
                out.append((md, float(dist)))
            return out

        # memory fallback: cosine similarity
        _ensure_memory_client()
        client = _client
        if not client["vectors"]:
            return []
        best = []
        for vec, md in zip(client["vectors"], client["metadatas"]):
            dot = sum(x * y for x, y in zip(vec, embedding))
            na = math.sqrt(sum(x * x for x in vec))
            nb = math.sqrt(sum(y * y for y in embedding))
            score = dot / (na * nb + 1e-12)
            best.append((md, score))
        best.sort(key=lambda x: x[1], reverse=True)
        return best[:k]


# --- simple text embedding and RAG helpers ---
def _text_to_embedding(text: str, dim: int = 64) -> List[float]:
    """Try to use sentence-transformers for quality embeddings, fall back to a
    deterministic hash-based embedding when the package or model is unavailable.

    The sentence-transformers model is loaded lazily and cached for subsequent
    calls. The `dim` parameter is used as a hint for the fallback vector size
    when transformers are not available.
    """
    # lazy cached embedder
    global _st_model, _st_encoder_dim
    if "_st_model" not in globals():
        _st_model = None
        _st_encoder_dim = None

    # Try to use sentence-transformers for higher-quality semantic embeddings.
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # prefer a compact model that works well offline; allow override via env
            st_name = os.environ.get("ST_MODEL", "all-MiniLM-L6-v2")
            _st_model = SentenceTransformer(st_name)
            _st_encoder_dim = getattr(_st_model, 'get_sentence_embedding_dimension', lambda: None)() or None
        except Exception:
            _st_model = None
            _st_encoder_dim = None

    if _st_model is not None:
        try:
            emb = _st_model.encode([text or ""], show_progress_bar=False)[0]
            # convert numpy to list if needed and normalize
            try:
                import numpy as np
                arr = np.array(emb, dtype=float)
                norm = float(np.linalg.norm(arr))
                if norm > 0:
                    arr = (arr / norm).tolist()
                else:
                    arr = arr.tolist()
                return arr
            except Exception:
                # fallback to simple list
                lst = list(emb)
                norm = math.sqrt(sum(x * x for x in lst))
                if norm > 0:
                    lst = [x / norm for x in lst]
                return lst
        except Exception:
            # if an error occurs with sentence-transformers, fall back below
            pass

    # --- deterministic hash-based fallback ---
    vec = [0.0] * dim
    try:
        txt = (text or "").lower()
        words = [w for w in re.split(r"\W+", txt) if w]
    except Exception:
        words = []
    if not words:
        return vec
    for w in words:
        h = abs(hash(w))
        idx = h % dim
        vec[idx] += 1.0
    # normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def seed_default_models() -> bool:
    """Seed the vector DB with descriptions of implemented models (best-effort).

    Each entry metadata contains: {'id','model_name','library','description'}.
    This implementation uses the module-level DEFAULT_DOCS and attempts to
    persist the index for FAISS/Chroma backends where possible.
    """
    try:
        initialize()
    except Exception:
        pass

    docs = DEFAULT_DOCS
    ids = []
    embs = []
    metas = []
    for d in docs:
        ids.append(d['id'])
        embs.append(_text_to_embedding(d['description']))
        metas.append({'model_name': d['model_name'], 'library': d['library'], 'description': d['description'], 'id': d['id']})
    try:
        upsert(ids, embs, metas)
    except Exception:
        # best-effort
        return False

    # Attempt best-effort persisting for FAISS / Chroma
    try:
        if _backend == 'faiss':
            if _faiss_index_path or _faiss_autosave:
                try:
                    save_faiss(_faiss_index_path or 'faiss_index')
                except Exception:
                    pass
        elif _backend == 'chroma':
            try:
                if hasattr(_client, 'persist'):
                    _client.persist()
                if hasattr(_collection, 'persist'):
                    _collection.persist()
            except Exception:
                pass
    except Exception:
        pass

    return True


def search_by_text(text: str, k: int = 4):
    """Compute embedding for text and query vector DB; return list of (metadata, score)."""
    emb = _text_to_embedding(text)
    return query(emb, k=k)


# --- FAISS on-disk support ---
def save_faiss(path: str):
    try:
        import json
        import faiss
    except Exception as e:
        raise RuntimeError(f"FAISS save requires faiss installed: {e}")

    if _backend != "faiss":
        raise RuntimeError("FAISS backend not initialized")
    client = _client
    if client is None or client.get("index") is None:
        raise RuntimeError("FAISS index empty")

    idx_path = path + ".index"
    meta_path = path + ".meta.json"
    faiss.write_index(client["index"], idx_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(client.get("metadatas", []), f, ensure_ascii=False)
    return idx_path, meta_path


def load_faiss(path: str):
    try:
        import json
        import os
        import faiss
    except Exception as e:
        raise RuntimeError(f"FAISS load requires faiss installed: {e}")

    idx_path = path + ".index"
    meta_path = path + ".meta.json"
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("FAISS index or metadata not found at path")

    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    global _backend, _client, _faiss_index_path
    _backend = "faiss"
    _client = {"index": index, "vectors": [], "metadatas": metadatas}
    _faiss_index_path = path
    return True


def save(path: str):
    if _backend == "faiss":
        return save_faiss(path)
    raise RuntimeError("Save not supported for backend: " + str(_backend))


def load(path: str):
    return load_faiss(path)


