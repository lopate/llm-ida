"""Vector DB wrapper with optional Chromadb or FAISS backends.
If chromadb/faiss missing, fall back to in-memory store.
Provides robust initialization and safe operations for memory backend.
Supports FAISS on-disk save/load via env var `FAISS_PATH` and autosave via `FAISS_AUTOSAVE`.
"""
from typing import List, Dict, Optional
import threading
import os
import math

_backend: Optional[str] = None
_client = None
_collection = None
_lock = threading.Lock()
_faiss_index_path: Optional[str] = None
_faiss_autosave: bool = False


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
            D, I = client["index"].search(vec, k)
            out = []
            for idx, dist in zip(I[0], D[0]):
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

"""Vector DB wrapper with optional Chromadb or FAISS backends.
If chromadb or faiss are not installed, falls back to an in-memory store.
"""
from typing import List, Dict, Optional
import threading

_backend = None
_client = None
_collection = None
_lock = threading.Lock()


def _init_chroma():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except Exception:
        return False

    client = chromadb.Client()
    # create or get collection
    try:
        col = client.get_collection(name="models")
    except Exception:
        col = client.create_collection(name="models")

    global _backend, _client, _collection
    _backend = "chroma"
    _client = client
    _collection = col
    return True


def _init_faiss():
    try:
        import faiss
        import numpy as np
    except Exception:
        return False

    # simple in-memory FAISS index
    global _backend, _client, _collection
    _backend = "faiss"
    _client = {"index": None, "vectors": [], "metadatas": []}
    return True


def initialize():
    # Try chroma, then faiss, else memory
    if _init_chroma():
        return True
    if _init_faiss():
        return True
    # fallback to memory
    global _backend, _client
    _backend = "memory"
    _client = {"vectors": [], "metadatas": [], "ids": []}
    return True


def available():
    return _backend is not None


def upsert(ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
    """Insert or update vectors into the backend."""
    with _lock:
        if _backend == "chroma":
            # chroma expects list of dicts or list of metadatas
            _collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            return True

        if _backend == "faiss":
            import numpy as np
            import faiss
            vecs = np.array(embeddings).astype('float32')
            # append
            client = _client
            if client["index"] is None:
                d = vecs.shape[1]
                index = faiss.IndexFlatIP(d)
                client["index"] = index
            else:
                index = client["index"]
            index.add(vecs)
            client["vectors"].extend(vecs.tolist())
            client["metadatas"].extend(metadatas)
            return True

        # memory
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
            # chroma returns dict with 'ids','metadatas','distances'
            metadatas = res.get('metadatas', [[]])[0]
            distances = res.get('distances', [[]])[0]
            out = []
            for m, d in zip(metadatas, distances):
                out.append((m, float(1 - d)))
            return out

        if _backend == "faiss":
            import numpy as np
            import faiss
            client = _client
            if client["index"] is None:
                return []
            vec = np.array(embedding).astype('float32').reshape(1, -1)
            D, I = client["index"].search(vec, k)
            out = []
            for idx, dist in zip(I[0], D[0]):
                md = client["metadatas"][idx] if idx < len(client["metadatas"]) else {}
                out.append((md, float(dist)))
            return out

        # memory fallback: cosine similarity
        client = _client
        if not client["vectors"]:
            return []
        import math
        best = []
        for vec, md in zip(client["vectors"], client["metadatas"]):
            dot = sum(x * y for x, y in zip(vec, embedding))
            na = math.sqrt(sum(x * x for x in vec))
            nb = math.sqrt(sum(y * y for y in embedding))
            score = dot / (na * nb + 1e-12)
            best.append((md, score))
        best.sort(key=lambda x: x[1], reverse=True)
        return best[:k]


    # --- FAISS on-disk support ---
    _faiss_index_path = None


    def save_faiss(path: str):
        """Save FAISS index and metadata to disk (index -> path.index, metadata -> path.meta.json)"""
        try:
            import json
            import os
            import faiss
        except Exception as e:
            raise RuntimeError(f"FAISS save requires faiss installed: {e}")

        if _backend != "faiss":
            raise RuntimeError("FAISS backend not initialized")
        client = _client
        if client["index"] is None:
            raise RuntimeError("FAISS index empty")

        idx_path = path + ".index"
        meta_path = path + ".meta.json"
        faiss.write_index(client["index"], idx_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(client["metadatas"], f, ensure_ascii=False)
        return idx_path, meta_path


    def load_faiss(path: str):
        """Load FAISS index and metadata from disk (path.index and path.meta.json)"""
        try:
            import json
            import os
            import numpy as np
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
        # try to load into faiss regardless of current backend
        return load_faiss(path)

