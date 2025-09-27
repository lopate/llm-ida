"""Vector DB wrapper implemented as a class with FAISS / Chroma / memory backends.
This module exposes a `VectorDB` class and a module-level `default_db` instance for
backwards compatibility with existing call sites that import `app.vector_db`.

Only a minimal, well-contained implementation is provided here to keep tests
fast and deterministic. FAISS/Chroma support is attempted if the libraries are
available at runtime; otherwise a memory-only backend is used.
"""

from typing import List, Dict, Optional, Any, Tuple
import threading
import os
import math
import json
import re


DEFAULT_DOCS = [
    {
        'id': 'pysteps_persistence',
        'model_name': 'persistence',
        'library': 'pysteps',
        'description': 'Persistence and optical-flow based nowcasting for radar/gridded fields; fast, low-latency classical nowcasting. Example usage: load radar grid, apply advection/optical-flow, extrapolate frames for short horizons. Good for sub-hour nowcasts where motion dominates.'
    },
    {
        'id': 'statsforecast_autoarima',
        'model_name': 'StatsForecastAutoARIMA',
        'library': 'statsforecast',
        'description': 'Automatic ARIMA model selection and forecasting for univariate time series using the statsforecast backend. Usage: fit StatsForecastAutoARIMA on each sensor series for fast, scalable ARIMA forecasts with seasonal adjustments; reliable baseline for hourly/daily seasonal series.'
    },
    {
        'id': 'sktime_autoets',
        'model_name': 'AutoETS',
        'library': 'sktime',
        'description': 'Automatic ETS (error-trend-seasonality) modelling for univariate series. AutoETS selects an exponential smoothing state space model automatically and is a robust baseline for seasonal/time-series exhibiting trend and/or multiplicative seasonality.'
    },
    {
        'id': 'sktime_forest',
        'model_name': 'forest',
        'library': 'sktime',
        'description': 'ReducedRegressionForecaster using RandomForestRegressor. Usage: reduce multivariate/time-window into feature vectors per timestep and train RandomForest; handles nonlinearities and is robust to missing values.'
    },
    {
        'id': 'sktime_naive',
        'model_name': 'naive',
        'library': 'sktime',
        'description': 'Simple persistence / last-value forecaster (naive). Fast baseline that repeats the last observed value for the forecast horizon; useful as a lightweight fallback when data is sparse or complex models are unavailable.'
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


class VectorDB:
    """Simple Vector DB with pluggable backends.

    Backends supported:
    - 'memory' : in-process Python lists (always available, fast for tests)
    - 'faiss'  : if faiss is importable (best for larger datasets)
    - 'chroma' : if chromadb is importable

    The class keeps all state on the instance. Use `default_db` for the
    module-level instance used by older code.
    """

    def __init__(self, backend: Optional[str] = None, faiss_path: Optional[str] = None, faiss_autosave: bool = False):
        self._backend = backend
        self._client: Any = None
        self._collection = None
        self._lock = threading.Lock()
        self._faiss_index_path = faiss_path
        self._faiss_autosave = faiss_autosave
        self._seeded = False
        # lazy sentence-transformer state
        self._st_model = None
        self._st_encoder_dim = None
        # memory backend storage
        self._memory_ids: List[str] = []
        self._memory_vecs: List[List[float]] = []
        self._memory_meta: List[Dict[str, Any]] = []

    def _text_to_embedding(self, text: Optional[str], dim: int = 128) -> List[float]:
        """Return a deterministic embedding for text.

        Prefer sentence-transformers if available, otherwise fall back to a
        hash-based sparse vector. The result is L2-normalized.
        """
        # Try sentence-transformers lazily
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                name = os.environ.get("ST_MODEL", "all-MiniLM-L6-v2")
                self._st_model = SentenceTransformer(name)
                try:
                    self._st_encoder_dim = self._st_model.get_sentence_embedding_dimension()
                except Exception:
                    self._st_encoder_dim = None
            except Exception:
                self._st_model = None
                self._st_encoder_dim = None

        if self._st_model is not None:
            try:
                emb = self._st_model.encode([text or ""], show_progress_bar=False)[0]
                # Convert numpy arrays to lists and normalize
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
                    lst = list(emb)
                    norm = math.sqrt(sum(x * x for x in lst))
                    if norm > 0:
                        lst = [x / norm for x in lst]
                    return lst
            except Exception:
                # fall through to deterministic hash fallback
                pass

        # deterministic hash-based fallback
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
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def initialize(self) -> bool:
        """Ensure the backend is configured and ready. Returns True on success."""
        if self._backend is None:
            # prefer faiss/chroma if available
            try:
                import faiss  # type: ignore
                self._backend = 'faiss'
            except Exception:
                try:
                    import chromadb  # type: ignore
                    self._backend = 'chroma'
                except Exception:
                    self._backend = 'memory'

        if self._backend == 'memory':
            # already ready
            return True

        if self._backend == 'faiss':
            try:
                import faiss  # type: ignore
                dim = 128
                index = faiss.IndexFlatL2(dim)
                self._client = {'index': index, 'metadatas': []}
                return True
            except Exception:
                self._backend = 'memory'
                return True

        if self._backend == 'chroma':
            try:
                import chromadb  # type: ignore
                client = chromadb.Client()
                try:
                    col = client.get_collection(name='models')
                except Exception:
                    col = client.create_collection(name='models')
                self._client = client
                self._collection = col
                return True
            except Exception:
                self._backend = 'memory'
                return True

        return False

    def available(self) -> str:
        return self._backend or 'memory'

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Insert vectors into the backend. For memory backend we append to lists."""
        if metadatas is None:
            metadatas = [{} for _ in ids]
        if self._backend == 'memory':
            self._memory_ids.extend(ids)
            self._memory_vecs.extend(embeddings)
            self._memory_meta.extend(metadatas)
            return True

        if self._backend == 'faiss' and isinstance(self._client, dict) and 'index' in self._client:
            try:
                import numpy as np
                # ensure an index exists; if it's None, try to create one from available faiss classes
                if self._client.get('index') is None:
                    try:
                        import faiss  # type: ignore
                        dim = int(len(embeddings[0]) if embeddings and len(embeddings[0]) else 128)
                        idx_cls = getattr(faiss, 'IndexFlatL2', None) or getattr(faiss, 'IndexFlatIP', None) or getattr(faiss, 'IndexFlat', None)
                        if idx_cls is None:
                            raise RuntimeError('No suitable FAISS index class found')
                        index = idx_cls(dim)
                        self._client['index'] = index
                    except Exception:
                        return False
                arr = np.array(embeddings, dtype='float32')
                self._client['index'].add(arr)
                self._client.setdefault('metadatas', []).extend(metadatas)
                if self._faiss_autosave and self._faiss_index_path:
                    self.save(self._faiss_index_path)
                return True
            except Exception:
                return False

        if self._backend == 'chroma' and self._collection is not None:
            try:
                self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                return True
            except Exception:
                return False

        return False

    def query(self, embedding: List[float], k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """Return up to k (metadata, score) pairs sorted by similarity (cosine approximated by dot on normalized vectors)."""
        if self._backend == 'memory':
            if not self._memory_vecs:
                return []
            def dot(a, b):
                return sum(x * y for x, y in zip(a, b))
            scores = [(i, dot(embedding, v)) for i, v in enumerate(self._memory_vecs)]
            scores.sort(key=lambda x: x[1], reverse=True)
            res = []
            for idx, sc in scores[:k]:
                res.append((self._memory_meta[idx], float(sc)))
            return res

        if self._backend == 'faiss' and isinstance(self._client, dict) and 'index' in self._client:
            try:
                import numpy as np
                vec = np.array([embedding], dtype='float32')
                D, I = self._client['index'].search(vec, k)
                res = []
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    meta = self._client.get('metadatas', [])[int(idx)] if self._client.get('metadatas') else {}
                    res.append((meta, float(dist)))
                return res
            except Exception:
                return []

        if self._backend == 'chroma' and self._collection is not None:
            try:
                results = self._collection.query(query_embeddings=[embedding], n_results=k)
                res = []
                for mlist, slist in zip(results.get('metadatas', []), results.get('distances', [])):
                    for m, s in zip(mlist, slist):
                        res.append((m, float(s)))
                return res
            except Exception:
                return []

        return []

    def seed_default_models(self) -> bool:
        docs = DEFAULT_DOCS
        ids = [d['id'] for d in docs]
        embs = [self._text_to_embedding(d['description']) for d in docs]
        metas = [{'model_name': d['model_name'], 'library': d['library'], 'description': d['description'], 'id': d['id']} for d in docs]
        try:
            return bool(self.upsert(ids, embs, metas))
        except Exception:
            return False

    def search_by_text(self, text: str, k: int = 4):
        emb = self._text_to_embedding(text)
        return self.query(emb, k=k)

    def save(self, path: str):
        # Only FAISS backend supports save; raise for other backends or missing index
        if self._backend == 'faiss' and isinstance(self._client, dict) and 'index' in self._client and self._client.get('index') is not None:
            import faiss
            idx_path = path + '.index'
            meta_path = path + '.meta.json'
            faiss.write_index(self._client['index'], idx_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self._client.get('metadatas', []), f, ensure_ascii=False)
            return idx_path, meta_path

        raise RuntimeError('Save not supported for backend: ' + str(self._backend))

    def load(self, path: str):
        if self._backend == 'faiss':
            import faiss
            idx_path = path + '.index'
            meta_path = path + '.meta.json'
            if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
                raise FileNotFoundError('FAISS index or metadata not found at path')
            index = faiss.read_index(idx_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadatas = json.load(f)
            self._client = {'index': index, 'metadatas': metadatas}
            self._faiss_index_path = path

    # Public helpers to avoid external code mutating private attributes directly
    def set_backend(self, backend: Optional[str]):
        """Set the backend type for this instance. Use initialize() to (re)create client state."""
        self._backend = backend

    def set_client(self, client: Any):
        """Directly set the underlying client/state for advanced tests or integrations.

        Prefer initialize()/upsert/query for normal usage.
        """
        self._client = client

    def set_faiss_options(self, path: Optional[str], autosave: bool = False):
        self._faiss_index_path = path
        self._faiss_autosave = autosave

    def reset_st_model(self):
        """Clear cached sentence-transformers state so it will be lazily reloaded.

        Tests should call this instead of poking `_st_model`/`_st_encoder_dim`.
        """
        self._st_model = None
        self._st_encoder_dim = None

    # Public wrapper for the internal embedding helper (preferred over _text_to_embedding)
    def text_to_embedding(self, text: Optional[str], dim: int = 128) -> List[float]:
        return self._text_to_embedding(text, dim)


# NOTE: default_db and module-level wrapper functions removed intentionally.
# This module now exposes only the VectorDB class and DEFAULT_DOCS. Callers
# must create and manage VectorDB instances explicitly (e.g. via
# `from app.vector_db import VectorDB; db = VectorDB()`).
