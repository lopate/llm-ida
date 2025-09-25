# Temporary demo runner for testing run_service with VECTOR_DB
from pathlib import Path
import sys
repo = Path(__file__).resolve().parents[1]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

print('Python:', sys.executable)
try:
    from app.model_selector import VECTOR_DB
    from app.runner_service import run_service
    print('Imported app.model_selector.VECTOR_DB and run_service')
except Exception as e:
    print('Import error:', e)
    raise

# Init and seed
try:
    VECTOR_DB.initialize()
    VECTOR_DB.seed_default_models()
    print('Vector DB initialized and seeded')
except Exception as e:
    print('Vector DB init/seed warning:', e)

import numpy as np

def gen_univariate(T=50, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = 0.02 * t + np.sin(2*np.pi*t/24) + rng.normal(scale=0.3, size=T)
    return x

uni = gen_univariate()

try:
    res = run_service('univariate series with daily seasonality', uni, db=VECTOR_DB, task='forecast', horizon=6)
    print('run_service returned keys:', list(res.keys()))
    print('choice:', res.get('choice'))
    print('result keys:', list(res.get('result', {}).keys()))
except Exception as e:
    print('run_service error:', e)
    raise
