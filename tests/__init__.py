"""Test package initializer: configure environment to avoid native thread shutdown issues.

This file runs very early when pytest imports test modules as a package. It does NOT mock
any external libraries — it only sets environment variables to limit the number of threads
used by OpenMP/MKL/OpenBLAS/NumExpr and tries to reduce torch's threads if torch is already
installed. These measures reduce chances of segmentation faults caused by native thread
finalizers running at process exit.
"""
import os
import warnings

# Limit parallelism in native numeric libraries which can cause exit-time races
env_limits = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}

for k, v in env_limits.items():
    os.environ.setdefault(k, v)

# If torch is already importable, try to limit its threads as well. This call is best-effort
# and will be skipped if torch isn't installed yet (we avoid importing heavy libs eagerly).
try:
    import importlib
    spec = importlib.util.find_spec('torch')
    if spec is not None:
        import torch
        try:
            # limit intra-op and inter-op parallelism
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            # ignore if functions are not available
            pass
except Exception:
    # do not raise during test discovery; best-effort only
    warnings.warn('Could not pre-configure torch thread limits (continuing)')
