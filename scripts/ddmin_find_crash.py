#!/usr/bin/env python3
"""DDMin-like bisection to find minimal subset of test files that crash pytest.
This runs pytest in-process (single python process) for subsets and looks for return code 139.
It caches results to /tmp/ddmin_cache.json to avoid re-running the same subsets.
"""
import json
import subprocess
import sys
from pathlib import Path
from itertools import combinations
import os

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / 'tests'
CACHE_FILE = Path('/tmp/ddmin_cache.json')

def list_tests():
    return sorted([str(p) for p in TEST_DIR.glob('test_*.py')])

def run_subset(files):
    env = os.environ.copy()
    env.update({
        'OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1',
        'CUDA_VISIBLE_DEVICES':'',
    })
    cmd = [sys.executable, '-m', 'pytest', '-q'] + files
    print('RUN', len(files), 'files')
    p = subprocess.run(cmd, env=env)
    return p.returncode

def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        CACHE_FILE.write_text(json.dumps(cache))
    except Exception:
        pass

def ddmin(tests):
    cache = load_cache()

    def check(sub):
        key = ','.join(sub)
        if key in cache:
            return cache[key]
        rc = run_subset(list(sub))
        cache[key] = rc
        save_cache(cache)
        return rc

    def is_crash(rc):
        # pytest subprocess returncode may be negative for signals (e.g. -11)
        if rc == 139:
            return True
        if rc < 0 and abs(rc) == 11:
            return True
        return False

    # If full set doesn't crash, give up
    full_rc = check(tests)
    if not is_crash(full_rc):
        print('Full set did not crash (rc=%s)' % full_rc)
        return None

    n = len(tests)
    subset = list(tests)
    changed = True
    while changed and len(subset) > 1:
        changed = False
        # try removing single items
        for i in range(len(subset)):
            cand = subset[:i] + subset[i+1:]
            rc = check(cand)
            if rc == 139:
                subset = cand
                changed = True
                print('Reduced to', len(subset))
                break
        # try halves
        if not changed:
            mid = len(subset)//2
            left = subset[:mid]
            right = subset[mid:]
            if is_crash(check(left)):
                subset = left
                changed = True
                continue
            if is_crash(check(right)):
                subset = right
                changed = True
                continue
            # try small combinations
            for size in range(2, min(6, len(subset))+1):
                for comb in combinations(subset, size):
                    rc = check(list(comb))
                    if is_crash(rc):
                        subset = list(comb)
                        changed = True
                        break
                if changed:
                    break

    return subset

def main():
    tests = list_tests()
    print('Total tests:', len(tests))
    res = ddmin(tests)
    if res:
        print('Minimal crashing subset:')
        for r in res:
            print(r)
    else:
        print('No crash subset found')

if __name__ == '__main__':
    main()
