#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / 'tests'

def list_tests():
    return sorted([str(p) for p in TEST_DIR.glob('test_*.py')])

def run_pytest(files):
    cmd = [sys.executable, '-m', 'pytest', '-q'] + files
    print('RUN', ' '.join(cmd))
    p = subprocess.run(cmd)
    return p.returncode

def find_min_crash(files):
    # If full set doesn't crash, return None
    rc = run_pytest(files)
    if rc != 139:
        print('Full set did not crash (rc=%s)' % rc)
        return None

    # Try to find minimal subset via recursive bisection
    def recurse(subset):
        if not subset:
            return None
        if len(subset) == 1:
            rc = run_pytest(subset)
            return subset if rc == 139 else None
        mid = len(subset)//2
        left = subset[:mid]
        right = subset[mid:]
        rc = run_pytest(left)
        if rc == 139:
            return recurse(left)
        rc = run_pytest(right)
        if rc == 139:
            return recurse(right)
        # If neither half crashes alone, try mixed combinations
        # Try sliding window of small sizes
        for size in range(2, min(6, len(subset))+1):
            for i in range(len(subset)-size+1):
                chunk = subset[i:i+size]
                rc = run_pytest(chunk)
                if rc == 139:
                    return recurse(chunk)
        return None

    return recurse(files)

def main():
    tests = list_tests()
    print('Found', len(tests), 'test files')
    res = find_min_crash(tests)
    if res:
        print('\n=== Minimal crashing subset:')
        for r in res:
            print(r)
    else:
        print('No crashing subset found (surprising)')

if __name__ == '__main__':
    main()
