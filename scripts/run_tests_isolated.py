"""Run each test file in a separate subprocess to avoid process-wide native finalizer races.

Usage:
    python scripts/run_tests_isolated.py    # runs all tests under `tests/`

This does not mock or alter test behavior; it simply runs pytest per-file so native
extensions (torch/faiss/etc.) are initialized and torn down in separate processes,
preventing cross-test segfaults caused by exit-time races.
"""
import glob
import subprocess
import sys

def main():
    files = sorted(glob.glob('tests/test_*.py'))
    if not files:
        print('No test files found')
        return 1
    failures = 0
    for f in files:
        print('\n=== Running', f, '===')
        cmd = [sys.executable, '-m', 'pytest', f, '-q']
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f'File {f} failed with exit code {rc}')
            failures += 1
    if failures:
        print(f'Finished: {failures} file(s) failed')
        return 2
    print('All test files passed (isolated)')
    return 0

if __name__ == '__main__':
    sys.exit(main())
