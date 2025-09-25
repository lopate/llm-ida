#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / 'tests'

HEAVY_KEYWORDS = [
    'import torch', 'torch.', 'from torch', 'torch_geometric', 'import tslearn', 'from tslearn',
    'import sktime', 'from sktime', 'tensorflow', 'jax', 'cuda', 'gpu', 'cupy'
]

def list_tests():
    return sorted([p for p in TEST_DIR.glob('test_*.py')])

def is_heavy(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    for kw in HEAVY_KEYWORDS:
        if kw in text:
            return True
    return False

def run_pytest(files):
    if not files:
        print('No files to run')
        return 0
    cmd = [sys.executable, '-m', 'pytest', '-q'] + [str(p) for p in files]
    print('\nRunning:', ' '.join(cmd))
    p = subprocess.run(cmd)
    return p.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', '-g', type=int, default=2, help='Number of heavy groups')
    parser.add_argument('--group-index', type=int, default=None, help='If set, run only this heavy group (0-based)')
    parser.add_argument('--run-heavy', action='store_true', help='Also run heavy groups after light tests (all groups)')
    parser.add_argument('--dry-run', action='store_true', help='Do not run tests, only print grouping')
    args = parser.parse_args()

    tests = list_tests()
    light = []
    heavy = []
    for t in tests:
        if is_heavy(t):
            heavy.append(t)
        else:
            light.append(t)

    print(f'Total tests: {len(tests)}; light: {len(light)}; heavy: {len(heavy)}')

    if args.dry_run:
        print('\nLight tests:')
        for p in light:
            print(' ', p.name)
        print('\nHeavy tests (will be split into', args.groups, 'groups):')
        for p in heavy:
            print(' ', p.name)
        # show heavy group commands
        groups = [[] for _ in range(args.groups)]
        for i, p in enumerate(heavy):
            groups[i % args.groups].append(p)
        print('\nHeavy group commands:')
        for i, g in enumerate(groups):
            if not g:
                continue
            files = ' '.join(str(x) for x in g)
            print(f'Group {i+1}: pytest -q {files}')
        return

    # run light tests first
    if light:
        rc = run_pytest(light)
        if rc != 0:
            print('Light tests failed with rc', rc)
            return rc
    else:
        print('No light tests to run')

    # prepare heavy groups
    groups = [[] for _ in range(args.groups)]
    for i, p in enumerate(heavy):
        groups[i % args.groups].append(p)

    # If group-index specified, run only that group
    if args.group_index is not None:
        gi = args.group_index
        if gi < 0 or gi >= len(groups):
            print('group-index out of range', gi)
            return 2
        g = groups[gi]
        print(f'Running heavy group {gi+1}/{len(groups)}: {len(g)} tests')
        rc = run_pytest(g)
        if rc != 0:
            print('Heavy group failed with rc', rc)
            return rc
        print('Heavy group finished')
        return 0

    for i, g in enumerate(groups):
        if not g:
            continue
        print(f'\nHeavy group {i+1}/{len(groups)}: {len(g)} tests')
        if args.run_heavy:
            rc = run_pytest(g)
            if rc != 0:
                print('Heavy group failed with rc', rc)
                return rc
        else:
            print('Run with --run-heavy to execute this group:')
            files = ' '.join(str(x) for x in g)
            print(f'  pytest -q {files}')

    print('\nAll requested runs finished')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
