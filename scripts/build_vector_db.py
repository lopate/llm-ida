#!/usr/bin/env python3
"""Minimal CLI for building and inspecting the vector DB of models.

This file is intentionally small and import-safe so tests can import it to
verify a sys.path guard without executing CLI logic.
"""
import sys
import os


def _ensure_repo_on_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_on_path()


def main():
    # Keep the CLI minimal; heavy operations live in app.model_selector
    print("build_vector_db CLI - use app.model_selector.VECTOR_DB for operations")


if __name__ == '__main__':
    main()
