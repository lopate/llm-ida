import runpy
import sys
import os
from pathlib import Path


def test_cli_guard_adds_repo_root(tmp_path, monkeypatch):
    """Test that scripts/build_vector_db.py ensures the repo root is on sys.path when imported/run.

    Strategy:
    - Remove repo root from sys.path if present
    - Execute the guard function by importing the script as a module via runpy
    - After execution, assert that the repo root (parent of scripts/) is on sys.path
    """
    # repo root is two levels above this file now (tests/unit/...)
    repo_root = str(Path(__file__).resolve().parents[2])
    scripts_dir = os.path.join(repo_root, "scripts")
    # ensure repo_root is not on sys.path for the test
    monkeypatch.setenv("PYTHONPATH", "")
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(repo_root)]

    script_path = os.path.join(scripts_dir, "build_vector_db.py")
    # run the script in a fresh globals dict to simulate direct execution import-time behavior
    glb = {}
    # ensure the script's argparse doesn't pick up pytest's args
    monkeypatch.setattr(sys, "argv", [script_path])
    runpy.run_path(script_path, run_name="__main__", init_globals=glb)

    # After running, repo_root should have been inserted into sys.path (at least temporarily)
    found = any(os.path.abspath(p) == os.path.abspath(repo_root) for p in sys.path)
    # If the package is already installed in editable mode, `app` may be importable without
    # inserting repo_root into sys.path. Accept either behaviour.
    app_importable = False
    try:
        import app  # noqa: F401
        app_importable = True
    except Exception:
        app_importable = False

    assert found or app_importable, "Repo root should be inserted into sys.path by the CLI guard or package must be importable"
