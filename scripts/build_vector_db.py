#!/usr/bin/env python3
"""CLI for building and inspecting the vector DB of models.

Usage:
  python scripts/build_vector_db.py --rebuild        # initialize vector DB and upsert built-in models
  python scripts/build_vector_db.py --list          # list known models (catalog)
  python scripts/build_vector_db.py --add 'id|name|library|desc'  # add a model and upsert its embedding
  python scripts/build_vector_db.py --show          # query DB with each model desc and show top hit
"""
import argparse
import json
import sys
import os


# If this script is run directly from the repo root, users often forget to set PYTHONPATH=.
# Try to ensure the repository root is on sys.path so `from app import ...` works without external env setup.
def _ensure_repo_on_path():
    try:
        return
    except Exception:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)


# Ensure importability
_ensure_repo_on_path()


def main():
    parser = argparse.ArgumentParser(description="Build / inspect vector DB for model selector")
    parser.add_argument("--rebuild", action="store_true", help="Initialize and upsert built-in models")
    parser.add_argument("--list", action="store_true", help="List built-in model catalogue")
    parser.add_argument("--add", type=str, help="Add model entry as 'id|name|library|desc'")
    parser.add_argument("--show", action="store_true", help="Query DB using model descriptions and show top hit")
    parser.add_argument("--save", type=str, help="Save FAISS index to given base path (without extension)")
    parser.add_argument("--load", type=str, help="Load FAISS index from given base path (without extension)")
    args = parser.parse_args()

    # lazy imports from app
    try:
        from app import vector_db
        from app import model_selector
    except Exception as e:
        print(f"Failed to import app modules: {e}")
        sys.exit(2)

    if args.rebuild:
        print("Initializing vector DB...")
        vector_db.initialize()
        print("Upserting built-in models...")
        # model_selector.init_vector_db will compute embeddings and upsert
        try:
            model_selector.init_vector_db()
        except Exception as e:
            print(f"Error during init_vector_db: {e}")
            sys.exit(1)
        print("Rebuild complete.")

    if args.save:
        path = args.save
        try:
            vector_db.save(path)
            print(f"Saved FAISS index to {path}")
        except Exception as e:
            print(f"Failed to save FAISS index: {e}")

    if args.load:
        path = args.load
        try:
            vector_db.load(path)
            print(f"Loaded FAISS index from {path}")
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")

    if args.list:
        print("Known catalogue entries:")
        for m in model_selector.MODEL_DB:
            print(json.dumps(m, ensure_ascii=False))

    if args.add:
        parts = args.add.split("|", 3)
        if len(parts) != 4:
            print("--add expects format id|name|library|desc")
            sys.exit(2)
        mid, name, lib, desc = parts
        entry = {"id": mid, "name": name, "library": lib, "desc": desc}
        model_selector.MODEL_DB.append(entry)
        # embed and upsert just this entry
        try:
            embs = model_selector._embed_texts([desc])
        except Exception:
            embs = [[0.0] * 64]
        vector_db.upsert([mid], [embs[0]], [{"name": name, "library": lib, "desc": desc}])
        print(f"Added and upserted {mid}")

    if args.show:
        print("Querying DB for each model description and showing top hit:")
        for m in model_selector.MODEL_DB:
            desc = m["desc"]
            try:
                emb = model_selector._embed_texts([desc])[0]
                res = vector_db.query(emb, k=1)
                print(f"Model {m['id']}: query -> {res}")
            except Exception as e:
                print(f"Failed to query for {m['id']}: {e}")

    if not any([args.rebuild, args.list, args.add, args.show]):
        parser.print_help()


if __name__ == '__main__':
    main()
