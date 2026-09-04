#!/usr/bin/env python3
"""Point a DRAKES checkout at your own data directory.

DRAKES hardcodes the authors' cluster path, ``/data/scratch/wangchy/seqft/``,
in ten files across the DNA and protein experiments -- as a module-level
``base_path``, as an argparse default, and inside the hydra output config. The
upstream READMEs tell you to change them by hand; this does it in one pass.

Safe to run more than once: it only rewrites the literal author path, so a
second run finds nothing to do.

    python set_base_path.py --repo /workspace/DRAKES --base-path /workspace/drakes_data
"""

import argparse
import json
import pathlib
import sys

AUTHOR_PATH = "/data/scratch/wangchy/seqft/"
SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".ipynb"}


def rewrite(path: pathlib.Path, new_path: str) -> int:
    """Replace the author path in one file. Returns the number of hits."""
    text = path.read_text(encoding="utf-8")
    hits = text.count(AUTHOR_PATH)
    if not hits:
        return 0

    if path.suffix == ".ipynb":
        # Notebooks are JSON, and the path lives inside source-line strings.
        # Rewriting the raw text works because the path contains no characters
        # that JSON escapes, but re-serialising keeps the file provably valid.
        nb = json.loads(text)
        replaced = json.loads(json.dumps(nb).replace(AUTHOR_PATH, new_path))
        path.write_text(json.dumps(replaced, indent=1) + "\n", encoding="utf-8")
    else:
        path.write_text(text.replace(AUTHOR_PATH, new_path), encoding="utf-8")

    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="path to the DRAKES checkout")
    ap.add_argument(
        "--base-path",
        required=True,
        help="directory holding mdlm/, proteindpo_data/, pmpnn/ etc.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report matches without writing"
    )
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        print(f"error: no such directory: {repo}", file=sys.stderr)
        return 1

    # A trailing slash matters: the code does os.path.join(base_path, 'mdlm/...')
    # in some places and f-string concatenation in others.
    new_path = args.base_path.rstrip("/") + "/"

    total_files = 0
    total_hits = 0
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if ".git" in path.parts:
            continue

        if args.dry_run:
            hits = path.read_text(encoding="utf-8").count(AUTHOR_PATH)
        else:
            hits = rewrite(path, new_path)

        if hits:
            total_files += 1
            total_hits += hits
            print(f"  {hits:>2} x  {path.relative_to(repo)}")

    verb = "would rewrite" if args.dry_run else "rewrote"
    if total_hits:
        print(f"{verb} {total_hits} occurrence(s) in {total_files} file(s) -> {new_path}")
    else:
        print(f"no occurrences of {AUTHOR_PATH} found (already patched?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
