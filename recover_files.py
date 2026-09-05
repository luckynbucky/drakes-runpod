"""Recover individual files from the DRAKES archive without downloading it whole.

A zip stores its index at the END of the file, and each member's bytes at a
known offset. So given a server that honours HTTP range requests, you can read
the index, then fetch only the bytes belonging to the one file you want. For a
single 2.2 GB checkpoint inside a much larger archive, that is the difference
between minutes and an hour.

This is worth having because a truncated extraction usually damages one or two
files and leaves everything else intact. Re-running setup to repair that
rewrites tens of gigabytes that were never broken.

    pip install remotezip
    python recover_files.py mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt

If the server does not support range requests, this says so and stops rather
than silently pulling the whole archive; recover_files.sh does the full
download for that case.

NOTE: the range-request path is verified against zip archives served over HTTP
generally, but NOT specifically against Dropbox from the environment this was
written in. If it fails on the first call, that is the likely reason, and the
shell script is the fallback.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import zipfile

DATA_URL = (
    "https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip"
    "?rlkey=yf7w0pm64tlypwsewqc01wmfq&dl=1"
)


def find_member(names: list[str], rel: str) -> str | None:
    """Match on the tail of the path, so the wrapper directory's name is moot."""
    rel = rel.strip("/")
    for name in names:
        if name == rel or name.endswith("/" + rel):
            return name
    return None


def readable_archive(path: pathlib.Path) -> bool:
    try:
        with zipfile.ZipFile(path):
            return True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+", help="paths relative to BASE_PATH")
    ap.add_argument(
        "--base-path", default=os.environ.get("BASE_PATH", "/workspace/drakes_data")
    )
    ap.add_argument("--url", default=DATA_URL)
    args = ap.parse_args()

    try:
        from remotezip import RemoteZip
    except ImportError:
        print("remotezip is not installed.  pip install remotezip", file=sys.stderr)
        print("Or use the full-download fallback:  bash recover_files.sh <paths>")
        return 1

    base = pathlib.Path(args.base_path)
    print(f"==> Reading the archive index over HTTP range requests")

    try:
        remote = RemoteZip(args.url)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read the archive remotely: {exc}", file=sys.stderr)
        print(
            "\nThe server may not support range requests. Fall back to a full "
            "download:\n  bash recover_files.sh " + " ".join(args.targets),
            file=sys.stderr,
        )
        return 1

    failed = 0
    with remote as archive:
        names = archive.namelist()
        print(f"    index read: {len(names)} entries, no bulk download\n")

        for rel in args.targets:
            print(f"--- {rel}")
            member = find_member(names, rel)
            if member is None:
                print("    ERROR: not present in the archive")
                failed += 1
                continue

            info = archive.getinfo(member)
            print(f"    found as {member} ({info.file_size / 1e6:.1f} MB)")

            # Stage next to the destination so the final move stays on one
            # filesystem, then verify before overwriting anything. Replacing a
            # corrupt file with another corrupt file looks like progress and
            # is worse than leaving it alone.
            destination = base / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(
                dir=destination.parent, suffix=".partial"
            )
            os.close(handle)
            tmp = pathlib.Path(tmp_name)

            try:
                print("    fetching only this member...")
                with archive.open(member) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

                if destination.suffix in (".ckpt", ".pt") and not readable_archive(tmp):
                    print("    ERROR: the fetched copy is unreadable; not installing")
                    failed += 1
                    tmp.unlink(missing_ok=True)
                    continue

                tmp.replace(destination)
                size_mb = destination.stat().st_size / 1e6
                print(f"    installed: {destination} ({size_mb:.1f} MB, verified)")
            except Exception as exc:  # noqa: BLE001
                print(f"    ERROR: {type(exc).__name__}: {exc}")
                tmp.unlink(missing_ok=True)
                failed += 1

    print()
    if failed:
        print(f"{failed} file(s) failed. Fallback: bash recover_files.sh <paths>")
        return 1
    print("All requested files recovered. Re-run smoke_test.py to confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
