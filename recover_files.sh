#!/usr/bin/env bash
# Recover individual files from the DRAKES archive without re-extracting it all.
#
#   bash recover_files.sh mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt
#
# A truncated extraction usually damages one or two files while leaving the
# rest intact, so re-running the whole setup wastes an hour rewriting data that
# is already fine. This re-fetches the archive, pulls out only the paths asked
# for, verifies them, and puts them in place.
#
# The archive is downloaded to whichever of the staging candidates has the most
# free space. On RunPod the container disk and the network volume are separate,
# so staging off the volume keeps the peak requirement to the archive alone
# rather than the archive plus the volume's existing contents.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/workspace/drakes_data}"
DATA_URL="https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&dl=1"

if [ "$#" -eq 0 ]; then
  echo "usage: bash recover_files.sh <path under BASE_PATH> [more paths...]"
  echo
  echo "example:"
  echo "  bash recover_files.sh mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt"
  exit 1
fi

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# --- pick the roomiest staging location -------------------------------------
STAGE=""
BEST=0
for candidate in "${STAGE_DIR:-}" /tmp "$BASE_PATH"; do
  [ -n "$candidate" ] || continue
  [ -d "$candidate" ] || continue
  avail=$(free_gb "$candidate")
  [ -n "$avail" ] || continue
  if [ "$avail" -gt "$BEST" ]; then
    BEST=$avail
    STAGE=$candidate
  fi
done
[ -n "$STAGE" ] || { echo "ERROR: no usable staging directory"; exit 1; }

WORK="$STAGE/drakes_recover"
mkdir -p "$WORK"
ZIP="$WORK/DRAKES_data.zip"

echo "==> Recovering ${#} file(s) into $BASE_PATH"
echo "==> Staging in $WORK (${BEST} GB free)"
echo "==> Free space at $BASE_PATH: $(free_gb "$BASE_PATH") GB"

# --- fetch, resuming and re-fetching a truncated archive --------------------
if [ -f "$ZIP" ] && unzip -l "$ZIP" >/dev/null 2>&1; then
  echo "==> Archive already staged and readable"
else
  [ -f "$ZIP" ] && echo "==> Staged archive is truncated; resuming"
  wget --continue --show-progress -O "$ZIP" "$DATA_URL"
fi

if ! unzip -l "$ZIP" >/dev/null 2>&1; then
  echo "ERROR: downloaded archive is still unreadable. Delete $ZIP and retry."
  exit 1
fi

# --- extract each requested path --------------------------------------------
EXTRACT="$WORK/extracted"
rm -rf "$EXTRACT"; mkdir -p "$EXTRACT"

failed=0
for rel in "$@"; do
  echo
  echo "--- $rel"
  # The archive wraps its contents in a top-level directory, so match on the
  # tail of the path rather than assuming the wrapper's name.
  inner=$(unzip -l "$ZIP" | awk '{print $4}' | grep -E "(^|/)${rel}$" | head -1 || true)
  if [ -z "$inner" ]; then
    echo "    ERROR: not found in the archive"
    failed=1
    continue
  fi
  echo "    found in archive as: $inner"

  # Not -q. Extracting a multi-gigabyte member takes minutes, and silence
  # here is indistinguishable from a hang -- which has now caused confusion
  # three separate times in this project.
  echo "    extracting (this takes a few minutes for a large checkpoint)..."
  unzip -o "$ZIP" "$inner" -d "$EXTRACT"
  src="$EXTRACT/$inner"

  # Verify before overwriting: replacing a broken file with another broken
  # file is worse than leaving it, because it looks like progress.
  case "$rel" in
    *.ckpt|*.pt)
      if ! python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1])" "$src" 2>/dev/null; then
        echo "    ERROR: extracted copy is also unreadable; not installing it"
        failed=1
        continue
      fi
      echo "    verified: readable archive"
      ;;
  esac

  mkdir -p "$(dirname "$BASE_PATH/$rel")"
  mv "$src" "$BASE_PATH/$rel"
  echo "    installed: $BASE_PATH/$rel ($(du -h "$BASE_PATH/$rel" | cut -f1))"
done

echo
if [ "$failed" -eq 0 ]; then
  echo "==> All requested files recovered. Removing staged archive."
  rm -rf "$WORK"
  echo "==> Re-run smoke_test.py to confirm."
else
  echo "==> Some files failed. Staged archive kept at $ZIP for another attempt."
  exit 1
fi
