#!/usr/bin/env bash
# Bootstrap the DRAKES regulatory-DNA experiment on a RunPod pod.
#
# Run this from a pod terminal (web console or SSH):
#   bash setup_dna.sh
#
# Everything lands under $BASE_PATH, which defaults to the persistent
# /workspace volume so a pod stop does not wipe the checkpoints.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/workspace/drakes_data}"
REPO_DIR="${REPO_DIR:-/workspace/DRAKES}"
ENV_NAME="${ENV_NAME:-sedd}"

echo "==> BASE_PATH = $BASE_PATH"
echo "==> REPO_DIR  = $REPO_DIR"

if [ ! -d /workspace ]; then
  echo "WARNING: /workspace does not exist. You are probably on a pod with no"
  echo "network volume attached, so everything here dies when the pod stops."
  echo "Set BASE_PATH/REPO_DIR to somewhere you have chosen deliberately, or"
  echo "recreate the pod with a volume. Continuing in 10s..."
  sleep 10
fi

mkdir -p "$BASE_PATH"

# --- 1. System packages --------------------------------------------------
# RunPod PyTorch images are Ubuntu with root; unzip and git are not always present.
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  # tmux matters here: this script runs long enough that a dropped SSH or
  # browser session would otherwise SIGHUP it partway through.
  apt-get install -y -qq git wget unzip build-essential tmux
fi

# --- 2. Conda ------------------------------------------------------------
# The DRAKES DNA env pins python 3.9.18, which is older than most RunPod
# images ship, so we use conda rather than the image's system python.
if ! command -v conda >/dev/null 2>&1; then
  if [ -d /workspace/miniconda3 ]; then
    # An earlier run installed it, but this shell has not picked up the
    # conda init written into ~/.bashrc. Reuse it rather than reinstalling,
    # which would fail on the existing directory.
    echo "==> Reusing Miniconda already at /workspace/miniconda3"
    export PATH="/workspace/miniconda3/bin:$PATH"
  else
    echo "==> Installing Miniconda"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /workspace/miniconda3
    rm /tmp/miniconda.sh
    export PATH="/workspace/miniconda3/bin:$PATH"
    conda init bash
  fi
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "==> Creating conda env '${ENV_NAME}' (python 3.9.18)"
  # conda-forge only, with --override-channels. Recent Miniconda refuses to use
  # Anaconda's "defaults" channels (repo.anaconda.com/pkgs/*) until their Terms
  # of Service are explicitly accepted, which otherwise stops this script dead
  # with CondaToSNonInteractiveError. conda-forge carries the pinned python
  # build and no such gate.
  #
  # If you would rather use the defaults channels, that is your call to make,
  # not this script's -- accept their terms yourself first:
  #   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  #   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  conda create -y -n "$ENV_NAME" python=3.9.18 -c conda-forge --override-channels
fi
conda activate "$ENV_NAME"

# --- 3. Clone DRAKES -----------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
  echo "==> Cloning DRAKES"
  git clone https://github.com/ChenyuWang-Monica/DRAKES.git "$REPO_DIR"
fi

# --- 4. Python dependencies ---------------------------------------------
# This mirrors drakes_dna/env.sh. We use the pip wheels for torch rather than
# the conda channel: same version, much faster to resolve.
echo "==> Installing PyTorch 2.3.1 + cu121"
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing DRAKES dependencies"
# These versions are pinned deliberately. Python 3.9 reached end of life in
# October 2025 and most of these projects have dropped support for it since,
# so an unpinned install sends pip walking backwards through release history
# hunting for a 3.9-compatible combination -- on nine of these packages at
# once. That is a combinatorial search which can run for hours and prints
# nothing while it does it. Each pin below is the newest release that still
# supports 3.9, which turns the search into a direct install.
#
# The pins help correctness too, not just speed: they land near the versions
# DRAKES was developed against, and gReLU 1.0.2 is a 2024 release whose
# checkpoints expect that era's ecosystem.
#
# If you move this off Python 3.9, drop the pins rather than bumping them.
pip install --prefer-binary packaging ninja
# numpy<2 is not incidental. NumPy 2.0 removed long-deprecated aliases, and
# gReLU 1.0.2 still calls np.product, which raises AttributeError at predict
# time -- after the oracle has loaded, so it looks like a model problem rather
# than a dependency one. torch 2.3.1 also predates NumPy 2, so holding the
# whole stack on 1.x is the consistent choice, not just a workaround.
pip install --prefer-binary "numpy<2"
pip install --prefer-binary \
  "transformers==4.57.6" \
  "datasets==4.5.0" \
  "lightning==2.6.0" \
  "scipy==1.13.1" \
  "pandas==2.3.3" \
  "matplotlib==3.9.4" \
  "notebook==7.5.7" \
  "wandb==0.26.1" \
  "ipykernel==6.31.0" \
  timm rich omegaconf
# biopython is not a DRAKES dependency; the physics reward test suite validates
# against its independent implementation of the same thermodynamic model.
# remotezip lets recover_files.py pull a single member out of the data
# archive over HTTP range requests, instead of re-downloading all of it.
pip install --prefer-binary "biopython==1.85" remotezip
pip install --prefer-binary --upgrade hydra-core hydra-submitit-launcher

# causal-conv1d is listed in env.sh because upstream MDLM can use a Mamba
# backbone. DRAKES's DNA config uses `backbone: cnn`, so this is not needed and
# it compiles CUDA kernels from source (slow, and fails without nvcc). Try it,
# but do not let a failure stop the setup.
if ! pip install --quiet causal-conv1d 2>/dev/null; then
  echo "    (causal-conv1d skipped -- not required for the CNN backbone)"
fi

# --- 5. gReLU (the reward oracle framework) ------------------------------
# Must be 1.0.2: later versions changed the LightningModel checkpoint format
# and the provided oracle .ckpt files will not load.
if ! python -c "import grelu" 2>/dev/null; then
  echo "==> Installing gReLU v1.0.2"
  git clone https://github.com/Genentech/gReLU.git /tmp/gReLU
  git -C /tmp/gReLU checkout -q v1.0.2
  pip install --prefer-binary /tmp/gReLU
  rm -rf /tmp/gReLU
fi

python -m ipykernel install --user --name "$ENV_NAME" \
  --display-name "Python (${ENV_NAME})" >/dev/null

# --- 6. Data and pretrained weights --------------------------------------
# ~ tens of GB; the pod's network is fast, your laptop's is not, which is why
# this downloads on the pod rather than being uploaded.
# Free space in whole GB at a given path.
free_gb() {
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

ZIP_PATH="$BASE_PATH/DRAKES_data.zip"

# The archive wraps everything in one top-level directory, and its name is not
# stable across releases -- the current bundle uses data_and_model. The DRAKES
# code expects mdlm/, proteindpo_data/ and the rest directly under base_path,
# so flatten whatever the wrapper happens to be called rather than matching a
# hardcoded name. This sits outside the download guard above so that it also
# repairs a tree left nested by an earlier version of this script.
if [ ! -d "$BASE_PATH/mdlm" ]; then
  for wrapper in "$BASE_PATH"/*/; do
    [ -d "$wrapper" ] || continue
    if [ -d "${wrapper}mdlm" ]; then
      echo "==> Flattening $(basename "$wrapper")/ into $BASE_PATH"
      # A move within one filesystem only rewrites directory entries, so this
      # is instant regardless of how many gigabytes are involved.
      mv "$wrapper"* "$BASE_PATH"/
      mv "$wrapper".[!.]* "$BASE_PATH"/ 2>/dev/null || true
      rmdir "$wrapper"
      break
    fi
  done
fi


if [ ! -d "$BASE_PATH/mdlm" ]; then
  AVAIL=$(free_gb "$BASE_PATH")
  echo "==> Free space at $BASE_PATH: ${AVAIL:-unknown} GB"
  if [ -n "${AVAIL:-}" ] && [ "$AVAIL" -lt 25 ]; then
    echo "    WARNING: under 25 GB free. The archive has to be downloaded and"
    echo "    then extracted alongside itself, so the peak requirement is the"
    echo "    compressed size plus the extracted size. This may not fit."
  fi

  DATA_URL="https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&dl=1"
  # Presence is not completeness. A half-downloaded archive is still a file,
  # and treating it as done means every later step fails on a truncated zip.
  # Reading the central directory back is a cheap completeness check.
  if [ -f "$ZIP_PATH" ] && unzip -l "$ZIP_PATH" >/dev/null 2>&1; then
    echo "==> Archive already downloaded and readable, skipping"
  else
    if [ -f "$ZIP_PATH" ]; then
      echo "==> Existing archive is truncated; resuming download"
    else
      echo "==> Downloading DRAKES_data.zip (this takes a while)"
    fi
    wget --continue --show-progress -O "$ZIP_PATH" "$DATA_URL"
  fi

  # Measure before committing to the extract, so running out of room produces
  # a useful message rather than a half-written tree.
  UNCOMPRESSED_BYTES=$(unzip -l "$ZIP_PATH" | tail -1 | awk '{print $1}')
  UNCOMPRESSED_GB=$(( (UNCOMPRESSED_BYTES + 1073741823) / 1073741824 ))
  ZIP_GB=$(du -BG "$ZIP_PATH" | cut -f1 | tr -dc '0-9')
  AVAIL=$(free_gb "$BASE_PATH")
  echo "==> Archive is ${ZIP_GB} GB compressed, ~${UNCOMPRESSED_GB} GB extracted."
  echo "    Free space: ${AVAIL:-unknown} GB"

  if [ -n "${AVAIL:-}" ] && [ "$AVAIL" -lt "$UNCOMPRESSED_GB" ]; then
    cat <<MSG

ERROR: not enough free space to extract the archive.
  needed:    ~${UNCOMPRESSED_GB} GB
  available: ${AVAIL} GB

The bundle carries data for BOTH experiments, but the DNA experiment only
needs the mdlm/ tree. Extract just that, which is much smaller:

  unzip '$ZIP_PATH' 'mdlm/*' -d '$BASE_PATH'
  rm '$ZIP_PATH'
  bash $0

Or list what is in the archive first, to see where the space is going:

  unzip -l '$ZIP_PATH' | sort -k1 -n -r | head -30

Otherwise, resize the network volume in the RunPod console and re-run.
MSG
    exit 1
  fi

  echo "==> Unzipping (quiet; a large archive can sit here several minutes)"
  # -o overwrites without asking. Without it, re-extracting over an earlier
  # attempt stops on an interactive "replace ...? [y]es, [n]o, [A]ll" prompt,
  # which in a script left running unattended means it waits forever for a
  # keypress nobody is there to give.
  unzip -qo "$ZIP_PATH" -d "$BASE_PATH"
  rm -f "$ZIP_PATH"
  echo "==> Free space after extract: $(free_gb "$BASE_PATH") GB"
else
  echo "==> Data already present at $BASE_PATH/mdlm, skipping download"
fi

if [ ! -d "$BASE_PATH/mdlm" ]; then
  echo "ERROR: $BASE_PATH/mdlm not found after extraction. What is there:"
  ls -la "$BASE_PATH"
  echo "The DRAKES code reads base_path/mdlm/..., so that tree has to exist."
  exit 1
fi

# --- 7. Point the code at BASE_PATH --------------------------------------
# DRAKES hardcodes the authors' own scratch path in several files.
echo "==> Rewriting hardcoded base_path"
python "$(dirname "$0")/set_base_path.py" --repo "$REPO_DIR" --base-path "$BASE_PATH"

cat <<EOF

========================================================================
Setup complete.

  conda activate ${ENV_NAME}
  cd ${REPO_DIR}/drakes_dna

Verify the install:
  python $(dirname "$0")/smoke_test.py --base-path ${BASE_PATH}

Then fine-tune:
  python finetune_reward_bp.py --name run1 --base_path ${BASE_PATH}
========================================================================
EOF
