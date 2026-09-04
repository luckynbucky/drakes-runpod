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
  apt-get install -y -qq git wget unzip build-essential
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
pip install --quiet torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing DRAKES dependencies"
pip install --quiet packaging ninja
pip install --quiet transformers datasets omegaconf rich timm scipy wandb \
  lightning ipykernel notebook
# biopython is not a DRAKES dependency; the physics reward test suite validates
# against its independent implementation of the same thermodynamic model.
pip install --quiet biopython matplotlib pandas
pip install --quiet --upgrade hydra-core hydra-submitit-launcher

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
  pip install --quiet /tmp/gReLU
  rm -rf /tmp/gReLU
fi

python -m ipykernel install --user --name "$ENV_NAME" \
  --display-name "Python (${ENV_NAME})" >/dev/null

# --- 6. Data and pretrained weights --------------------------------------
# ~ tens of GB; the pod's network is fast, your laptop's is not, which is why
# this downloads on the pod rather than being uploaded.
if [ ! -d "$BASE_PATH/mdlm" ]; then
  echo "==> Downloading DRAKES_data.zip (this takes a while)"
  DATA_URL="https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&dl=1"
  wget --show-progress -O "$BASE_PATH/DRAKES_data.zip" "$DATA_URL"
  echo "==> Unzipping"
  unzip -q "$BASE_PATH/DRAKES_data.zip" -d "$BASE_PATH"
  # The zip may contain a single top-level folder; flatten it if so.
  if [ -d "$BASE_PATH/DRAKES_data" ] && [ ! -d "$BASE_PATH/mdlm" ]; then
    mv "$BASE_PATH/DRAKES_data"/* "$BASE_PATH/"
    rmdir "$BASE_PATH/DRAKES_data"
  fi
  rm -f "$BASE_PATH/DRAKES_data.zip"
else
  echo "==> Data already present at $BASE_PATH/mdlm, skipping download"
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
