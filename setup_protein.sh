#!/usr/bin/env bash
# Bootstrap the DRAKES protein-stability experiment on a RunPod pod.
#
#   bash setup_protein.sh
#
# This is the heavier of the two experiments. It builds the MultiFlow
# environment (python 3.10, torch 2.0.1 + cu117) and installs PyRosetta.
#
# GPU CHOICE MATTERS HERE. torch 2.0.1+cu117 ships kernels up to sm_86, so it
# runs on A100 / A6000 / A5000 / 3090 but NOT on H100, H200, L40S or 40-series
# and newer. Pick an Ampere card for this one. (The DNA experiment is on torch
# 2.3.1+cu121 and has no such restriction.)

set -euo pipefail

BASE_PATH="${BASE_PATH:-/workspace/drakes_data}"
REPO_DIR="${REPO_DIR:-/workspace/DRAKES}"
ENV_NAME="${ENV_NAME:-multiflow}"

echo "==> BASE_PATH = $BASE_PATH"
echo "==> REPO_DIR  = $REPO_DIR"

mkdir -p "$BASE_PATH"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  # tmux matters here: this script runs long enough that a dropped SSH or
  # browser session would otherwise SIGHUP it partway through.
  apt-get install -y -qq git wget unzip build-essential tmux
fi

# --- Conda ---------------------------------------------------------------
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

if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/ChenyuWang-Monica/DRAKES.git "$REPO_DIR"
fi

# --- MultiFlow environment ----------------------------------------------
# multiflow.yml pins exact conda build hashes, so solving it is slow. mamba
# does the same job in a fraction of the time; fall back to conda if absent.
if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "==> Creating conda env '${ENV_NAME}' from multiflow.yml (slow, ~15 min)"
  SOLVER=conda
  if command -v mamba >/dev/null 2>&1; then
    SOLVER=mamba
  else
    conda install -y -n base -c conda-forge --override-channels mamba \
      >/dev/null 2>&1 && SOLVER=mamba || true
  fi
  "$SOLVER" env create -f "$REPO_DIR/drakes_protein/multiflow.yml"
fi
conda activate "$ENV_NAME"

# --- Local package + torch-scatter --------------------------------------
echo "==> Installing the local drakes_protein package"
pip install --quiet -e "$REPO_DIR/drakes_protein"

echo "==> Installing torch-scatter for torch 2.0.1+cu117"
pip install --quiet torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# --- PyRosetta -----------------------------------------------------------
# Used by the evaluation scripts for structure scoring. Free for academic and
# non-commercial use; the installer accepts that license on your behalf, so
# read https://www.pyrosetta.org/home/licensing-pyrosetta before running this
# for anything commercial.
if ! python -c "import pyrosetta" 2>/dev/null; then
  echo "==> Installing PyRosetta (large download, ~1.5 GB)"
  pip install --quiet pyrosetta-installer
  python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
fi

# --- Data ----------------------------------------------------------------
if [ ! -d "$BASE_PATH/proteindpo_data" ]; then
  echo "==> Downloading DRAKES_data.zip"
  DATA_URL="https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&dl=1"
  wget --show-progress -O "$BASE_PATH/DRAKES_data.zip" "$DATA_URL"
  unzip -q "$BASE_PATH/DRAKES_data.zip" -d "$BASE_PATH"
  if [ -d "$BASE_PATH/DRAKES_data" ] && [ ! -d "$BASE_PATH/proteindpo_data" ]; then
    mv "$BASE_PATH/DRAKES_data"/* "$BASE_PATH/"
    rmdir "$BASE_PATH/DRAKES_data"
  fi
  rm -f "$BASE_PATH/DRAKES_data.zip"
else
  echo "==> Data already present, skipping download"
fi

echo "==> Rewriting hardcoded base_path"
python "$(dirname "$0")/set_base_path.py" --repo "$REPO_DIR" --base-path "$BASE_PATH"

cat <<EOF

========================================================================
Setup complete.

  conda activate ${ENV_NAME}
  cd ${REPO_DIR}/drakes_protein/fmif

Verify:
  python $(dirname "$0")/smoke_test.py --base-path ${BASE_PATH} --experiment protein

Fine-tune:
  python finetune_reward_bp.py --wandb_name=run1

Evaluate:
  cd scripts && bash ours.sh
========================================================================
EOF
