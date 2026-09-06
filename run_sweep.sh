#!/usr/bin/env bash
# Launch the comparison runs. Run this inside tmux.
#
#   tmux new -A -s sweep
#   bash /workspace/drakes-runpod/run_sweep.sh
#
# A script rather than a pasted block, for three reasons: a paste that fails
# partway carries on in the wrong state, the exact settings are the experiment
# and belong in version control, and this way the runs are reproducible by
# someone who was not here when they were chosen.

set -euo pipefail

BASE_PATH="${BASE_PATH:-/workspace/drakes_data}"
KIT_DIR="${KIT_DIR:-/workspace/drakes-runpod}"
DRAKES_DIR="${DRAKES_DIR:-/workspace/DRAKES/drakes_dna}"
EPOCHS="${EPOCHS:-30}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> Syncing the kit into the DRAKES checkout"
git -C "$KIT_DIR" pull --ff-only
cp "$KIT_DIR"/{physics_reward.py,hairpin_reward.py,grelu_offline.py,finetune_multiobjective.py} \
   "$DRAKES_DIR"/

cd "$DRAKES_DIR"

common=(
  --skip_grelu_artifact
  --base_path "$BASE_PATH/"
  --num_epochs "$EPOCHS"
  --batch_size 4
  --num_accum_steps 32
  --eval_oracle_device cpu
  --save_every_n_epochs 10
)

# 1. Baseline. This is the run that cannot be skipped: without it, a GC decline
#    under the physics constraint cannot be attributed to the physics
#    constraint, because DRAKES fine-tuning alone might do the same thing.
echo
echo "======================================================================"
echo "RUN 1/3  baseline: activity only, no physics term"
echo "======================================================================"
python finetune_multiobjective.py "${common[@]}" \
  --name base --w_phys 0

# 2. The observed failure mode, reproduced deliberately so it is a measurement
#    rather than an anecdote: the hairpin constraint with composition free to
#    move, which previously took GC from 0.44 to 0.28.
echo
echo "======================================================================"
echo "RUN 2/3  hairpin constraint, GC unpinned"
echo "======================================================================"
python finetune_multiobjective.py "${common[@]}" \
  --name unpinned --w_phys 0.5

# 3. The same constraint with composition held in the synthesizable window and
#    a KL anchor that actually anchors. If the arrangement effect grows here
#    and not in run 2, the model learned sequence design rather than base
#    composition -- which is the claim the whole project rests on.
echo
echo "======================================================================"
echo "RUN 3/3  hairpin constraint, GC pinned to [0.35, 0.65], alpha 0.01"
echo "======================================================================"
python finetune_multiobjective.py "${common[@]}" \
  --name gcpinned --w_phys 0.5 --w_gc 5.0 --alpha 0.01

echo
echo "======================================================================"
echo "All runs complete. Compare them with:"
echo "  python $KIT_DIR/report_run.py $BASE_PATH/mdlm/reward_bp_results_final/*"
echo "======================================================================"
