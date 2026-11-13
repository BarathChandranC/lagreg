#!/bin/bash
#SBATCH --job-name=ladder_posthoc_nsweep_seeds
#SBATCH --array=0-13
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

REPO=${REPO:-$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)}
BATCH=512
TARGET_STEPS=100000
EVAL_BLOCK_I=10          # held-out block for R^pop -- see bounds check below

# All ladder output goes here instead of inside each model's own directory,
# named by seed/N so results from every (N, seed) pair land in one place
# without colliding.
OUT_ROOT="$REPO/results_ladder_posthoc"
mkdir -p "$OUT_ROOT"

cd "$REPO/scripts"

# Same grid as the training/eval sweep -- one model dir per (N, seed) pair.
LENGTHS=(512 1024 2048 4096 8192 16384 65536)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${LENGTHS[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Must match how N_ITER was computed at training time -- analyze_checkpoints.py
# doesn't need N_ITER itself (it discovers whatever checkpoints exist on disk),
# but MODEL_DIR was built from it, so it has to be reproduced exactly to
# locate the right directory.
STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))

echo "=== task $SLURM_ARRAY_TASK_ID : N=$N seed=$SEED n_iter=$N_ITER eval_block_i=$EVAL_BLOCK_I ==="

MODEL_DIR="$REPO/results_transformer_pick_0_for_training/models_restricted_continuous_${SEED}/test_model_script_${N_ITER}_${N}_cross_entropy_loss/8_512"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model dir not found: $MODEL_DIR" >&2
    echo "Check that the training sweep for N=$N seed=$SEED actually completed "
    echo "and that N_ITER=$N_ITER matches what train_transformer.py used." >&2
    exit 1
fi

# EVAL_BLOCK_I=10 needs block 10 to be fully in-bounds: right edge is
# (EVAL_BLOCK_I+1)*N, so the underlying dataset needs at least 11*N sequences
# for this N. Check up front rather than let the smaller-N tasks burn a
# training-block pass before failing on the held-out one.
python -c "
import sys, torch
sys.path.insert(0, '$REPO')
from utils import load_data
_, sequences, *_ = load_data('$REPO/data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy', torch.device('cpu'))
need = ($EVAL_BLOCK_I + 1) * $N
if len(sequences) < need:
    print(f'ERROR: dataset has {len(sequences)} sequences, need >= {need} '
          f'for eval-block-i=$EVAL_BLOCK_I at N=$N', file=sys.stderr)
    sys.exit(1)
print(f'dataset OK: {len(sequences)} sequences >= {need} needed')
"

# ---- pass 1: training-block diagnostic (R^emp) -----------------------------
python -u analyze_checkpoints.py \
    --model-dir "$MODEL_DIR" \
    --bp-factorized-layers 0,1,2,3,4 \
    --ladder-batch-size 512

mv "$MODEL_DIR/ladder_posthoc.json" "$OUT_ROOT/ladder_emp_seed${SEED}_N${N}.json"
if [ -f "$MODEL_DIR/ladder_posthoc.npz" ]; then
    mv "$MODEL_DIR/ladder_posthoc.npz" "$OUT_ROOT/ladder_emp_seed${SEED}_N${N}.npz"
fi

# ---- pass 2: held-out-block diagnostic (R^pop) -----------------------------
python -u analyze_checkpoints.py \
    --model-dir "$MODEL_DIR" \
    --eval-block-i "$EVAL_BLOCK_I" \
    --bp-factorized-layers 0,1,2,3,4 \
    --ladder-batch-size 512

mv "$MODEL_DIR/ladder_posthoc.json" "$OUT_ROOT/ladder_pop_seed${SEED}_N${N}.json"
if [ -f "$MODEL_DIR/ladder_posthoc.npz" ]; then
    mv "$MODEL_DIR/ladder_posthoc.npz" "$OUT_ROOT/ladder_pop_seed${SEED}_N${N}.npz"
fi

echo "=== task $SLURM_ARRAY_TASK_ID done -> $OUT_ROOT/ladder_{emp,pop}_seed${SEED}_N${N}.{json,npz} ==="