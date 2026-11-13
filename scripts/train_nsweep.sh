#!/bin/bash
#SBATCH --job-name=train_nsweep_full
#SBATCH --array=0-13
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out
#
# Fresh sweep, 512 .. 65536 x 2 seeds, STEP-MATCHED at 10^5 = 100000
# gradient steps, 45 log-spaced STEP-based checkpoints (step 100 .. 100000).
#
# 14 array tasks = 7 dataset sizes x 2 seeds. Task index decomposes as:
#     n_idx    = SLURM_ARRAY_TASK_ID / 2   (0..6 -> N)
#     seed_idx = SLURM_ARRAY_TASK_ID % 2   (0..1 -> seed)
#
#        N    steps/epoch   n_iter   grad steps
#      512              1   100000       100000
#     1024              2    50000       100000
#     2048              4    25000       100000
#     4096              8    12500       100000
#     8192             16     6250       100000
#    16384             32     3125       100000
#    65536            128      782        100096   (not exact -- see WARNING below)

set -euo pipefail

REPO=${REPO:-$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)}
BATCH=512
TARGET_STEPS=100000
CHECKPOINTS=45

cd "$REPO/scripts"

LENGTHS=(512 1024 2048 4096 8192 16384 65536)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${LENGTHS[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}

STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
ACTUAL_TOTAL=$(( N_ITER * STEPS_PER_EPOCH ))

echo "=== task $SLURM_ARRAY_TASK_ID : N=$N seed=$SEED ==="
echo "steps/epoch = $STEPS_PER_EPOCH   n_iter = $N_ITER   total_steps = $ACTUAL_TOTAL"
if [[ "$ACTUAL_TOTAL" != "$TARGET_STEPS" ]]; then
    echo "WARNING: total_steps ($ACTUAL_TOTAL) != TARGET_STEPS ($TARGET_STEPS) -- " \
         "checkpoint grid for this N will not exactly match the others"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_transformer.py \
    --seed "$SEED" \
    --data-path ../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy \
    --reduced-length "$N" \
    --pick-i-for-training 0 \
    --batch-size "$BATCH" \
    --n-iter "$N_ITER" \
    --hidden-size-encoder 512 \
    --num-layers 8 \
    --nhead 4 \
    --model-channels 32 \
    --t-final 500 \
    --s 1e-4 \
    --lr 3e-4 \
    --lr-schedule default \
    --checkpointing-period-steps "$CHECKPOINTS" \
    --use-cross-entropy-loss

echo "=== task $SLURM_ARRAY_TASK_ID done ==="