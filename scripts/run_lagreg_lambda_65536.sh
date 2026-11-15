#!/bin/bash
#SBATCH --job-name=lagreg_lam65536
#SBATCH --array=0-39
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out
#
# lambda scan at N = 65536 for the RHM lag regularizer:
#     reg_lambda in {0.01, 0.03, 0.3, 1.0}  x  seed in {0..9}  = 40 tasks
# (0.0 and 0.1 already exist from run_lagreg_nsweep_full.sh / the earlier sweep.)
#
# Same settings as run_lagreg_nsweep_full.sh (200000 steps, margin 0.08,
# huber 1.0, same model/lr) and the same results root + add-flag convention
# (lagreg_reg<tag>), so these runs sit next to the existing reg0p0 / reg0p1
# runs and the combine_* scripts pick them up with --reg-lambda <value>.
#
# Task index (seed-major, lambda fastest):
#     seed_idx   = TASK_ID / N_LAMBDAS
#     lambda_idx = TASK_ID % N_LAMBDAS
#
# Submit from $REPO/scripts after:  mkdir -p logs

set -euo pipefail

REPO=${REPO:-$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)}
BATCH=512
TARGET_STEPS=200000
CHECKPOINTS=50
EVAL_BLOCK_I=10
N=65536

OUT_ROOT="$REPO/results_lagreg_sweep"
mkdir -p "$OUT_ROOT"
cd "$REPO/scripts"

SEEDS=(0 1 2 3 4 5 6 7 8 9)
LAMBDAS=(0.01 0.03 0.3 1.0)
N_LAMBDAS=${#LAMBDAS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( TASK_ID / N_LAMBDAS ))]}
LAMBDA=${LAMBDAS[$(( TASK_ID % N_LAMBDAS ))]}
ADD_FLAG="lagreg_reg${LAMBDA/./p}"

STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))

echo "=== task $TASK_ID : N=$N seed=$SEED reg_lambda=$LAMBDA add_flag=$ADD_FLAG ==="
echo "steps/epoch = $STEPS_PER_EPOCH   n_iter = $N_ITER   total_steps = $(( N_ITER * STEPS_PER_EPOCH ))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_and_eval.py \
    --seed "$SEED" \
    --data-path ../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy \
    --reduced-length "$N" \
    --pick-i-for-training 0 \
    --eval-block-i "$EVAL_BLOCK_I" \
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
    --use-cross-entropy-loss \
    --n-evals "$CHECKPOINTS" \
    --n-ladder-eval "$N" \
    --n-gen 1000 \
    --reg-lambda "$LAMBDA" \
    --reg-delta-margin 0.08 \
    --reg-delta-huber 1.0 \
    --add-flag "$ADD_FLAG" \
    --results-dir "$OUT_ROOT"

echo "=== task $TASK_ID done ==="