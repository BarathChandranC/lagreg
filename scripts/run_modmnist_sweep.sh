#!/bin/bash
#SBATCH --job-name=modmnist_reg
#SBATCH --array=0-44
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out
#
# 45 tasks = 5 N x 3 seeds x 3 lambdas. Task index:
#     lambda_idx = TASK_ID % N_LAMBDAS
#     seed_idx   = (TASK_ID / N_LAMBDAS) % N_SEEDS
#     n_idx      = TASK_ID / (N_LAMBDAS * N_SEEDS)
#
# lambda = 0 is the control: --reg-monitor computes and logs the same per-batch
# n's (batch_n_global, batch_n_local, batch_lag) without applying the penalty.
# Every run uses --reg-monitor, so all three lambdas log identical keys.
#
# Output folders (train_modmnist.py appends _reg<lambda> only when lambda > 0):
#     seed{S}_N{N}_pool100v2              lambda = 0 (control)
#     seed{S}_N{N}_pool100v2_reg0p1       lambda = 0.1
#     seed{S}_N{N}_pool100v2_reg0p5       lambda = 0.5
# Plot each with --flag pool100v2 / pool100v2_reg0p1 / pool100v2_reg0p5.
#
# Cost: the regularizer adds reg_batch * 4 * reg_completions = 32*4*2 = 256
# extra forward+backward passes per step on top of the batch of 128 (about 3x
# the training compute per step; evals unchanged). The lambda = 0 control pays
# the forward passes only. Check one task's step rate early and adjust --time.
#
# Submit from $REPO/scripts after:  mkdir -p logs

set -euo pipefail

REPO=/scratch/work/chandrb1/biased-generalization
K=5
TILE=14
POOL_PER_DIGIT=100
TOTAL_STEPS=200000
BATCH=128
OUT_ROOT="$REPO/results_modmnist"
FLAG=pool100v2
T_FRACS=0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95

LENGTHS=(256 1024 4096 16384 65536)
SEEDS=(0 1 2)
LAMBDAS=(0.0 0.1 0.5)

N_SEEDS=${#SEEDS[@]}
N_LAMBDAS=${#LAMBDAS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
LAMBDA_IDX=$(( TASK_ID % N_LAMBDAS ))
SEED_IDX=$(( (TASK_ID / N_LAMBDAS) % N_SEEDS ))
N_IDX=$(( TASK_ID / (N_LAMBDAS * N_SEEDS) ))

N=${LENGTHS[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LAMBDA=${LAMBDAS[$LAMBDA_IDX]}

cd "$REPO/scripts"

CLF="../checkpoints/tile_cnn_tile${TILE}.pt"
if [ ! -f "$CLF" ]; then
    echo "ERROR: $CLF missing -- run train_tile_classifier.py --tile-size $TILE first" >&2
    exit 1
fi

echo "=== task $TASK_ID : K=$K N=$N seed=$SEED lambda=$LAMBDA pool/digit=$POOL_PER_DIGIT steps=$TOTAL_STEPS ==="

python -u train_modmnist.py \
    --K "$K" --tile-size "$TILE" \
    --pool-per-digit "$POOL_PER_DIGIT" --pool-seed 0 \
    --n-train "$N" --seed "$SEED" \
    --total-steps "$TOTAL_STEPS" --batch-size "$BATCH" --lr 2e-4 \
    --n-evals 25 --eval-t-fracs "$T_FRACS" \
    --n-eval 256 --n-completions 12 \
    --n-gen 5000 --sample-steps 250 \
    --fid --fid-n-ref 10000 \
    --classifier-path "$CLF" \
    --mnist-root ../data \
    --reg-lambda "$LAMBDA" \
    --reg-margin 0.0 \
    --reg-batch 32 \
    --reg-completions 2 \
    --reg-every 1 \
    --reg-monitor \
    --out-root "$OUT_ROOT" \
    --add-flag "$FLAG"

echo "=== done (N=$N seed=$SEED lambda=$LAMBDA) ==="
