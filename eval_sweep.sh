#!/bin/bash
#SBATCH --job-name=eval_nsweep_seeds
#SBATCH --array=0-13
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

REPO=/scratch/work/chandrb1/biased-generalization
BATCH=512
TARGET_STEPS=100000

cd "$REPO/scripts"

LENGTHS=(512 1024 2048 4096 8192 16384 65536)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${LENGTHS[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}

STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))

echo "=== task $SLURM_ARRAY_TASK_ID : N=$N seed=$SEED n_iter=$N_ITER ==="

SUBPATH="results_transformer_pick_0_for_training/models_restricted_continuous_${SEED}/test_model_script_${N_ITER}_${N}_cross_entropy_loss/8_512"

python -u sequential_learning.py \
    --models-root "$REPO" \
    --subpath "$SUBPATH" \
    --N-test 1000 \
    --reps 5 \
    --t-denoising-ratio 0.15 \
    --n-gen 1000 \
    --bp-factorized-layers 0,1,2,3,4 \
    --bp-backend torch \
    --output-root ../plots \
    --ckpt-tag step \
    --job-id "seed${SEED}_N${N}"

echo "=== task $SLURM_ARRAY_TASK_ID done ==="