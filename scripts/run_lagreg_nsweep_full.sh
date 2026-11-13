#!/bin/bash
#SBATCH --job-name=lagreg_nsweep_full
#SBATCH --array=0-111%42
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out
#
# N in {512,1024,2048,4096,8192,16384,65536} x seed in {2..9} x
# reg_lambda in {0.0 (control), 0.1}, step-matched at 200000 gradient steps,
# CHECKPOINTS log-spaced evals (step 100 .. total) via train_and_eval.py.
#
# Seeds 0 and 1 already exist from the earlier sweep, so this runs seeds 2..9.
# 112 array tasks = 8 seeds x 7 N x 2 lambdas. Tasks are SEED-MAJOR: each
# seed's 14 jobs (7 N x 2 lambdas) have consecutive indices, and %42 caps
# concurrency at 42 = 3 seeds' worth. Slurm dispatches in index order, so the
# sweep works through roughly 3 seeds at a time (a 4th seed's first jobs can
# start as soon as any running job finishes). Task index:
#     seed_idx   = TASK_ID / (N_N * N_LAMBDAS)
#     rem        = TASK_ID % (N_N * N_LAMBDAS)
#     n_idx      = rem / N_LAMBDAS
#     lambda_idx = rem % N_LAMBDAS
#
#         N    steps/epoch   n_iter   grad steps
#       512              1   200000       200000
#      1024              2   100000       200000
#      2048              4    50000       200000
#      4096              8    25000       200000
#      8192             16    12500       200000
#     16384             32     6250       200000
#     65536            128     1562       199936   (64 short, 0.03%)
#
# OUTPUT DIRECTORIES: reg_lambda is part of ADD_FLAG (lagreg_reg0p0 /
# lagreg_reg0p1), so the two lambdas at the same (N, seed) never write into
# one shared directory concurrently. Filenames still carry _reg{tag}.
#
# BEFORE SUBMITTING:
#   - mkdir -p logs  in the directory you submit from (Slurm opens the
#     --output file before this script runs)
#   - check wall-clock of one N=65536 task: --n-ladder-eval=N makes the
#     ladder eval at large N the dominant cost; 12h is unverified there
#   - dataset needs >= (EVAL_BLOCK_I+1)*N = 720896 sequences at N=65536
#     (checked below)
 
set -euo pipefail
 
REPO=/scratch/work/chandrb1/biased-generalization
BATCH=512
TARGET_STEPS=200000
CHECKPOINTS=50
EVAL_BLOCK_I=10
 
OUT_ROOT="$REPO/results_lagreg_sweep"
mkdir -p "$OUT_ROOT"
 
cd "$REPO/scripts"
 
N_VALS=(512 1024 2048 4096 8192 16384 65536)
SEEDS=(2 3 4 5 6 7 8 9)
LAMBDAS=(0.0 0.1)
 
N_N=${#N_VALS[@]}
N_LAMBDAS=${#LAMBDAS[@]}
 
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$(( TASK_ID / (N_N * N_LAMBDAS) ))
REM=$(( TASK_ID % (N_N * N_LAMBDAS) ))
N_IDX=$(( REM / N_LAMBDAS ))
LAMBDA_IDX=$(( REM % N_LAMBDAS ))
 
N=${N_VALS[$N_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LAMBDA=${LAMBDAS[$LAMBDA_IDX]}
ADD_FLAG="lagreg_reg${LAMBDA/./p}"
 
STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
ACTUAL_TOTAL=$(( N_ITER * STEPS_PER_EPOCH ))
 
echo "=== task $TASK_ID : N=$N seed=$SEED reg_lambda=$LAMBDA add_flag=$ADD_FLAG ==="
echo "steps/epoch = $STEPS_PER_EPOCH   n_iter = $N_ITER   total_steps = $ACTUAL_TOTAL"
if [[ "$ACTUAL_TOTAL" != "$TARGET_STEPS" ]]; then
    echo "WARNING: total_steps ($ACTUAL_TOTAL) != TARGET_STEPS ($TARGET_STEPS)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
 
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