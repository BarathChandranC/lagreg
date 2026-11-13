#!/bin/bash
#SBATCH --job-name=lagreg_lambda_sweep_field
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out
#
# LAMBDA in {0.0, 0.1}, single N (4096), single seed (0), STEP-MATCHED at
# 100000 gradient steps, 45 log-spaced STEP-based eval points, single
# attractor-field diffusion timestep t=50. Adapted from the N-sweep
# template (results_lagreg_field_sweep) -- here the array axis is LAMBDA
# instead of N, so N/SEED/CHECKPOINTS/TARGET_STEPS are fixed constants and
# only LAMBDA is indexed by SLURM_ARRAY_TASK_ID.
#
#     task    reg_lambda
#        0           0.0
#        1           0.1
#
#        N    steps/epoch   n_iter   grad steps
#     4096              8    12500       100000   exact
#
# OUTPUTS: save_path depends on N/n_iter/seed/add_flag but NOT on
# reg_lambda, so both tasks would otherwise share one save_path.
# train_eval_log/ladder_emp/ladder_pop/full_params ARE already
# reg_lambda-tagged by train_and_eval_field.py itself (reg_tag =
# str(lambda) with '.'->'p'), so those never collide -- but
# attractor_plots/step_{gstep:07d}_t{t:04d}.png is NOT reg-tagged, and
# both lambda values share the same eval-step grid (same N/n_iter/n_evals),
# so without separating them the two tasks WOULD overwrite each other's
# attractor PNGs. Fixed here by folding lambda into ADD_FLAG, giving each
# task its own save_path entirely (simplest fix, and consistent with how
# the N-sweep template already separates by giving each N its own
# save_path).
#
# --save-gen-samples added: dumps the raw (n_gen x seq_len) generated token
# array at every eval step to <save_path>/gen_samples/gen_samples_step_{gstep:07d}.npz,
# plus the one fixed noise batch to <save_path>/gen_samples/x_init.npz --
# gen_samples/ is per-task (since save_path is per-lambda via ADD_FLAG
# above), so the two lambda tasks' generated samples never collide either.
#
# ATTRACTOR-FIELD: single t=50 (not the previous 50,100,200) --
# --attractor-grid-n 15 kept modest since this still multiplies against
# CHECKPOINTS(=45) extra model forward passes per task on top of the
# lagreg training itself -- check ONE task's wall-clock before trusting
# --time 12:00:00.
#
# UNVERIFIED, WORTH CHECKING BEFORE A LONG SUBMIT (carried over from the
# N-sweep template):
#   - --reg-delta-margin/--reg-delta-huber stay at script defaults
#     (0.08 / 1.0) -- calibrated/left unmeasured from a single N=4096,
#     lambda=0 control run, not re-derived for lambda=0.1 specifically.
#   - eval_seqs bounds check needs len(sequences) >= (EVAL_BLOCK_I+1)*N
#     = 11*4096 = 45056 here.
#   - the attractor companion search (x_a/x_b/x_c) runs independently PER
#     TASK (it depends on the train block, which is identical here since N
#     and seed are fixed across both lambda values) -- so unlike the
#     N-sweep, x_a/x_b/x_c and the alpha/beta axes SHOULD end up identical
#     across the two lambda tasks (same train block, same
#     find_companions_nn() call), making the two tasks' attractor PNGs
#     directly comparable on the same axes. Worth confirming meta.json
#     matches between the two output directories once both tasks finish.

set -euo pipefail

REPO=/scratch/work/chandrb1/biased-generalization
BATCH=512
TARGET_STEPS=100000
CHECKPOINTS=45
EVAL_BLOCK_I=10
SEED=0
N=4096
ATTRACTOR_T=50

OUT_ROOT="$REPO/results_lagreg_field_sweep"
mkdir -p "$OUT_ROOT" logs

cd "$REPO/scripts"

LAMBDA_VALS=(0.0 0.1)
LAMBDA=${LAMBDA_VALS[$SLURM_ARRAY_TASK_ID]}
LAMBDA_TAG=$(echo "$LAMBDA" | tr '.' 'p')
ADD_FLAG="lagreg_field_lam${LAMBDA_TAG}"

STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
ACTUAL_TOTAL=$(( N_ITER * STEPS_PER_EPOCH ))

echo "=== task $SLURM_ARRAY_TASK_ID : N=$N seed=$SEED reg_lambda=$LAMBDA add_flag=$ADD_FLAG ==="
echo "steps/epoch = $STEPS_PER_EPOCH   n_iter = $N_ITER   total_steps = $ACTUAL_TOTAL"
if [[ "$ACTUAL_TOTAL" != "$TARGET_STEPS" ]]; then
    echo "WARNING: total_steps ($ACTUAL_TOTAL) != TARGET_STEPS ($TARGET_STEPS) -- " \
         "checkpoint grid for this run will not exactly match a differently-N'd run"
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

python train_and_eval_field.py \
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
    --results-dir "$OUT_ROOT" \
    --save-final-model \
    --plot-attractor \
    --attractor-save-npz \
    --attractor-ts "$ATTRACTOR_T" \
    --attractor-grid-n 15 \
    --save-gen-samples

echo "=== task $SLURM_ARRAY_TASK_ID done ==="