#!/bin/bash
#SBATCH --job-name=lagreg_detach_ablation
#SBATCH --array=0-5
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%A_%a.out
#
# Three-way detach-ablation at a single N, single reg_lambda, x 2 seeds:
# train_and_eval.py (original single-detach-on-n_local-only version, the
# code that had the klg leak through n_global), train_and_eval_double_detach.py
# (the fix -- klg detached in both n_local and the n_global copy used to form
# lag), train_and_eval_nodetach.py (fully live ablation -- no detach anywhere,
# expected to diverge fastest/worst). N and reg_lambda are FIXED (not swept)
# because the whole point is to isolate the effect of the detach pattern
# alone; sweeping reg_lambda too would multiply the array size for no benefit
# here (reg_lambda=0.0 would make all three scripts numerically identical,
# since the regularizer term -- and therefore the detach pattern inside it --
# is dead code at lambda=0).
#
# N=65536, TARGET_STEPS=25000 (NOT step-matched to any earlier sweep -- this
# is a fresh, smaller-budget run picked for the ablation, not meant to be
# combined with the 100000- or 200000-step full sweeps in the same plot
# without accounting for the different step count).
#
#         N    steps/epoch   n_iter   grad steps
#     65536            128      195        24960   MISMATCH (short by 40;
#                                                    25000 is not divisible
#                                                    by 128 -- same kind of
#                                                    rounding gap flagged in
#                                                    the earlier sweeps)
#
# 6 array tasks = 3 script variants x 2 seeds. Task index decomposes as:
#     script_idx = SLURM_ARRAY_TASK_ID / 2   (0..2 -> which train_and_eval*.py)
#     seed_idx   = SLURM_ARRAY_TASK_ID % 2   (0..1 -> seed)
#
# reg_lambda IS FIXED AT 0.1, not 0.0 -- chosen so the regularizer (and thus
# the detach-pattern difference between the three scripts) is actually
# active. This is the same default used in the earlier full N-sweep. Change
# LAMBDA below if you want a different operating point; there's deliberately
# no lambda_idx in the task decomposition since only one value is run.
#
# ADD_FLAG is set PER SCRIPT VARIANT (not a shared constant like in the
# earlier sweep) specifically so save_path itself differs across the three
# variants: N, seed, n_iter, reduced_length and reg_lambda are IDENTICAL
# across all three tasks at a given seed, and create_save_path() does not key
# on which script produced the run, so without a distinguishing add_flag the
# three variants would collide into the same save_path and their
# train_eval_log_reg0p1.json / ladder_*_reg0p1.json files would overwrite
# each other. reg_tag (from reg_lambda) is the same "0p1" for all three here,
# so it cannot do the disambiguating work it did in the earlier sweep (where
# it separated different LAMBDA values within one add_flag) -- add_flag has
# to carry that weight instead this time.
#
# --time 3:00:00 is a ROUGH, UNVERIFIED guess: 25000/200000 of the earlier
# 12:00:00 budget for the (bigger) full sweep would suggest ~1.5h, but that
# sweep's 12h budget was itself flagged as unverified at the N=65536 end, and
# the per-step BP-oracle-call overhead here does not shrink with fewer total
# steps (--reg-every defaults to 1, i.e. 3 extra BP calls EVERY step) -- so
# the *per-step* cost, not just the total, dominates at this N. Check the
# wall-clock of one task before trusting this for a longer resubmit.
#
# CHECKPOINTS is 45 (matches the plain train_transformer.py baseline sweep
# shown alongside this one, not the 50 used in the earlier full lagreg
# sweep) -- pick whichever you want consistent with your downstream plotting
# code; this is a free parameter, not derived from anything above.
#
# OUTPUT IS NOW FULLY SEPARATED PER SCRIPT VARIANT, not just by add_flag
# inside one shared tree: each variant gets its own --results-dir
# ($PARENT_ROOT/$ADD_FLAG), so create_save_path()'s output for the three
# variants lives under three disjoint directory trees regardless of how that
# function composes its path internally (previously all three shared
# $OUT_ROOT and relied ENTIRELY on add_flag showing up somewhere in
# create_save_path()'s own naming to avoid collision -- that's a weaker
# guarantee than three separate roots, since it depends on unseen internals
# of utils.py).
#
# SLURM's own stdout/stderr logs (logs/%x_%A_%a.out) are NOT further split
# into per-variant subfolders -- #SBATCH --output is resolved by sbatch at
# SUBMIT time, before this script runs, so it only has %x/%A/%a to work
# with, not the ADD_FLAG computed below from SCRIPT_IDX. They're still
# unique per task via %a though: tasks 0-1 = lagreg_origbug, 2-3 =
# lagreg_doubledetach, 4-5 = lagreg_nodetach (same script_idx = %a/2 mapping
# used below), so nothing overwrites, it's just flat rather than nested.

set -euo pipefail

REPO=/scratch/work/chandrb1/biased-generalization
BATCH=512
TARGET_STEPS=25000
CHECKPOINTS=45
EVAL_BLOCK_I=10
N=65536
LAMBDA=0.1

PARENT_ROOT="$REPO/results_lagreg_detach_ablation"
mkdir -p logs

cd "$REPO/scripts"

SCRIPTS=(train_and_eval.py train_and_eval_double_detach.py train_and_eval_nodetach.py)
ADD_FLAGS=(lagreg_origbug lagreg_doubledetach lagreg_nodetach)
SEEDS=(0 1)

TASK_ID=${SLURM_ARRAY_TASK_ID}
SCRIPT_IDX=$(( TASK_ID / 2 ))
SEED_IDX=$(( TASK_ID % 2 ))

SCRIPT=${SCRIPTS[$SCRIPT_IDX]}
ADD_FLAG=${ADD_FLAGS[$SCRIPT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

RESULTS_DIR="$PARENT_ROOT/$ADD_FLAG"
mkdir -p "$RESULTS_DIR"

STEPS_PER_EPOCH=$(( (N + BATCH - 1) / BATCH ))
N_ITER=$(( TARGET_STEPS / STEPS_PER_EPOCH ))
ACTUAL_TOTAL=$(( N_ITER * STEPS_PER_EPOCH ))

echo "=== task $TASK_ID : script=$SCRIPT N=$N seed=$SEED reg_lambda=$LAMBDA add_flag=$ADD_FLAG ==="
echo "steps/epoch = $STEPS_PER_EPOCH   n_iter = $N_ITER   total_steps = $ACTUAL_TOTAL"
if [[ "$ACTUAL_TOTAL" != "$TARGET_STEPS" ]]; then
    echo "WARNING: total_steps ($ACTUAL_TOTAL) != TARGET_STEPS ($TARGET_STEPS) -- " \
         "checkpoint grid will not exactly hit $TARGET_STEPS"
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

python "$SCRIPT" \
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
    --results-dir "$RESULTS_DIR"

echo "=== task $TASK_ID done ==="