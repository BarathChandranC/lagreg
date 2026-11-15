#!/bin/bash
#SBATCH --job-name=modmnist_dit
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --output=logs/%x_%A_%a.out
#
# DiT-S on Quarter-MNIST, NO regularizer (plain eps-MSE training + diagnostics).
# ONE setting: N = 65536, seed 0. The only thing that varies is the lr schedule.
#
#     task 0   --lr-schedule constant   (the default; folder has no suffix)
#     task 1   --lr-schedule cosine     (folder gets _cos)
#
# Both warm up linearly to --lr over 500 steps. "constant" then holds 1e-4 for
# all 200k steps (what the finished run used); "cosine" decays it to 0.1x = 1e-5
# by the last step, the decay shape the RHM sweep uses. The _cos suffix keeps
# the two from colliding.
#
# Output:
#   results_modmnist/K10_tile32/seed0_N65536_ditS8_sumpair10_pool100v2[_cos]/
#     train_eval_log.json     every scalar, one row per eval
#     samples/cats_step*.png  one real image from each of valid / global /
#                             local / memorized, titled with the tile
#                             classifier's read-out (on by default)
#     samples/grid_step*.png  plain 8x8 grid of samples
#     final_model.pt
# Plot:
#   python plot_modmnist.py --root ../results_modmnist/K10_tile32 \
#       --flags ditS8_sumpair10_pool100v2,ditS8_sumpair10_pool100v2_cos --seeds 0
#
# GEOMETRY is fixed in modules/data.py: TILE_SIZE = 32, so every composite is
# 64x64 with the native 28x28 MNIST digit centred in each quadrant -- Lu et
# al.'s Quarter-MNIST layout. PATCH=8 gives 64 tokens, 16 per tile, which is
# their tokenisation exactly. Other divisors of 32 are available (1,2,4,16,32);
# smaller = finer and more locally biased, larger = coarser and cheaper.
# DiT-S: depth 12, hidden 384, 6 heads, adaLN-Zero, ~32.5M params.
#
# T GRID: 5 noise levels instead of 19. The decomposition is reported per t and
# the t-averages are a midpoint rule over this grid, so fewer points = a coarser
# approximation of the uniform-over-t test loss, and cost scales linearly with
# the number of t (19 -> 5 cuts the decomposition to 26%).
# Raising the tile from 14x14 to 32x32 moves the global-signal window UP in t
# and shrinks it, because the per-tile log-likelihood ratio scales with
# dimension: with 5x more pixels the class is identified at far higher noise,
# eta collapses to one-hot earlier, and m_glob -> 0. On synthetic prototypes the
# peak fell ~10x and moved from t ~ 450-900 to t ~ 750-950. T_FRACS below is set
# for that, but CHECK THE STARTUP TABLE on the first job -- it prints
# E|m_glob|^2 per t -- and move the grid if the window sits elsewhere on real
# MNIST. A grid that misses the window makes every n_global meaningless.
#
# PRECISION (V100 = Volta: fp16 tensor cores, NO bf16, NO TF32; plain fp32
# matmuls run at only ~15.7 TFLOPS, hence fp16 everywhere):
#   --amp fp16        training forward; a GradScaler is enabled automatically.
#                     If train_loss ever logs as nan, drop to --amp none.
#   --sample-amp fp16 DDIM forward only. Samples are just classified / FID'd.
#   --eval-amp fp16   the DECOMPOSITION forward. All estimators still accumulate
#                     in fp32; only the denoiser's output is fp16. fp16 has an
#                     11-bit significand (rel. err ~5e-4 on outputs of order 1),
#                     so the bias it adds to R_global is ~0.04% of
#                     E||m_glob||^2 ~ 6.5e-4 per pixel. bf16 (8-bit, ~4e-3)
#                     would NOT be usable here -- its error alone exceeds
#                     E||m_glob||^2. SANITY CHECK: pop_t*_pyth_resid in the log
#                     must stay at the same order as an fp32 run (~1e-4); if it
#                     grows, set --eval-amp none.
#                     On an A100/H100/H200 partition, switch all three to bf16
#                     (wider range, no GradScaler) and drop the partition pin.
#   --compile         torch.compile(dynamic=False), errors suppressed to eager.
#                     1-3 min of compilation up front.
#
# --time 12h is a GUESS. Run task 0 alone first and read its step rate and
# eval_seconds before submitting task 1:
#     sbatch --array=0 run_modmnist_dit_sweep.sh
#
# Submit from $REPO/scripts after:  mkdir -p logs

set -euo pipefail

REPO=${REPO:-$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)}
# RULE / K. The tile class is (digit % K), so K=10 means the class IS the digit.
#   modk    + K=5   sum c_i = 0 (mod 5).  Every TRIPLE of classes is exactly
#                   independent: no pairwise or three-way signal exists, the
#                   constraint lives only in the four-way joint.  chance = 0.200
#   sumpair + K=10  c_0+c_1 = c_2+c_3 on digits (Lu et al., 670 valid tuples).
#                   Pairwise correlated, so gradient descent has a low-order
#                   path to it -- and FID is NOT blind to this rule.
#                   chance = 0.0705 (non-uniform per-tile marginals).
#   pairwise+ K     c_3 = c_0, c_1/c_2 free.  Control isolating whether the
#                   four-way structure is the obstacle.  chance = 1/K.
# Non-default rules are appended to the run folder name (e.g. _sumpair10), so
# they never collide with the mod-5 runs.  The tile classifier is a plain digit
# classifier and is shared by every rule.
RULE=sumpair
K=10
N=65536
SEED=0
POOL_PER_DIGIT=100
TOTAL_STEPS=200000
BATCH=16
PATCH=8            # 32px tile / 8 = 16 tokens per tile (Lu et al.); divisors: 1,2,4,8,16,32
OUT_ROOT="$REPO/results_modmnist"
FLAG=pool100v2
T_FRACS=0.5,0.75,0.85,0.92,0.97

SCHEDULES=(constant cosine)
TASK_ID=${SLURM_ARRAY_TASK_ID}
SCHED=${SCHEDULES[$TASK_ID]}

cd "$REPO/scripts"

CLF="../checkpoints/tile_cnn_tile32.pt"
if [ ! -f "$CLF" ]; then
    echo "ERROR: $CLF missing -- run train_tile_classifier.py first" >&2
    exit 1
fi

echo "=== task $TASK_ID : DiT-S/$PATCH rule=$RULE K=$K N=$N seed=$SEED sched=$SCHED pool/digit=$POOL_PER_DIGIT steps=$TOTAL_STEPS batch=$BATCH ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u train_modmnist.py \
    --K "$K" --rule "$RULE" \
    --pool-per-digit "$POOL_PER_DIGIT" --pool-seed 0 \
    --n-train "$N" --seed "$SEED" \
    --total-steps "$TOTAL_STEPS" --batch-size "$BATCH" --lr 1e-4 \
    --lr-schedule "$SCHED" \
    --n-evals 25 --eval-t-fracs "$T_FRACS" \
    --n-eval 256 --n-completions 12 --eval-batch-size 512 \
    --n-gen 1000 --sample-steps 250 \
    --fid --fid-n-ref 10000 \
    --classifier-path "$CLF" \
    --mnist-root ../data \
    --arch dit --dit-size S --dit-patch "$PATCH" \
    --amp fp16 \
    --sample-amp fp16 \
    --eval-amp fp16 \
    --compile \
    --save-final-model \
    --out-root "$OUT_ROOT" \
    --add-flag "$FLAG"

echo "=== done (N=$N seed=$SEED sched=$SCHED) ==="