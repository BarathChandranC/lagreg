#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WDM_ROOT="${WDM_ROOT:-$REPO_ROOT/wdmdm}"
REPRO_ROOT="${REPRO_ROOT:?Set REPRO_ROOT to an isolated reproduction directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-}"
shift || true
MODEL_ROOT="${MODEL_ROOT:-$REPRO_ROOT/celeba/checkpoints}"
OUT_ROOT="${OUT_ROOT:-$REPRO_ROOT/celeba/evaluations}"
mkdir -p "$OUT_ROOT"

case "$ACTION" in
    score)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" compare_scores.py -D CelebA -n 1024 -s 32 -LR 0.0001 \
            -O Adam -W 32 -B 512 -Ns 1000 --model_root "$MODEL_ROOT" \
            --out_dire "$OUT_ROOT" "$@"
        ;;
    loss)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" loss_compute.py -D CelebA -n 1024 -s 32 -LR 0.0001 \
            -O Adam -W 32 -B 512 --eval_N 50 --model_root "$MODEL_ROOT" \
            --out_dire "$OUT_ROOT" "$@"
        ;;
    sample)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" sample_split_inference.py --scan_all_pairs --indices 0-14 \
            -n 1024 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -Ns 512 \
            --model-root "$MODEL_ROOT" --out-dir "$OUT_ROOT" "$@"
        ;;
    aggregate)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" cos_dis_aggregate.py --saves_dir "$OUT_ROOT" \
            --out_dir "$OUT_ROOT" -n 1024 "$@"
        ;;
    filter)
        CELEBA_TENSOR="${CELEBA_TENSOR:-$WDM_ROOT/Experiments/Data/CelebA/CelebA32.pt}"
        FILTERED_DATA_ROOT="${FILTERED_DATA_ROOT:-$WDM_ROOT/Experiments/Data/CelebA_filtered}"
        cd "$WDM_ROOT/Experiments/src/Utils"
        exec "$PYTHON_BIN" filter_celeba.py --input-path "$CELEBA_TENSOR" \
            --output-dir "$FILTERED_DATA_ROOT" --n-train 1024 --n-test 2048 "$@"
        ;;
    freeze)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" select_freeze_checkpoint.py "$@"
        ;;
    cross)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" compare_scores_cross.py -s 32 -W 32 -Ns 512 \
            --device "${DEVICE:-cuda:0}" "$@"
        ;;
    plot)
        cd "$WDM_ROOT/Experiments/src/Generation"
        exec "$PYTHON_BIN" celeba_multiscale_complexity_plot.py \
            --result-root "$OUT_ROOT" --methods PCA,Wavelet --levels 0,1,2 \
            --time 100 --out-stem celeba_wavelet_pca_complexity "$@"
        ;;
    *)
        echo "Usage: $0 {score|loss|sample|aggregate|filter|freeze|cross|plot} [script arguments]" >&2
        exit 2
        ;;
esac
