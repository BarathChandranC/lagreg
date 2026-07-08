#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WDM_ROOT="${WDM_ROOT:-$REPO_ROOT/wdmdm}"
REPRO_ROOT="${REPRO_ROOT:?Set REPRO_ROOT to an isolated reproduction directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-}"
SAVE_ROOT="$REPRO_ROOT/celeba/checkpoints"
mkdir -p "$SAVE_ROOT"
cd "$WDM_ROOT/Experiments/src/Training"

COMMON=(-n 1024 -s 32 -LR 0.0001 -O Adam -W 32 -t -1 --save-root "$SAVE_ROOT")
case "$ACTION" in
    standard)
        index="${2:?Provide model index 0-14}"
        exec "$PYTHON_BIN" run_Unet.py "${COMMON[@]}" -i "$index" -se "$index" --epochs 5000
        ;;
    filtered)
        method="${2:?Provide PCA or Wavelet}"
        level="${3:?Provide level 0-3}"
        split="${4:?Provide split 0 or 1}"
        FILTERED_DATA_ROOT="${FILTERED_DATA_ROOT:-$WDM_ROOT/Experiments/Data/CelebA_filtered}"
        data_file="$FILTERED_DATA_ROOT/CelebA32_${method}_L${level}_index${split}.pt"
        test_file="$FILTERED_DATA_ROOT/CelebA32_${method}_L${level}_test.pt"
        exec "$PYTHON_BIN" run_Unet.py "${COMMON[@]}" -i "$split" -se "$split" \
            --epochs 3500 --suffix "${method}_L${level}" \
            --data-file "$data_file" --data-test-file "$test_file"
        ;;
    *)
        echo "Usage: $0 standard <index> | filtered <PCA|Wavelet> <level> <split>" >&2
        exit 2
        ;;
esac
