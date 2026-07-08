#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${CONTROLLED_MANIFEST:-$REPO_ROOT/reproduction/controlled_manifest.csv}"
REPRO_ROOT="${REPRO_ROOT:?Set REPRO_ROOT to an isolated reproduction directory}"
RHM_DATA="${RHM_DATA:-$REPO_ROOT/data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INDEX="${1:-${RUN_INDEX:-${SLURM_ARRAY_TASK_ID:-}}}"

if [[ -z "$RUN_INDEX" || ! "$RUN_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Usage: RUN_INDEX=<0-56> bash reproduction/run_controlled_training.sh" >&2
    exit 2
fi

row="$(sed -n "$((RUN_INDEX + 2))p" "$MANIFEST")"
if [[ -z "$row" ]]; then
    echo "No manifest row at zero-based index $RUN_INDEX" >&2
    exit 2
fi
IFS=, read -r run_id setting train_size split seed n_iter batch_size test_block analysis_role <<< "$row"
if [[ "$setting" != "controlled" ]]; then
    echo "Unexpected manifest setting: $setting" >&2
    exit 2
fi

RESULTS_DIR="$REPRO_ROOT/controlled/checkpoints"
mkdir -p "$RESULTS_DIR" "$REPRO_ROOT/controlled/metadata"
printf '%s\n' "$row" > "$REPRO_ROOT/controlled/metadata/$run_id.csv"

cd "$REPO_ROOT/scripts"
exec "$PYTHON_BIN" train_transformer.py \
    --data-path "$RHM_DATA" \
    --results-dir "$RESULTS_DIR" \
    --seed "$seed" \
    --reduced-length "$train_size" \
    --pick-i-for-training "$split" \
    --batch-size "$batch_size" \
    --n-iter "$n_iter" \
    --use-cross-entropy-loss
