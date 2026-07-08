#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO_ROOT="${REPRO_ROOT:?Set REPRO_ROOT to an isolated reproduction directory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-}"
shift || true
OUT="$REPRO_ROOT/controlled/evaluations/$ACTION"
mkdir -p "$OUT"
cd "$REPO_ROOT/scripts"

case "$ACTION" in
    sequential)
        exec "$PYTHON_BIN" sequential_learning.py \
            --N-test 3000 --reps 5 --test-block-i 5 --output-root "$OUT" "$@"
        ;;
    score-time)
        exec "$PYTHON_BIN" score_divergence_along_t.py \
            --train-size 5000 --train-size-2 12000 --train-size-3 70000 \
            --N-test 2000 --reps 5 --test-block-i 5 \
            --bp-factorized-layers "0,1,2" --data-ids "0,0,1" \
            --same-dataset 0 --output-root "$OUT" "$@"
        ;;
    nn)
        exec "$PYTHON_BIN" nn_divergence.py \
            --N-test 50000 --test-block-i 5 --output-root "$OUT" "$@"
        ;;
    loss)
        exec "$PYTHON_BIN" loss_decomposition.py \
            --fixed-t-ratio 0.3 --test-block-i 5 --plots --output-root "$OUT" "$@"
        ;;
    uturn-main)
        exec "$PYTHON_BIN" uturn_overlap.py \
            --N-test 1000 --reps 100 --test-block-i 5 --output-root "$OUT" "$@"
        ;;
    uturn-appendix)
        exec "$PYTHON_BIN" uturn_overlap.py \
            --N-test 100 --reps 1000 --test-block-i 5 --output-root "$OUT" "$@"
        ;;
    uturn-random)
        exec "$PYTHON_BIN" uturn_overlap_random.py \
            --N-test 100 --reps 100 --output-root "$OUT" "$@"
        ;;
    *)
        echo "Usage: $0 {sequential|score-time|nn|loss|uturn-main|uturn-appendix|uturn-random} [script arguments]" >&2
        exit 2
        ;;
esac
