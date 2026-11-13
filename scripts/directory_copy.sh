#!/bin/bash
# collect_lagreg_logs.sh -- copy train_eval_log_reg{0p0,0p1}.json for the
# TARGET=200000 generation, every N, preserving the directory structure.
set -euo pipefail

SRC=/scratch/work/chandrb1/biased-generalization/results_lagreg_sweep   # where the runs live
DST_ROOT=/scratch/work/chandrb1/biased-generalization/results_lagreg_200k
TARGET=200000
BATCH=512
LENGTHS=(512 1024 2048 4096 8192 16384 65536)
TAGS=(0p0 0p1)
DRY_RUN=${DRY_RUN:-0}            # DRY_RUN=1 bash collect_lagreg_logs.sh  -> only list

for TAG in "${TAGS[@]}"; do
  DST="$DST_ROOT/reg$TAG"
  for N in "${LENGTHS[@]}"; do
    SPE=$(( (N + BATCH - 1) / BATCH ))
    NIT=$(( TARGET / SPE ))
    n=0
    while IFS= read -r -d '' f; do
      rel="${f#"$SRC"/}"
      if [ "$DRY_RUN" = 1 ]; then
        echo "  $rel"
      else
        mkdir -p "$DST/$(dirname "$rel")"
        cp -p "$f" "$DST/$rel"
      fi
      n=$((n + 1))
    done < <(find "$SRC" -path "*test_model_script_${NIT}_${N}_*" \
                         -name "train_eval_log_reg${TAG}.json" -print0)
    echo "reg$TAG  N=$N  n_iter=$NIT  -> $n file(s)"
  done
done