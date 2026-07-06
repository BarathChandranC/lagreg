# Reproduction launchers

Run launchers from the repository root. Set `REPRO_ROOT` to a new output
directory for each checkpoint profile so results from different training runs
cannot overwrite one another.

## Controlled setting

`controlled_manifest.csv` is the source of truth for the 57 training runs:
45 balanced runs (three sizes × five splits × three seeds), plus seeds 3–14
for the 12k split-0 ensemble. Test data always comes from block 5.

```bash
export REPRO_ROOT=/path/to/reproduction/<git-sha>
export RHM_DATA=/path/to/labeled_data_restrictedfixed_6_4_1.0_4_0.npy

# zero-based manifest row, also accepts SLURM_ARRAY_TASK_ID
bash reproduction/run_controlled_training.sh 0
```

Training outputs share this deterministic layout below
`$REPRO_ROOT/controlled/checkpoints`:

```text
results_transformer_pick_<split>_for_training/
  models_restricted_continuous_<seed>/
    test_model_script_<iterations>_<size>_cross_entropy_loss/8_512/
```

### Explicit checkpoint profile

The following profile assumes the 12k split-0 ensemble layout produced by the
training launcher. Change `CHECKPOINT_ROOT` and `REPRO_ROOT` to evaluate a
second checkpoint source with identical code and settings.

```bash
export CHECKPOINT_ROOT=/absolute/path/to/controlled/checkpoints
export REPRO_ROOT=/absolute/path/to/outputs/checkpoint-profile-name
export PYTHON_BIN="${PYTHON_BIN:-python}"

SUBPATH=test_model_script_30000_12000_cross_entropy_loss/8_512
MODEL0=results_transformer_pick_0_for_training/models_restricted_continuous_0/$SUBPATH
MODEL1=results_transformer_pick_0_for_training/models_restricted_continuous_1/$SUBPATH

PATHS_15=""
for seed in $(seq 0 14); do
    path="$CHECKPOINT_ROOT/results_transformer_pick_0_for_training/models_restricted_continuous_${seed}/$SUBPATH"
    PATHS_15="${PATHS_15:+$PATHS_15,}$path"
done

# Fig. 1(b) and Fig. 6: split-0 seeds 0 and 1.
bash reproduction/run_controlled_evaluation.sh sequential \
    --models-root "$CHECKPOINT_ROOT" --subpath1 "$MODEL0" --subpath2 "$MODEL1" \
    --train-size 12000 --same-dataset 1 --bp-factorized-layers 0,1,2

# Fig. 4(a) and appendix NN-divergence panels: full 15-seed trajectory.
bash reproduction/run_controlled_evaluation.sh nn --paths-gen "$PATHS_15"

# Fig. 4(b): full split-0 15-seed loss decomposition.
bash reproduction/run_controlled_evaluation.sh loss \
    --base-dir "$CHECKPOINT_ROOT/results_transformer_pick_0_for_training" \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
    --subpath "$SUBPATH"

# Fig. 5: run once at each epoch selected by the NN-divergence analysis.
UTURN_PATHS="$CHECKPOINT_ROOT/$MODEL0,$CHECKPOINT_ROOT/$MODEL1,<absolute-disjoint-model-path>"
bash reproduction/run_controlled_evaluation.sh uturn-main \
    --paths-gen "$UTURN_PATHS" --which-epoch <nn-minimum>
bash reproduction/run_controlled_evaluation.sh uturn-main \
    --paths-gen "$UTURN_PATHS" --which-epoch <test-loss-minimum>

# Appendix full-training U-turn and random-start diagnostic.
bash reproduction/run_controlled_evaluation.sh uturn-appendix \
    --paths-gen "$UTURN_PATHS" --which-epoch <selected-epoch>
bash reproduction/run_controlled_evaluation.sh uturn-random \
    --paths-gen "$CHECKPOINT_ROOT/$MODEL0" --which-epoch <selected-epoch>
```

Fig. 3(b) compares 5k, 12k, and 70k models and therefore requires all three
checkpoint families. It cannot be produced from a 12k-only retraining root.

```bash
bash reproduction/run_controlled_evaluation.sh score-time \
    --base-path "$CHECKPOINT_ROOT/results_transformer_pick_{}_for_training" \
    --which-epoch1 <5k-test-loss-minimum> \
    --which-epoch2 <12k-test-loss-minimum> \
    --which-epoch3 <70k-test-loss-minimum>
```

The evaluation launcher fixes the figure-specific public paper settings:
3,000×5 for fixed-time sequential/sample-split evaluation, 2,000×5 for the
Fig. 3(b) along-time comparison, and 50,000 examples for NN divergence.
`uturn-main` uses 1,000 starting points × 100 trajectories for Fig. 5;
`uturn-appendix` uses 100 starting points × 1,000 trajectories for the full
appendix sequences. Required model/checkpoint selections remain explicit CLI
arguments so evaluation never silently discovers a different run.

## CelebA

Set a distinct output and checkpoint root for each standard or filtered CelebA
profile:

```bash
export REPRO_ROOT=/absolute/path/to/outputs/checkpoint-profile-name
export MODEL_ROOT=/absolute/path/to/celeba/checkpoints
export OUT_ROOT="$REPRO_ROOT/celeba/evaluations"

# 15 standard models, indices 0..14
bash reproduction/run_celeba_training.sh standard 0

# 16 filtered models: two methods × four levels × two splits
bash reproduction/run_celeba_training.sh filtered Wavelet 2 0

# Standard and multiscale evaluation entry points
for index in $(seq 0 14); do
    bash reproduction/run_celeba_evaluation.sh loss -i "$index"
done
bash reproduction/run_celeba_evaluation.sh score -is 0 -ie 14 -t 50
bash reproduction/run_celeba_evaluation.sh score -is 0 -ie 14 -t 100
bash reproduction/run_celeba_evaluation.sh score -is 0 -ie 14 -t 150
bash reproduction/run_celeba_evaluation.sh score -is 0 -ie 14 -t 200
bash reproduction/run_celeba_evaluation.sh sample
bash reproduction/run_celeba_evaluation.sh aggregate
bash reproduction/run_celeba_evaluation.sh filter
bash reproduction/run_celeba_evaluation.sh freeze --policy max_similarity \
    --comparisons-dir /path/to/within-level/results
bash reproduction/run_celeba_evaluation.sh cross \
    --model-a-dir /path/to/L3 --model-b-dir /path/to/L2 \
    --ckpt-b <selected> --data-test-file /path/to/test.pt \
    --out-dir /path/to/cross --label L3_vs_L2
bash reproduction/run_celeba_evaluation.sh plot
```

The `plot` action uses Matplotlib text by default. Pass `--use-tex` only when
an external LaTeX installation is available.

Standard models train for 5,000 epochs. Filtered models train for 3,500
epochs; at `n=1024`, batch size 512, this is 7,000 optimizer steps.
