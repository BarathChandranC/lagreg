# Biased Generalization in Diffusion Models

Code for the numerical experiments in the paper [Biased Generalization in Diffusion Models](https://arxiv.org/abs/2603.03469) by J. Garnier-Brun, L. Biggio, D. Beltrame, M. Mézard, and L. Saglietti. Accepted as a Spotlight at ICML 2026.

Hierarchical-data (controlled setting) experiments live in `scripts/` (experiment files) and `modules/` (core logic).

CelebA experiments are in the `wdmdm/` submodule (fork of [Bonnaire et al.](https://github.com/tbonnair/Why-Diffusion-Models-Don-t-Memorize)).

Unless otherwise provided, all default arguments and hardcoded constants in the scripts (such as seed = 0) are configured to reproduce the experimental setup detailed in the paper.

## Setup

The primary setup is uv. On Linux x86-64 the lock selects the CUDA 12.1
PyTorch wheels; on macOS it selects native CPU wheels.

```bash
cd /path/to/biased-generalization
git submodule update --init --recursive
uv sync --frozen
```

Conda remains fully supported:

```bash
bash setup_env.sh --gpu  # PyTorch 2.5.1, torchvision 0.20.1, CUDA 12.1
bash setup_env.sh --cpu
```

Both setup paths install the `modules` package.

## Data generation

Controlled setting data is generated once and reused by all experiments:

```bash
python modules/gen_filtered_hierarchical_data_wforbidden.py \
    --n-trials 1000000 --workers 8 \
    --output data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy
# saves data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy  (q=6, l=4, σ=1, q_eff=4)
```

By default, the script generates the dataset used for all hierarchical-data experiments in the paper.

## Training

### Transformer for controlled setting

Run from `scripts/`. Three (size, iteration) settings cover the paper figures:

```bash
# n=5k, 20k iterations
python train_transformer.py --seed 0 --reduced-length 5000 --pick-i-for-training 0 \
    --batch-size 512 --n-iter 20000 --use-cross-entropy-loss

# n=12k, 30k iterations
python train_transformer.py --seed 0 --reduced-length 12000 --pick-i-for-training 0 \
    --batch-size 512 --n-iter 30000 --use-cross-entropy-loss

# n=70k, 35k iterations
python train_transformer.py --seed 0 --reduced-length 70000 --pick-i-for-training 0 \
    --batch-size 512 --n-iter 35000 --use-cross-entropy-loss
```

The tracked `reproduction/controlled_manifest.csv` lists
the 45 balanced split/seed runs and the 12 additional 12k ensemble seeds. Use
`reproduction/run_controlled_training.sh`.

### CelebA U-Net

See `wdmdm/README.md`. In brief, from `wdmdm/Experiments/src/Training`:

```bash
python run_Unet.py -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1 \
                   -se 0 --epochs 5000 --save-root ../../Saves_new
```

Train 15 independent splits (`-i 0 ... 14`) for the sample-split analysis.

## Figure reproduction

Run the portable launchers below from the repository root. Model-dependent
experiments require explicit checkpoint paths; there is no fresh-clone default
that can safely guess which trained run to use. See `reproduction/README.md`
for complete commands, output layouts, and checkpoint-profile examples.

### Hierarchical-data figures

| Figure | Script | Required launcher/action |
|--------|--------|-----------------|
| Fig. 1(b), Fig. 6 | `sequential_learning.py` | `bash reproduction/run_controlled_evaluation.sh sequential --models-root <root> --subpath1 <model-1> --subpath2 <model-2>` |
| Fig. 3(b) | `score_divergence_along_t.py` | `bash reproduction/run_controlled_evaluation.sh score-time --base-path '<root>/results_transformer_pick_{}_for_training' --which-epoch1 <5k-min> --which-epoch2 <12k-min> --which-epoch3 <70k-min>` |
| Fig. 4(a), App. 10–11 (left) | `nn_divergence.py` | `bash reproduction/run_controlled_evaluation.sh nn --paths-gen <comma-separated-model-paths>` |
| Fig. 4(b) | `loss_decomposition.py` | `bash reproduction/run_controlled_evaluation.sh loss --base-dir <split-0-root> --seeds 0,1,...,14 --subpath <model-subpath>` |
| Fig. 5 | `uturn_overlap.py` | `bash reproduction/run_controlled_evaluation.sh uturn-main --paths-gen <comma-separated-model-paths> --which-epoch <epoch>` |
| App. 10–11 (right) | `uturn_overlap.py` | `bash reproduction/run_controlled_evaluation.sh uturn-appendix --paths-gen <comma-separated-model-paths> --which-epoch <epoch>` |
| App. Fig. 12 | `uturn_overlap_random.py` | `bash reproduction/run_controlled_evaluation.sh uturn-random --paths-gen <model-path> --which-epoch <epoch>` |

The launchers require `REPRO_ROOT` and write each action to a separate
subdirectory. Direct scripts also accept `--output-root`; their docstrings
document the underlying CLI.

### CelebA figures

| Figure | Script (in `wdmdm/Experiments/src/`) | Command |
|--------|---------------------------------------|---------|
| Fig. 1(a) left | `Generation/loss_compute.py` | `python loss_compute.py -D CelebA -n 1024 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 --eval_N 50 --model_root ../../Saves_new --out_dire ../../Saves_new` |
| Fig. 1(a) right, App. 8 | `Generation/sample_split_inference.py` | `python sample_split_inference.py --scan_all_pairs --indices 0-14 -n 1024 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -Ns 512 --model-root ../../Saves_new --out-dir ../../Saves_new` |
| Fig. 2, App. 9 | `Generation/compare_scores.py` | `for t in 50 100 150 200; do python compare_scores.py -D CelebA -n 1024 -is 0 -ie 14 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -Ns 1000 -t "$t" --model_root ../../Saves_new --out_dire ../../Saves_new; done` |
| App. Fig. 7 | `Generation/celeba_multiscale_complexity_plot.py` | `python celeba_multiscale_complexity_plot.py --result-root ../../Saves_new/<fresh_root> --methods PCA,Wavelet --levels 0,1,2 --time 100 --out-stem celeba_wavelet_pca_complexity` |

The complete filtering, 3,500-epoch training, within-level selection, and
cross-level evaluation workflow is documented in `wdmdm/README.md`.
Portable environment-variable-driven launchers for the controlled and CelebA
training/evaluation stages are documented in `reproduction/README.md`.

## Citation

```bibtex
@inproceedings{
garnier-brun2026biased,
title={Biased Generalization in Diffusion Models},
author={Jerome Garnier-Brun and Luca Biggio and Davide Beltrame and Marc Mezard and Luca Saglietti},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=IqTvyp40O6}
}
```
