import os
import sys
import json
import math
import logging
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import torch

# Add the project root to sys.path so we can import from 'modules'
# Use insert(0) to ensure we load from this directory, overriding any PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.training import train_model
from utils import load_data, create_model, setup_device, create_save_path, load_checkpoint, load_params
import numpy as np
from modules.ladder_diagnostics import LadderEvaluator, level_excess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main(args):
    """Train a denoising transformer on controlled setting data.

    Handles data loading/generation, model creation, checkpoint resumption,
    LR scheduling, mixed-precision training, and periodic checkpointing.

    Args:
        args (argparse.Namespace): CLI arguments (see ``argparse`` block at
            the bottom of this script).
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # Set device
    device = setup_device()
    torch.set_float32_matmul_precision('high')

    # Load data
    vocab_size, sequences, seq_len, k, rho, _ = load_data(args.data_path, device)
    left_idx = args.pick_i_for_training * args.reduced_length
    right_idx = args.pick_i_for_training * args.reduced_length + args.reduced_length
    if left_idx < 0 or right_idx > len(sequences):
        raise ValueError(
            f"Requested slice [{left_idx}:{right_idx}] is out of bounds for sequence length {len(sequences)}"
        )
    full_sequences = sequences[left_idx:right_idx]

    logger.info("Data loaded from %s", args.data_path)
    logger.info("Reduced Data shape: %s", full_sequences.shape)
    uniq, inv, cnt = torch.unique(full_sequences, dim=0, return_inverse=True, return_counts=True)
    n_distinct = len(uniq)
    dup_pairs = int((cnt * (cnt - 1) // 2).sum())
    logger.info("train block: %d sequences, %d distinct (%d duplicate pairs, %.3f%% non-unique)",
                len(full_sequences), n_distinct, dup_pairs,
                100 * (1 - n_distinct / len(full_sequences)))
    logger.info("First sequence (check seed): %s", full_sequences[0])

    # Prepare transformer model
    model_args = {'hidden_size_encoder': args.hidden_size_encoder, 'nhead': args.nhead, 'num_layers': args.num_layers, 'model_channels': args.model_channels}
    transformer_model, _, actual_hidden_size = create_model(model_args, vocab_size, seq_len, device)
    if device.type == "cuda":
        transformer_model = torch.compile(transformer_model)
    logger.info("Transformer model created with parameters: %s", model_args)
    logger.info("Total model parameters: %d", sum(p.numel() for p in transformer_model.parameters()))

    # Load checkpoint if resuming.
    epoch_start = 0
    counter_start = 0
    if args.resume_checkpoint_path:
        logger.info("Resuming from checkpoint: %s", args.resume_checkpoint_path)
        checkpoint_path = args.resume_checkpoint_path
        # Validate model parameters with the checkpoint.
        params_path = os.path.join(checkpoint_path, 'full_params.json')
        if os.path.exists(params_path):
            model_params = load_params(params_path)
            assert model_params['hidden_size_encoder'] == args.hidden_size_encoder, (
                f"hidden_size_encoder {args.hidden_size_encoder} inconsistent with checkpoint "
                f"{model_params['hidden_size_encoder']}"
            )
            assert model_params['nhead'] == args.nhead, (
                f"nhead {args.nhead} inconsistent with checkpoint {model_params['nhead']}"
            )
            assert model_params['num_layers'] == args.num_layers, (
                f"num_layers {args.num_layers} inconsistent with checkpoint {model_params['num_layers']}"
            )
            assert model_params['model_channels'] == args.model_channels, (
                f"model_channels {args.model_channels} inconsistent with checkpoint {model_params['model_channels']}"
            )
            assert model_params['data_path'] == args.data_path, (
                f"data_path {args.data_path} inconsistent with checkpoint {model_params['data_path']}"
            )
        else:
            raise ValueError("full_params.json not found in the checkpoint folder.")

        epoch_start, counter_start, _ = load_checkpoint(
            model = transformer_model, optimizer=None, scheduler=None, checkpoint_dir=args.resume_checkpoint_path, training = True, device = device)
        logger.info("Checkpoint loaded. Starting from epoch %d, counter %d", epoch_start, counter_start)
        # Note: optimizer and scheduler states will be loaded later after they are created.

    # Set diffusion parameters.
    alpha_bars = alpha_bars_schedule(args.t_final, device=device, s=args.s, schedule='linear')
    alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)
    logger.info("Diffusion schedule created with t_final: %d, s: %f", args.t_final, args.s)

    # Prepare data embedding.
    full_embedded_sequences = torch.nn.functional.one_hot(full_sequences.long(), vocab_size).float()
    
    x0s = full_embedded_sequences
    steps_per_epoch = -(-x0s.shape[0] // args.batch_size)  # ceil(N / batch_size)
    total_steps = args.n_iter * steps_per_epoch
    logger.info("steps_per_epoch=%d  total_steps=%d", steps_per_epoch, total_steps)
    data_gen_args = {
        'x0s': x0s,
        't_final': args.t_final,
        'alpha_bars': alpha_bars,
        'alphas': alphas,
        'device': device,
    }
    ladder_evaluator = None
    if args.ladder_live:
        f_list = ([int(x) for x in args.bp_factorized_layers.split(',')]
                  if args.bp_factorized_layers else list(range(k + 1)))
        ladder_evaluator = LadderEvaluator(
            x0_oh_eval=x0s, t_final=args.t_final, alpha_bars=alpha_bars, alphas=alphas,
            rho=rho, k=k, vocab_size=vocab_size, factorized_layers_list=f_list,
            device=device, batch_size=args.ladder_batch_size,
            reweight=args.reweighting, rescale_loss=args.rescale_loss, seed=args.seed,
        )
        logger.info("Live ladder: ell=%d f_list=%s reweight=%s rescale_loss=%s N=%d",
                    k, f_list, args.reweighting, args.rescale_loss, len(x0s))
    # Set up optimizer.
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=args.lr)

    # Learning rate scheduler.
    if args.lr_schedule == 'default':
        WARMUP_STEPS = 50  # or pick a value meaningful in step-terms, e.g. 500
        def lr_lambda(step):
            if step <= WARMUP_STEPS:
                return 1.0 - (step / WARMUP_STEPS) * 0.5
            else:
                T = total_steps - WARMUP_STEPS
                return 0.1 + (0.5 - 0.1) * 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / T))
    elif args.lr_schedule == 'alternative':
        # 500-epoch warmup from 0 to 1.0, then cosine to 0.1.
        def lr_lambda(epoch):
            WARMUP_EPOCHS = 500
            n_iter = args.n_iter
            if epoch < WARMUP_EPOCHS:
                return float(epoch) / float(max(1, WARMUP_EPOCHS))
            else:
                T = n_iter - WARMUP_EPOCHS
                progress = float(epoch - WARMUP_EPOCHS) / float(max(1, T))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return 0.1 + (1.0 - 0.1) * cosine_decay
    elif args.lr_schedule == 'constant':
        # Fixed LR throughout, no warmup, no decay -- matches the paper's protocol.
        def lr_lambda(epoch):
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info("Optimizer and scheduler created with learning rate: %f", args.lr)

    # If resuming, load optimizer and scheduler states.
    if args.resume_checkpoint_path:
        _, _, _ = load_checkpoint(model = transformer_model, optimizer = optimizer, scheduler = scheduler, checkpoint_dir = args.resume_checkpoint_path, training = True, device = device)
    logger.info("Starting learning rate: %f", optimizer.param_groups[0]['lr'])
    logger.info("Scheduler state: %s", scheduler.state_dict())

    # Determine loss name.
    if args.use_compute_mse_loss and args.use_cross_entropy_loss:
        raise ValueError("Cannot use both MSE and cross-entropy loss at the same time.")
    elif args.use_compute_mse_loss:
        loss_name = "mse_loss"
        logger.info("Using MSE loss.")
    elif args.use_cross_entropy_loss:
        loss_name = "cross_entropy_loss"
        logger.info("Using cross-entropy loss.")
    else:
        raise ValueError("One loss function must be specified.")

    # Create save path.
    name_saving = args.model_name + "_" + f"pick_{args.pick_i_for_training}_for_training"
    save_path = create_save_path(
        args.data_path,
        args.n_iter,
        args.reduced_length,
        loss_name,
        args.num_layers,
        actual_hidden_size,
        False,
        args.rescale_loss,
        args.seed,
        model_type=name_saving,
        add_flag=args.add_flag,
        results_dir=args.results_dir
    )

    # Save parameters.
    with open(os.path.join(save_path, 'full_params.json'), 'w') as f:
        json.dump(vars(args), f)

    logger.info("Starting training from epoch: %d", epoch_start)
    logger.info("Starting training from counter: %d", counter_start)

    # Train the model.
    _, training_history = train_model(
        model=transformer_model,
        data_gen_args=data_gen_args,
        optimizer=optimizer,
        n_epochs=args.n_iter,
        batch_size=args.batch_size,
        use_compute_mse_loss=args.use_compute_mse_loss,
        use_cross_entropy_loss=args.use_cross_entropy_loss,
        scheduler=scheduler,
        checkpointing_period_epochs=(None if args.checkpointing_period_epochs == 0 else args.checkpointing_period_epochs),
        checkpointing_period_steps=(None if args.checkpointing_period_steps == 0 else args.checkpointing_period_steps),
        model_dir=save_path,
        checkpoint_id='test_model_script',
        save_final_model=args.save_final_model,
        epoch_start=epoch_start,
        count_fun=counter_start,
        reweighting=args.reweighting,
        rescale_loss=args.rescale_loss,
        ladder_eval_fn=(ladder_evaluator.evaluate if ladder_evaluator else None),
        ladder_log_path=(os.path.join(save_path, 'ladder_live.json') if ladder_evaluator else None),
    
    )

    if ladder_evaluator is not None and training_history['ladder']:
        rows = training_history['ladder']
        f_list = ladder_evaluator.factorized_layers_list
        ladder_dict = {f: [r[f'bp_{f}'] for r in rows] for f in f_list}
        R = level_excess(ladder_dict, ell=max(f_list), oracle_ladder=ladder_evaluator.bp_oracle)     # asserts the telescoping
        tag_key = 'step' if 'step' in rows[0] else 'epoch'
        np.savez(os.path.join(save_path, 'ladder_R.npz'),
                 **{tag_key: np.array([r[tag_key] for r in rows])}, R=R,
                 **{f'bp_{f}': np.array(ladder_dict[f]) for f in f_list})
        logger.info("Ladder saved: R %s over %d checkpoints", R.shape, len(rows))

if __name__ == '__main__':
    parser = ArgumentParser(description="Diffusion Transformer Training Script")

    # Experiment parameters.
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--resume-checkpoint-path', type=str, default=None, help='Path to resume training from a checkpoint.')
    parser.add_argument('--reweighting', action='store_true', help='Enable reweighting.')
    parser.add_argument('--model-name', type=str, default='transformer', help='Model name for saving directory.')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to save results to. If None, uses default relative path.')

    # Data parameters.
    parser.add_argument('--data-path', type=str, default="../data/labeled_data_restrictedfixed_6_4_1.0_4_0.npy", help='Path to the data file.')
    parser.add_argument('--reduced-length', type=int, default=1000, help='Number of sequences to use.')
    parser.add_argument('--pick-i-for-training', type=int, default=0, help='Pick the i-th chunk of reduced-length size for training.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')

    # Model parameters.
    parser.add_argument('--hidden-size-encoder', type=int, default=512, help='Hidden size for the encoder.')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads in multihead attention.')
    parser.add_argument('--num-layers', type=int, default=8, help='Number of transformer encoder layers.')
    parser.add_argument('--model-channels', type=int, default=32, help='Model channels for timestep embedding.')

    # Diffusion parameters.
    parser.add_argument('--t-final', type=int, default=500, help='Final time step.')
    parser.add_argument('--s', type=float, default=1e-4, help='Scale parameter for alpha bar schedule.')

    # Training parameters.
    parser.add_argument('--n-iter', type=int, default=20000, help='Number of training iterations.')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate.')
    parser.add_argument('--use-compute-mse-loss', action='store_true', help='Use MSE loss.')
    parser.add_argument('--use-cross-entropy-loss', action='store_true', help='Use cross entropy loss.')
    parser.add_argument('--checkpointing-period-epochs', type=int, default=0,
                    help='Number of log-spaced checkpoints to save, keyed on epoch. '
                         '0 disables. Mutually exclusive with --checkpointing-period-steps.')
    parser.add_argument('--checkpointing-period-steps', type=int, default=25,
                    help='Number of log-spaced checkpoints to save, keyed on global gradient step. '
                         '0 disables. Mutually exclusive with --checkpointing-period-epochs.')
    parser.add_argument('--save-final-model', action='store_true', help='Save the final model.')
    parser.add_argument('--rescale-loss', action='store_true', help='Rescale loss.')
    parser.add_argument('--add-flag', type=str, default=None, help='String to append to output directory name.')
    parser.add_argument('--lr-schedule', type=str, default='constant', choices=['default', 'alternative', 'constant'],
                        help='LR schedule: "default" (1.0→0.5 in 50ep, cosine→0.1) or "alternative" (500ep warmup, cosine→0.1).')
    parser.add_argument('--ladder-live', action='store_true',
                    help='Log bp_k^emp / R_d on the training block at each checkpoint.')
    parser.add_argument('--bp-factorized-layers', type=str, default=None,
                        help='Comma-separated k values; default is the full ladder 0..ell.')
    parser.add_argument('--ladder-batch-size', type=int, default=512,
                        help='Batch size for ladder eval (no backward pass -> can exceed training batch size).')
    args = parser.parse_args()
    if args.checkpointing_period_epochs != 0 and args.checkpointing_period_steps != 0:
        parser.error("Specify only one of --checkpointing-period-epochs / --checkpointing-period-steps.")
    if args.ladder_live and args.checkpointing_period_epochs == 0 and args.checkpointing_period_steps == 0:
        parser.error("--ladder-live needs a nonzero checkpointing period to hang its cadence on.")
    main(args)