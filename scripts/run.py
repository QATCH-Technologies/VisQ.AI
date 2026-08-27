"""Run the end-to-end VisQ.AI retraining and evaluation workflow.

This script provides a single operational entry point for processing training
data, optionally tuning model hyperparameters, retraining a final model on all
available data, and running selected evaluation workflows. It is intended to
give an ML engineer a straightforward way to produce and sanity-check a new
model without manually coordinating multiple training and evaluation scripts.

The training workflow consists of loading and preprocessing the selected
dataset, optionally running Group-Held-Out Optuna hyperparameter optimization,
and retraining the final model using the selected parameters. Evaluation is
performed directly through the consolidated evaluation modules rather than
through subprocesses or command-line argument round-tripping.

By default, the script runs the parity evaluation after training. The
zero-shot, LOGO, and learning-curve evaluations are opt-in because they may
require additional datasets or substantially more computation.

The script creates a fresh dated checkpoint directory for each invocation
unless an output directory is supplied programmatically. Data and model path
defaults are obtained from :mod:`visqai.constants` and :mod:`visqai.paths`.

The module is located under `scripts/` rather than `src/visqai/` because
it is an operational entry point rather than part of the installable library.
It is intended to be run from the repository root::

    python scripts/run.py

The VisQ.AI package must be installed so that its imports resolve regardless
of the current working directory.

This module replaces the former training and evaluation CLI entry points,
including `cli/train.py`, `cli/run_final_training.py`, and the standalone
evaluation CLIs. Model packaging is intentionally excluded from this workflow
because deployment packaging is a separate operational concern; see
`scripts/package.py`.

Examples:
    Run training with default parameters and the default parity evaluation::

        python scripts/run.py

    Run training with Optuna hyperparameter tuning::

        python scripts/run.py --data path/to/formulation_data.xlsx --trials 50

    Enable additional evaluation workflows::

        python scripts/run.py --eval-logo \
            --eval-zero-shot --zero-shot-data path/to/zero_shot.csv \
            --eval-learning-curve \
            --learning-curve-ibal-csv path/to/held_out.csv

    Train without running any evaluation::

        python scripts/run.py --no-eval

    Run an individual evaluation independently of this workflow::

        python -m visqai.eval.logo_eval --help

Functions:
    parse_args: Parse command-line arguments for the retraining and evaluation
        workflow.
    _train: Load data, optionally tune hyperparameters, and train the final
        model.
    _run_eval: Execute an evaluation phase while isolating evaluation failures
        from the completed training result.
    run: Execute the complete training and selected evaluation workflow.
    main: Parse command-line arguments, configure logging, and invoke the
        workflow.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import optuna
import torch

from visqai import constants, paths
from visqai.constants import DEFAULT_PARAMS
from visqai.eval import learning_curve_eval, logo_eval, parity_eval, zero_shot_eval
from visqai.logging_config import configure_logging
from visqai.training.data import load_and_preprocess
from visqai.training.run import train_final_model
from visqai.training.tuning import objective_cv

logger = logging.getLogger("VisqaiRun")


def parse_args(argv=None):
    """Parse command-line arguments for the retraining and evaluation workflow.

    Args:
        argv: Optional sequence of command-line arguments to parse. If
            `None`, arguments are read from `sys.argv`. Supplying an
            explicit sequence is useful for programmatic invocation and
            testing.

    Returns:
        argparse.Namespace: Parsed workflow configuration, including training
        data, Optuna trial count, training limits, random seed, evaluation
        selections, and evaluation dataset paths.
    """
    p = argparse.ArgumentParser(
        description="One-click retrain + evaluate: data-processing -> training/tuning -> eval."
    )
    p.add_argument(
        "--data",
        default=None,
        help="Path to the training CSV/XLSX. Defaults to the newest file in data/latest.",
    )
    p.add_argument(
        "--trials", type=int, default=0, help="Optuna trials (0 to skip tuning and use defaults)."
    )
    p.add_argument("--max-epochs", type=int, default=500, help="Max epochs for the final retrain.")
    p.add_argument(
        "--patience",
        type=int,
        default=80,
        help="Early-stopping patience (epochs) for the final retrain.",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")

    p.add_argument(
        "--eval-parity",
        dest="eval_parity",
        action="store_true",
        default=True,
        help="Run the parity eval (Ibalizumab context select + evaluate) against the freshly trained model. On by default.",
    )
    p.add_argument("--no-eval-parity", dest="eval_parity", action="store_false")
    p.add_argument(
        "--eval-zero-shot",
        dest="eval_zero_shot",
        action="store_true",
        default=False,
        help=(
            "Run the zero-shot eval (unusual/novel proteins) against the freshly trained "
            "model. Off by default -- needs --zero-shot-data (no curated copy ships with the repo)."
        ),
    )
    p.add_argument(
        "--zero-shot-data",
        dest="zero_shot_data",
        default=None,
        help="Zero-shot benchmark panel path, required if --eval-zero-shot is set.",
    )
    p.add_argument(
        "--eval-logo",
        dest="eval_logo",
        action="store_true",
        default=False,
        help="Run the LOGO scoreboard. Expensive -- retrains one model per held-out group. Off by default.",
    )
    p.add_argument(
        "--eval-learning-curve",
        dest="eval_learning_curve",
        action="store_true",
        default=False,
        help=(
            "Run the learning-curve convergence replay. Off by default -- needs "
            "--learning-curve-ibal-csv (no curated copy ships with the repo)."
        ),
    )
    p.add_argument(
        "--learning-curve-ibal-csv",
        dest="learning_curve_ibal_csv",
        default=None,
        help="Held-out evaluation CSV, required if --eval-learning-curve is set.",
    )
    p.add_argument(
        "--no-eval", action="store_true", default=False, help="Skip every eval phase (train only)."
    )
    return p.parse_args(argv)


def _train(data, out, trials, max_epochs, patience, seed):
    """Load training data, optionally tune hyperparameters, and train a model.

    The training data is loaded and preprocessed using
    :func:`visqai.training.data.load_and_preprocess`. When `trials` is
    greater than zero, Group-Held-Out Optuna optimization is performed using
    :func:`visqai.training.tuning.objective_cv`. The resulting best parameters
    are then used to retrain the final model on all available samples through
    :func:`visqai.training.run.train_final_model`.

    If no explicit data path is provided, the latest available data file is
    selected automatically. When a random seed is provided, NumPy and PyTorch
    random number generators are seeded before training. The training device
    is selected automatically based on CUDA availability.

    Args:
        data: Path to the training dataset. If `None`, the latest available
            data file is selected using :func:`visqai.paths.latest_data_file`.
        out: Directory in which preprocessing artifacts, checkpoints, and
            training outputs are stored.
        trials: Number of Optuna trials to execute. A value of `0` skips
            hyperparameter optimization and uses :data:`DEFAULT_PARAMS`.
        max_epochs: Maximum number of epochs permitted during final model
            training.
        patience: Number of epochs without improvement permitted by the final
            training procedure before early stopping.
        seed: Optional random seed used to initialize NumPy and PyTorch.

    Returns:
        dict: Training results returned by
        :func:`visqai.training.run.train_final_model`, including the best
        validation loss and number of epochs executed.
    """
    if data is None:
        data = paths.latest_data_file()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training data: {data}")
    samples, static_dim, physics_scaler, protected_indices = load_and_preprocess(data, save_dir=out)
    logger.info(
        f"Loaded {len(samples)} samples from {len(set(s['group'] for s in samples))} protein groups."
    )
    logger.info(f"Protecting {len(protected_indices)} load-bearing static features from masking.")

    best_params = dict(DEFAULT_PARAMS)
    if trials > 0:
        logger.info("Starting Group-Held-Out Optuna Optimization...")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(
            lambda t: objective_cv(
                t, samples, static_dim, device, physics_scaler, protected_indices
            ),
            n_trials=trials,
        )
        logger.info("--- Tuning Complete ---")
        logger.info(f"Best params: {study.best_params}")
        best_params = study.best_params
    else:
        logger.info("Skipping hyperparameter tuning since --trials=0. Using default params.")
        logger.info(f"Default params: {best_params}")

    logger.info("Retraining final model on ALL data...")
    result = train_final_model(
        samples,
        static_dim,
        physics_scaler,
        protected_indices,
        out_dir=out,
        params=best_params,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    logger.info(f"Model saved to {os.path.join(out, 'best_model.pth')}")
    logger.info(f"Best val loss: {result['best_loss']:.4f} after {result['epochs_run']} epochs")
    return result


def _run_eval(name: str, fn, **kwargs):
    """Run an evaluation phase without allowing it to abort training results.

    Evaluation failures are logged as warnings and converted to a `None`
    result. This isolates optional evaluation datasets and evaluation-specific
    failures from the successfully completed training workflow.

    Args:
        name: Human-readable name of the evaluation being executed. Used in
            logging messages and failure reporting.
        fn: Evaluation callable to invoke. It must accept the keyword
            arguments supplied through `kwargs`.
        **kwargs: Keyword arguments forwarded unchanged to the evaluation
            callable.

    Returns:
        Any: The value returned by `fn` when evaluation succeeds, or `None`
        when the evaluation raises an exception.
    """
    logger.info(f"\n{'='*70}\nEVAL: {name}\n{'='*70}")
    try:
        return fn(**kwargs)
    except Exception as e:
        logger.warning(f"  {name} eval failed -- {e}")
        return None


def run(
    data=None,
    out=None,
    trials=0,
    max_epochs=500,
    patience=80,
    seed=None,
    eval_parity=True,
    eval_zero_shot=False,
    zero_shot_data=None,
    eval_logo=False,
    eval_learning_curve=False,
    learning_curve_ibal_csv=None,
    no_eval=False,
):
    """Run model retraining followed by the selected evaluation workflows.

    A fresh dated checkpoint directory is created automatically when `out`
    is not supplied. The model is trained first, after which each enabled
    evaluation is executed against the newly trained checkpoint. Evaluation
    failures are isolated through :func:`_run_eval` so that a missing
    evaluation dataset or evaluation-specific error does not discard the
    training result.

    Setting `no_eval` disables all evaluation phases, regardless of the
    individual evaluation flags. Evaluation-specific datasets are required
    only for the corresponding evaluation and are skipped with a warning when
    they have not been supplied.

    Args:
        data: Path to the training dataset. If `None`, the latest available
            data file is selected automatically.
        out: Directory in which the checkpoint and evaluation outputs are
            stored. If `None`, a fresh dated checkpoint directory is created.
        trials: Number of Optuna hyperparameter-optimization trials. `0`
            skips tuning and uses the default training parameters.
        max_epochs: Maximum number of epochs for final model training.
        patience: Early-stopping patience, in epochs, for final training.
        seed: Optional random seed for NumPy and PyTorch.
        eval_parity: Whether to run the parity evaluation after training.
        eval_zero_shot: Whether to run the zero-shot evaluation.
        zero_shot_data: Path to the zero-shot evaluation dataset. Required when
            `eval_zero_shot` is enabled.
        eval_logo: Whether to run the leave-one-group-out (LOGO) evaluation.
        eval_learning_curve: Whether to run the learning-curve evaluation.
        learning_curve_ibal_csv: Path to the held-out CSV required by the
            learning-curve evaluation.
        no_eval: If `True`, disable every evaluation phase regardless of the
            individual evaluation flags.

    Returns:
        tuple: A two-element tuple containing:

            - `train_result`: The result returned by :func:`_train`.
            - `eval_results`: A dictionary mapping each enabled evaluation
              name to its result, or `None` when that evaluation failed or
              was otherwise skipped.
    """
    if out is None:
        out = paths.dated_run_dir(constants.CHECKPOINTS_DIR)
    if no_eval:
        eval_parity = eval_zero_shot = eval_logo = eval_learning_curve = False

    logger.info(f"Checkpoint directory: {out}")
    train_result = _train(data, out, trials, max_epochs, patience, seed)

    eval_results = {}
    if eval_parity:
        eval_results["parity"] = _run_eval(
            "parity", parity_eval.run, model_dir=out, out_dir=os.path.join(out, "eval_parity")
        )
    if eval_zero_shot:
        if zero_shot_data is None:
            logger.warning(
                "  zero_shot eval requested but --zero-shot-data wasn't given -- skipping."
            )
        else:
            eval_results["zero_shot"] = _run_eval(
                "zero_shot",
                zero_shot_eval.run,
                data_csv=zero_shot_data,
                model_dir=out,
                output_dir=os.path.join(out, "eval_zero_shot"),
            )
    if eval_logo:
        eval_results["logo"] = _run_eval(
            "logo", logo_eval.run, out_dir=os.path.join(out, "eval_logo")
        )
    if eval_learning_curve:
        if learning_curve_ibal_csv is None:
            logger.warning(
                "  learning_curve eval requested but --learning-curve-ibal-csv wasn't given "
                "-- skipping."
            )
        else:
            eval_results["learning_curve"] = _run_eval(
                "learning_curve",
                learning_curve_eval.run,
                ibal_csv=learning_curve_ibal_csv,
                model_dir=out,
                output_dir=os.path.join(out, "eval_learning_curve"),
            )

    logger.info(f"\n{'='*70}\nDONE\n{'='*70}")
    logger.info(f"Model: {os.path.join(out, 'best_model.pth')}")
    for name, result in eval_results.items():
        status = "ok" if result is not None else "FAILED (see warning above)"
        logger.info(f"  {name} eval: {status} -> {os.path.join(out, f'eval_{name}')}")

    return train_result, eval_results


def main(argv=None):
    """Run the command-line retraining and evaluation workflow.

    Parses the supplied command-line arguments, creates a fresh dated
    checkpoint directory, configures logging to that directory, and delegates
    execution to :func:`run`.

    Args:
        argv: Optional sequence of command-line arguments. If `None`,
            arguments are read from `sys.argv`.

    Returns:
        tuple: The training result and evaluation-result dictionary returned
        by :func:`run`.
    """
    args = parse_args(argv)
    out = paths.dated_run_dir(constants.CHECKPOINTS_DIR)
    configure_logging(log_dir=out)
    return run(out=out, **vars(args))


if __name__ == "__main__":
    main()
