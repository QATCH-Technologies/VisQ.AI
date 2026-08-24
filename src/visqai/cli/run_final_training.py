"""
run_final_training.py
======================
Entry point for the final Optuna tuning + retrain run on
data/raw/formulation_data_08242026.csv, the Whole-Charge-only feature set.

Not invoked automatically. Run manually when ready:

    visqai-train-final
    python -m visqai.cli.run_final_training

Any extra CLI args (e.g. --trials 100) override the defaults below, since
they're appended after them and argparse keeps the last value it sees.
"""

from __future__ import annotations

import sys

from visqai.cli.train import main as train_main

DATA = "data/raw/formulation_data_08242026.csv"
OUT_DIR = "models/experiments/final_whole_charge_run"
TRIALS = "50"  # Optuna hyperparameter-search trials
MAX_EPOCHS = "500"
PATIENCE = "80"
SEED = "0"


def main(argv=None):
    defaults = [
        "--data", DATA,
        "--out", OUT_DIR,
        "--trials", TRIALS,
        "--max-epochs", MAX_EPOCHS,
        "--patience", PATIENCE,
        "--seed", SEED,
    ]
    train_main(defaults + list(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    main()
