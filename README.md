<p align="center">
  <img src="assets/visqai-logo.png" alt="VisQ.AI logo" width="240">
</p>

# VisQ.AI

Physics-informed viscosity prediction for protein and antibody formulations. VisQ.AI uses a
Conditional Neural Process (CNP) architecture to predict formulation viscosity across shear
rates from fundamental physicochemical properties, combining a physics-informed zero-shot
prior with an optional few-shot correction learned from a handful of real measurements.

[![Build](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/build.yml/badge.svg)](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/build.yml)
[![CodeQL](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/codeql.yml/badge.svg)](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/codeql.yml)
[![Style](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/style.yml/badge.svg)](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/style.yml)
[![Test](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/test.yml/badge.svg)](https://github.com/QATCH-Technologies/VisQ.AI/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Overview

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - data flow, design rationale, and system components.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - development workflow, testing standards, code conventions.

## Installation

Requires Python 3.10+.

```bash
pip install -e .[dev]
```

## Data and Model Paths

By default, the package resolves training data and model checkpoints from a shared cloud directory
structure.

| Mode | Behavior |
| --- | --- |
| Standard | Paths resolve automatically when the shared Dropbox folder is mounted locally. |
| Custom | Override with the `VISQAI_DATA_ROOT` / `VISQAI_MODELS_ROOT` environment variables. |

## Quick Start

### Training and evaluation

Run the full pipeline including data processing, hyperparameter tuning, model
training, and sanity-check evaluation from a single entrypoint:

```bash
python scripts/run.py
```

Outputs are written to a timestamped directory under the configured checkpoints folder
(`<checkpoints>/<date>/<time>/`). See all options, including tuning trials and evaluation
suites:

```bash
python scripts/run.py --help
```

### Deployment packaging

Package the most recent model checkpoint into a signed deployment artifact (`.visq`):

```bash
python scripts/package.py
```

### Standalone evaluation

Each evaluation module can also run independently against an existing checkpoint (defaults to
the most recent one if `--model_dir` is omitted):

```bash
python -m visqai.eval.parity_eval --help
python -m visqai.eval.logo_eval --help
python -m visqai.eval.zero_shot_eval --help
python -m visqai.eval.learning_curve_eval --help
```

## Repository Layout

```
src/visqai/
  ├── constants.py, paths.py, validation.py, logging_config.py  # Shared infrastructure
  ├── features/      # Feature engineering and physics priors
  ├── training/       # Data loaders, training loops, and Optuna tuning
  ├── models/          # Model architecture (s)
  ├── inference/       # Prediction and serving API
  ├── eval/             # Evaluation suites
  └── packaging/     # Deployment packaging utilities
scripts/
  ├── run.py           # End-to-end training and evaluation pipeline
  └── package.py    # Artifact packaging utility
tests/
  ├── unit/             # Unit test suites
  └── integration/    # Integration test suites
```

## Testing

```bash
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coverage tooling and test-writing guidelines.

## License

Licensed under the GNU General Public License v3.0 (GPLv3). See [LICENSE](LICENSE) for details.
