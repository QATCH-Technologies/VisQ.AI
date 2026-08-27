# VisQ.AI

Physics-informed viscosity prediction for protein/antibody formulations. A Conditional Neural
Process (CNP) predicts a formulation's viscosity across shear rates from its physicochemical
properties, combining a physics-informed zero-shot prior with an optional few-shot correction
learned from a handful of real measurements on similar formulations.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, test conventions, and code
conventions used throughout this repo.

## Setup

Requires Python >= 3.10.

``
pip install -e .[dev]
``

Formulation data and trained checkpoints are **not** stored in this repo -- they live on the
team's shared Dropbox folder (`QATCH Team Folder/Formulations ML`). Paths resolve automatically
for anyone with that Dropbox mounted (see `src/visqai/constants.py`); override with the
`VISQAI_DATA_ROOT` / `VISQAI_MODELS_ROOT` environment variables if your Dropbox is mounted
somewhere nonstandard, or if you're pointing at a different data source entirely.

## Quick start

One command retrains the model on the newest data and runs a fast sanity-check eval:

``
python scripts/run.py
``

This is the entrypoint aimed at a novice ML engineer -- it chains data-processing ->
(optional Optuna tuning) -> training -> eval, in one process, and writes everything to a fresh
`<checkpoints>/<date>/<time>/` directory. Checkpoints are never given a hand-picked name, so
runs never collide and there's nothing to remember to invent. Run `python scripts/run.py --help`
for tuning trials, which evals to run, and the (opt-in, data-hungry) LOGO/zero-shot/learning-curve
evals.

Package the most recent checkpoint into a signed `.visq` deployment artifact:

``
python scripts/package.py
``

Every eval also runs standalone, against an existing checkpoint (defaults to the most recently
produced one if `--model_dir` is omitted):

``
python -m visqai.eval.parity_eval --help
python -m visqai.eval.logo_eval --help
python -m visqai.eval.zero_shot_eval --help
python -m visqai.eval.learning_curve_eval --help
``

## Repo layout

``
src/visqai/
  constants.py, paths.py, validation.py, logging_config.py   # shared infrastructure
  features/      # raw row -> engineered feature frame (physics priors, categorical/charge featurization)
  training/      # data loading, the CrossSampleCNP training loop, Optuna tuning
  models/        # the CrossSampleCNP architecture
  inference/     # ViscosityPredictorCNP -- the served prediction API
  eval/          # four self-contained evals: parity, LOGO, zero-shot, learning-curve
  packaging/     # signed .visq deployment packages
scripts/
  run.py         # one-click retrain + evaluate
  package.py     # build a signed .visq package
tests/
  unit/, integration/
``

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow and design rationale.

## Tests

``
pytest
``

See [CONTRIBUTING.md](CONTRIBUTING.md) for coverage tooling and test-writing conventions.

## License

GPLv3 -- see [LICENSE](LICENSE).
