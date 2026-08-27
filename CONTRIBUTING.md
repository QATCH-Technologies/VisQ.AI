# Contributing

## Setup

``
pip install -e .[dev]
``

This installs the package editable (so `import visqai` resolves from anywhere, including
`scripts/`) plus `pytest` and `pytest-cov`.

Data and model checkpoints live on the team's shared Dropbox folder, not in the repo -- see
[README.md](README.md#setup). Most tests don't need it (they build small synthetic fixtures
in-process); a few integration/regression tests skip gracefully when a real file isn't present
(look for `pytest.mark.skipif` and a clear `reason=`).

## Running tests

``
pytest
``

With coverage for a specific module (the pattern used throughout this repo's test-writing
sessions):

``
pytest --cov=visqai.training.loop --cov-report=term-missing tests/unit/test_training_loop.py
``

Coverage target: **new code should be well-covered**, but "well-covered" means the real
boundary/branch cases (the ones a caller could actually hit), not literally every line. A
defensive fallback for a condition that can't happen with valid input (e.g. a NaN-after-fit
warning branch) is fine to leave uncovered rather than mocking internals just to trip it.

### Testing stochastic / deep-learning code

`training/loop.py`, `training/run.py`, and anything touching `CrossSampleCNP` involve real
randomness (dropout, AdamW, random context/target splits). Don't assert exact loss values --
assert:
- shapes and dtypes are correct,
- outputs are finite (`np.isfinite` / `torch.isfinite`),
- model parameters actually changed after a training step (gradient really flowed -- snapshot
  params before/after and compare),
- the right code path ran (e.g. cross-check `validate_zero_shot` against a manual
  `decode_from_memory(zeros(...), ...)` call, rather than trusting the number it returns).

Build synthetic `samples` with a small local helper (`_make_samples(groups_and_counts, ...)` --
see `tests/unit/test_training_loop.py`) rather than a real dataset; a tiny `CrossSampleCNP`
(`hidden_dim=8-16`, `latent_dim=4-8`) keeps these tests fast (the whole suite runs in well under
a minute).

## Code conventions

These are the conventions this codebase has actually converged on -- follow the existing pattern
in the file you're touching over anything below if they conflict.

**Comments explain WHY, not WHAT.** A comment restating what the next line does is noise; a
comment explaining a non-obvious constraint, a past bug it fixes, or why a value was calibrated
to a specific number is what makes this codebase navigable. Most modules carry a module
docstring with the real history (a bug that was found, an experiment that motivated a design) --
read it before changing behavior nearby.

**Input validation lives at the boundary, not in hot paths.** `src/visqai/validation.py` has the
shared checks (`require_dataframe`, `require_path_exists`, `require_positive`, ...). Use them in
functions an external caller (a notebook, `scripts/run.py`, another script) can call directly with
bad input. Do **not** add them to `training/loop.py`'s `train_epoch`/`validate*` or
`models/cnp.py`'s `forward*`/`encode_memory`/`decode_from_memory` -- those run many times per
second on data that was already validated once, upstream; checking there is pure per-call
overhead with no caller it could ever help. See `validation.py`'s own module docstring for the
full reasoning.

**Constants: shared and simple vs. local and justified.** `src/visqai/constants.py` holds values
used across two or more modules that don't carry algorithm-specific justification (the shear-rate
schema, the plot palette, `DEFAULT_PARAMS`, the Dropbox storage roots). An empirically-calibrated
threshold with a real derivation (`CONTEXT_GATE_TOLERANCE`, `TAU2_CONC`, `PRIOR_TABLE`, ...) stays
local, next to the comment that justifies its exact value and the code that depends on it. Moving
it to `constants.py` would strip that context for no real sharing benefit. If you're adding a
constant, ask: is this the same value another module also needs, or is it "0.01, because a real
run measured X" -- the answer decides where it lives.

**Eval modules are self-contained.** Each `eval/*_eval.py` (parity, LOGO, zero-shot,
learning-curve) exposes `run(**kwargs)` (the real logic, importable and callable directly, e.g.
from `scripts/run.py`) and `main(argv=None)` (a thin argparse wrapper: `parse_args` -> resolve any
lazy path defaults -> `configure_logging` -> `run`). Only genuinely cross-eval code stays shared
(`eval/constants.py`... actually `visqai.constants`, `eval/metrics.py`, `eval/style.py`); anything
used by exactly one eval lives inside that eval's own file, even if that makes the file long. When
adding a fifth eval, follow this shape rather than reaching into another eval's internals.

**No hand-picked checkpoint/production names.** Every training run and every packaged model gets
an auto-generated `<root>/<date>/<time>/` directory (`visqai.paths.dated_run_dir`) and, for
packages, an auto-generated filename. Don't add a CLI flag that lets a caller invent a custom
name/location for these -- the whole point is that they never collide and there's nothing to
remember. `visqai.paths.latest_checkpoint_dir` is how code finds "the most recent one" back.

**No backwards-compatibility shims.** If a function/module moves, update every call site;
don't leave a re-export or an old name pointing at the new one "just in case." This repo went
through several structural consolidations (physics+features merged into `features/`, `cli/` and
`analysis/` removed, `eval/` reduced from 18 files to 4 self-contained ones, a single
`constants.py` replacing five scattered ones) specifically to avoid accumulating that kind of
debt -- don't reintroduce it.

## Adding a new eval

1. Create `src/visqai/eval/<name>_eval.py` following the shape above (`run()` + `main()` +
   argparse `parse_args()`).
2. Pull in only what's genuinely shared (`visqai.constants`, `eval.metrics`, `eval.style`,
   `visqai.paths`); everything eval-specific lives in the new file.
3. Wire it into `scripts/run.py` if it should be reachable from the one-click flow (see how
   `--eval-logo`/`--eval-zero-shot` are wired there, including the "off by default if it needs a
   data argument this repo doesn't ship a curated copy of" pattern).
4. Add a `tests/unit/test_<name>_eval.py` if there's real logic to test beyond argparse wiring.

## CI

`.github/workflows/` has three workflows (`python-app.yml`, `pylint.yml`, `codeql.yml`).
**`python-app.yml` currently does not match this repo's layout** -- it runs
`python -m unittest discover` from a `working-directory: visQAI` that no longer exists (the repo
uses `src/`-layout + `pytest`, not a top-level `visQAI/` package). Until it's updated, treat
`pytest` locally as the source of truth, not a green check on that workflow.

## Pull requests

Branch from `main`, keep PRs scoped to one logical change, and make sure `pytest` passes locally
before opening. There's no enforced review/branch-protection convention recorded in this repo
yet -- use ordinary judgment (a second look on anything touching `models/cnp.py`,
`training/loop.py`, or `inference/predictor.py`, since those are the highest-blast-radius files).
