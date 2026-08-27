# Contributing to VisQ.AI

## Development Setup

Install the package in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

This enables editable imports (e.g. `import visqai`) across the repository, including
`scripts/`, and installs testing tools (`pytest`, `pytest-cov`) and `flake8`.

**Data and models:** Datasets and model checkpoints are stored externally on the team's shared
cloud storage, not in version control. See [README.md](README.md#data-and-model-paths) for path configuration. 
Most tests use small, in-process synthetic fixtures; integration tests that require external files skip gracefully
when those files aren't present.

## Testing Standards

Run the full test suite:

```bash
pytest
```

Generate a coverage report for a specific module:

```bash
pytest --cov=visqai.training.loop --cov-report=term-missing tests/unit/test_training_loop.py
```

### Coverage philosophy

New code must be adequately covered, focusing on realistic boundary and branch cases a caller
might actually hit. Defensive fallbacks for conditions that can't occur (e.g. a post-fit NaN
warning under valid input) don't need artificial mocking purely to chase 100% line coverage.

### Testing deep learning and stochastic components

For modules involving randomness (`training/loop.py`, `models/cnp.py` with dropout or AdamW),
avoid asserting exact loss values. Instead assert:

- Tensor shapes and dtypes are correct.
- Model outputs are finite (`np.isfinite` / `torch.isfinite`).
- Model parameters update after a training step (compare parameter snapshots before/after to
  confirm gradient flow).
- The correct code paths execute - call `validate`-style functions directly and check their
  behavior rather than trusting an aggregate return value alone.

For synthetic data, use local helper functions (e.g. `_make_samples`) to build minimal inputs,
and use small model dimensions (`hidden_dim=8–16`, `latent_dim=4–8`) to keep tests fast.

## Code Conventions

Follow the established pattern in the file being modified. Where existing code conflicts with
the guidance below, defer to that file's own convention.

1. **Contextual comments.** Comments explain the *why*, not the *what* - don't restate what the
   code already says. Use them for non-obvious constraints, historical bug fixes, or the
   derivation behind a calibrated value. Module docstrings carry a module's historical context.

2. **Boundary input validation.** Validate at the external boundaries of the system, not inside
   performance-critical hot paths. Use the shared checks in
   [`src/visqai/validation.py`](src/visqai/validation.py) (`require_dataframe`,
   `require_positive`, ...) for externally-facing functions. Do not add validation to internal
   hot paths like `training/loop.py` or `models/cnp.py`'s forward methods - they run on
   already-validated data, so the extra overhead buys nothing.

3. **Constant management.** Constants shared across multiple modules with no
   algorithm-specific justification belong in
   [`src/visqai/constants.py`](src/visqai/constants.py). Empirically calibrated,
   algorithm-specific thresholds stay local to the module that uses them, next to the comment
   that justifies the value.

4. **Self-contained evaluation modules.** Each eval suite (parity, LOGO, zero-shot,
   learning-curve) is self-contained in its own `eval/*_eval.py`, exposing a `run(**kwargs)`
   function for the core logic and a `main(argv=None)` function as the CLI wrapper. Only code
   genuinely shared across evals (metrics, plotting style) lives outside the eval's own file.

5. **Automated checkpoint naming.** Never hand-pick or hardcode a path for a checkpoint or
   deployment package. Every training run and packaging step writes to an auto-generated
   `<root>/<date>/<time>/` directory via `visqai.paths.dated_run_dir`; use
   `visqai.paths.latest_checkpoint_dir` to look up the most recent run.

6. **No backwards-compatibility shims.** Don't keep legacy re-exports or wrapper shims for
   moved or renamed code. When something is relocated or renamed, update every call site
   immediately instead of letting compatibility debt accumulate.

7. **Code quality tools.** CI runs `flake8` (hard-fails on syntax errors/undefined names,
   warns on everything else) and `pylint --exit-zero` (informational only) against every push;
   run `pytest` locally before opening a PR. Follow standard PEP 8 conventions (e.g. `is` / `is
   not` for `None` comparisons), and use the project's `logging` module
   (`visqai.logging_config.configure_logging` + `logging.getLogger(__name__)`) rather than bare
   `print` for anything beyond a short-lived CLI progress message.

## Adding a New Evaluation Suite

1. Create `src/visqai/eval/<name>_eval.py` implementing the `run()` / `main()` structure
   described above.
2. Import only genuinely shared utilities; keep eval-specific logic self-contained in the new
   file.
3. Wire the new eval into `scripts/run.py` if it should run as part of the one-click pipeline,
   defaulting to off if it depends on external data.
4. Add `tests/unit/test_<name>_eval.py` covering the new eval's core logic.

## Continuous Integration

- **[python-app.yml](.github/workflows/python-app.yml)** - installs `requirements.txt`, lints
  with `flake8`, runs `pytest --cov=visqai --cov-report=term-missing`.
- **[pylint.yml](.github/workflows/pylint.yml)** - informational `pylint --exit-zero` pass.
- **[codeql.yml](.github/workflows/codeql.yml)** - CodeQL security analysis.

Treat a local `pytest` run as the source of truth for whether your change is correct; CI is the
same command run automatically on every push and PR.

## Pull Request Workflow

1. Branch from `main`.
2. Scope each PR to a single logical change.
3. Make sure `pytest` passes locally before opening the PR.
4. Request review from the team for changes to core architectural files (`models/cnp.py`,
   `training/loop.py`, `inference/predictor.py`).
