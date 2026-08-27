# Architecture

## What this system does

Given a formulation's physicochemical properties (protein type/concentration, buffer, salt,
stabilizer, surfactant, excipient, pH, temperature, ...), predict its viscosity at five shear
rates. Predictions come from a **Conditional Neural Process (CNP)**: a physics-informed
zero-shot **prior**, optionally corrected by a **few-shot** signal learned on the fly from a
handful of real measurements on similar formulations (no retraining required to use context).

## End-to-end data flow

``
raw formulation export (Dropbox, data/latest/*.xlsx or *.csv)
        |
        v
visqai.paths.load_table               -- xlsx/csv, whichever it is
        |
        v
visqai.features.dataprocessor.build_feature_frame
        |  categorical/charge featurization, physics priors, engineered columns
        v
visqai.training.data.load_and_preprocess
        |  fits ColumnTransformer + physics_scaler, PCHIP-resamples each curve
        |  onto a fixed shear grid -> list of {static, points, group, id} samples
        v
visqai.training.run.train_final_model
        |  (optionally preceded by visqai.training.tuning.objective_cv, Optuna)
        |  uses visqai.training.loop's train_epoch/validate/validate_zero_shot
        |  trains a visqai.models.cnp.CrossSampleCNP
        v
checkpoint saved: <MODELS_ROOT>/checkpoints/<date>/<time>/best_model.pth (+ preprocessor.pkl,
                  physics_scaler.pkl, protected_indices.pkl)
        |
        +--> visqai.eval.*_eval                    (sanity-check the checkpoint)
        |
        +--> visqai.inference.predictor.ViscosityPredictorCNP   (serve predictions)
        |
        +--> visqai.packaging.packager.SecurePredictorPackager  (ship to a client)
                    |
                    v
              <MODELS_ROOT>/production/<date>/<time>/visqai_*.visq  (signed zip)
``

`scripts/run.py` chains data-processing -> training -> eval in one process; `scripts/package.py`
runs the packaging step. Each stage is also independently callable (see
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-eval) for the eval-module shape).

## Package layout

### `visqai.constants` / `visqai.paths`

Single source of truth for values shared across modules (shear-rate schema, plot palette,
`DEFAULT_PARAMS`, the Dropbox storage roots) and the behavior that resolves/consumes them
(`load_table`, `dated_run_dir`, `latest_data_file`, `latest_checkpoint_dir`). Storage roots
resolve dynamically per machine -- `VISQAI_DATA_ROOT`/`VISQAI_MODELS_ROOT` env vars if set,
else Dropbox's own `info.json` (correct even with a customized install location), else a
`Path.home()`-based guess as a last resort. Nothing here raises at import time, so a machine
without Dropbox mounted at all (CI, a fresh clone) doesn't take unrelated constants down with it
-- a missing root only surfaces as a clear error when something actually tries to use a path
under it.

### `visqai.validation`

Shared input-validation helpers (`require_dataframe`, `require_path_exists`,
`require_positive`, `require_in`, ...) applied at the functions an external caller can reach
directly with bad input -- not inside `training.loop`'s hot path or `models.cnp`'s forward
methods. See its module docstring and [CONTRIBUTING.md](CONTRIBUTING.md#code-conventions).

### `visqai.features`

Raw row -> engineered feature frame, the shared input-processing layer both training and
inference go through (previously duplicated and divergent between the two -- see
`dataprocessor.py`'s module docstring for the bug that caused):

- `categorical.py` -- replaces one-hot categorical encodings (buffer/salt/stabilizer/surfactant/
  excipient/protein-class type) with physicochemical property vectors, so the model can
  interpolate between chemically similar entities instead of treating every category as an
  orthogonal label.
- `charge.py` -- the single raw net-charge measurement (`whole_charge`); every derived charge
  feature explored in earlier iterations was dropped, only this one made the final model.
- `priors.py` -- the charge-coupling-index (CCI) / regime classification and the
  concentration-threshold prior table that select a formulation's excipient-interaction prior.
- `dataprocessor.py` -- `build_feature_frame` (the full pipeline: numeric defaults, categorical/
  charge featurization, unit normalization, engineered physics columns, per-row priors) and
  `prepare_df` (row-level cleanup: int->float coercion, ID->str, optional bad-row filtering).

### `visqai.training`

- `data.py` -- `load_and_preprocess`: fits the `ColumnTransformer` (StandardScaler + one-hot) and
  a separate `physics_scaler` (on log-shear/log-viscosity pairs), PCHIP-interpolates each
  formulation's curve and resamples it onto a fixed shear grid, and packages everything into
  `{static, points, group, id}` sample dicts. Also patches sklearn's degenerate zero-variance
  `scale_=1` (a real held-out-fold hazard) to a fixed fallback scale.
- `loop.py` -- `train_epoch` (the combined prior-MSE + correction-MSE + contrastive-triplet +
  consistency + context-utility + norm-penalty loss, with hard-group EMA oversampling and
  separate gradient clipping for `prior_head`) and the validation functions
  (`validate`/`validate_zero_shot`/`validate_fewshot`) plus latent-collapse diagnostics
  (`log_latent_variance`/`log_flatness`). Hot-path code, deliberately not input-validated -- see
  `visqai.validation`'s docstring.
- `run.py` -- `train_final_model`: the outer training loop (stratified per-group early-stopping
  split, checkpoint selection on a 50/50 mix of context-informed and zero-shot validation loss --
  see "why" below).
- `tuning.py` -- `objective_cv`: Optuna's hyperparameter-search objective, scored on few-shot
  held-out error (the metric that matches deployment) rather than full-context error.

### `visqai.models`

`cnp.py` -- `CrossSampleCNP`: an attention-pooled encoder over context points, feeding a
**two-head decoder**:

- `prior_head(query)` -- the feature-only, zero-shot prediction. It never sees the pooled
  context `r`, so a bad/OOD context literally cannot reach it.
- `correction_head(query, r) = g` -- the only path with access to context. Its final layer is
  zero-initialized, so at construction (and for any `r` it hasn't learned to react to)
  `g` contributes exactly zero: `prediction = prior_head(query) + g(query, r=0) = prior_head(query)`.

This split exists because a single combined decoder let a bad/OOD context actively make a good
zero-shot prediction *worse* (documented regression case in `cnp.py`'s module docstring). With
the split, `training.loop.train_epoch` fits `prior_head` directly to `y` and `correction_head` to
the **residual** `y - prior_head(query).detach()` -- the `.detach()` is load-bearing, it's what
stops the correction head from ever being rewarded for re-deriving `y` on its own instead of
explaining what the prior gets wrong. The hard backstop one layer up is
`eval.logo_eval`'s **context gate**: the LOGO harness asserts few-shot never scores worse than
zero-shot on any held-out group, or the run fails loudly rather than shipping a regression.

### `visqai.inference`

`predictor.py` -- `ViscosityPredictorCNP`: loads a checkpoint + its fitted preprocessor/
physics_scaler, and serves `.predict()` (zero-shot by default) and `.learn()` (encode a context
set for few-shot). Also implements a **delta corrector** (query-conditioned local residual,
selectable `corrector_mode="linear"|"kernel"`) layered on top of the neural correction head for
the specific case where a few real context points are available -- see the module's own
"QUERY-CONDITIONED LOCAL RESIDUAL" docstring section for the empirical calibration behind it
(`TAU2_CONC`, `SIGMA2_WITHIN`, `KERNEL_BANDWIDTH_CANDIDATES`, ...). Descriptor-OOD down-weighting
(`visqai.constants.DESCRIPTOR_OOD_CLIP_SIGMA`) clips scaled numeric features at inference time the
same way `training.data` does at fit time, so a held-out formulation's real value can't inject an
unbounded activation the network never learned to handle.

### `visqai.eval`

Four self-contained evals (each a `run()` + `main()` pair -- see
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-eval)):

- **`parity_eval`** -- Ibalizumab context-select-and-score: greedy forward selection (+ swap
  refinement) of a small, strategic context set, then per-shear parity metrics/plots on the
  correctly held-out remainder.
- **`logo_eval`** -- the primary Phase-0 scoreboard: leave-one-**GROUP**-out across three axes
  (protein, ingredient, protein-class), reporting a feature-only HistGBM baseline alongside CNP
  zero-shot/few-shot log-MAE, plus the leave-one-ingredient-out property-vector ablation. This is
  where the context gate (above) is enforced.
- **`zero_shot_eval`** -- standalone benchmark on proteins entirely absent from training (by
  construction, not derivable from the training master file). Owns the held-out-panel quarantine
  registry (`HELDOUT_PANELS`) that tags any metric computed from a previously-EDA'd panel as an
  in-sample estimate, not a validation number, with a durable audit log of every access attempt.
- **`learning_curve_eval`** -- step-by-step context-addition replay: adds held-out samples one at
  a time, recording convergence + shape-fidelity metrics at each step, to visualize how much
  context actually helps and where it plateaus.

Genuinely shared code (`visqai.constants`'s shear schema/palette, `eval.metrics`'s two metric
entrypoints, `eval.style`'s plotting functions) stays shared; everything used by exactly one eval
lives inside that eval's file, including its own plotting and data-prep helpers.

### `visqai.packaging`

`packager.py` -- `SecurePredictorPackager`: bundles a checkpoint + the runtime-inference-only
source modules (`features/`, `models/cnp.py`, `inference/predictor.py` -- training/eval code is
deliberately never shipped to a client) into a signed `.visq` zip, with per-file RSA signatures
(`signer.py`) a client can verify before trusting the package. Output location and package name
are always auto-generated (see "Storage" below) -- never hand-picked.

### `scripts/`

`run.py` and `package.py` live outside `src/visqai/` deliberately: `src/` is the installable
library (`pip install -e .`), `scripts/` holds directly-invoked operational entrypoints that
import it. Neither is registered as a `pip`-installed console command -- run them with
`python scripts/run.py` / `python scripts/package.py` from the repo root.

## Storage: checkpoints and production packages

Nothing under `models/` lives in the repo (proprietary, and part of why `.git` history was
cleaned up during the initial handoff pass). Both checkpoints and `.visq` packages are written to
the team's Dropbox under an **auto-generated `<date>/<time>` directory**
(`visqai.paths.dated_run_dir`) -- never a hand-picked descriptive name. This means runs never
collide and there's nothing to remember to invent, at the cost of needing to look a run up by
timestamp (or just take the newest, via `visqai.paths.latest_checkpoint_dir`, which every eval
module's `--model_dir` defaults to when omitted).

``
<MODELS_ROOT>/
  checkpoints/<date>/<time>/
    best_model.pth, preprocessor.pkl, physics_scaler.pkl, protected_indices.pkl
    eval_parity/, eval_zero_shot/, eval_logo/, eval_learning_curve/   -- eval output lives
                                                                          alongside its checkpoint
  production/<date>/<time>/
    visqai_{single,ensemble}_<timestamp>.visq
``

Training/eval logs follow the same rule: `logging_config.configure_logging(log_dir=...)` is
called with the run's own dated directory, once it's resolved -- not a shared global `logs/`.

## Data

Formulation data lives at `<DATA_ROOT>/latest/` (the current master export, `.xlsx` or `.csv` --
`visqai.paths.load_table` handles either) and `<DATA_ROOT>/legacy/` (superseded snapshots).
There is no `data/processed/` in this repo -- prior derived splits (a pre-filtered "exclude
ibalizumab" training set, a curated zero-shot panel, etc.) were deliberately not carried forward
when data storage moved to Dropbox; evals that need one now require it explicitly
(`--data_csv`/`--ibal_csv`) rather than falling back to a stale checked-in copy.
