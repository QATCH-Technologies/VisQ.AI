"""
condition_shift.py
===================
Task 0.1 (issue1_query_conditioned_correction_plan.md): measures correction
quality under INDUCED CONDITION SHIFT between context and target, which the
random-split protein LOGO (visqai.eval.cnp_logo.run_cnp_logo) cannot see --
that harness draws context and target as a random k/rest split of the SAME
held-out protein's rows, so context and target are exchangeable draws from
the same condition distribution. This harness instead deliberately splits a
held-out protein's own rows along a condition axis (concentration /
ingredient presence / buffer identity) so context and target sit on
OPPOSITE sides of that axis -- the capability the correction mechanism
exists to provide (Weakness #1: "does context that spans a condition axis
improve prediction on held-out conditions along that axis") is invisible to
a random split and only shows up here.

For each held-out protein: trains a fresh fold model exactly as
visqai.eval.cnp_logo._train_fold_model does (fresh preprocessor per fold --
no leakage of held-out statistics), then for each of the three axes below,
scores prior_only (memory_vector=None) vs corrected (context-informed
.learn()) on the OPPOSITE side of the split from context.

Axes
----
- concentration: context = held-out protein's own lower Protein_conc half
  (below the protein's own median), target = upper half, and the reverse
  direction (context=upper, target=lower). Every protein with >=4 valid
  rows qualifies for both directions.
- ingredient (buffer/salt/stabilizer/surfactant/excipient presence): for
  each of visqai.eval.logo_splits.INGREDIENT_COLS present for this protein
  with both a "has this ingredient" and "does not" subset, context = the
  ingredient-absent rows, target = the ingredient-present rows (the
  direction that matters: predicting the effect of ADDING an ingredient
  from context that never had it).
- buffer: context = one Buffer_type's rows, target = another's, for every
  ordered pair of buffer types the protein has real rows in (only when the
  protein has >=2 distinct buffer types).

Output schema (one row per protein/axis/direction; STABLE -- Phase 1/2 tasks
assert on these columns, do not rename without updating them):

    protein, axis, direction, n_ctx_formulations, n_ctx_points,
    n_target_formulations, n_target_points, prior_only_log_mae,
    corrected_log_mae, delta

`delta = prior_only_log_mae - corrected_log_mae` (positive = the corrector
helped on the shifted target). This is the Task 1.x capability metric --
NOT cnp_logo's `lift`, which is scored on a same-distribution random split
and structurally cannot see condition-shift capability.

Ground rule #1 (see the plan): this eval is the Phase 1/2 acceptance
yardstick. Never train/select/calibrate on it -- `shift_validity_check`
below is a one-time sanity check on the SPLIT itself, not a metric to
optimize a corrector against.

TASK A.1 -- STRATIFIED REPORTING (why mean delta alone is not trustworthy)
---------------------------------------------------------------------------
A full acceptance run's headline numbers (mean delta per axis) turned out to
be almost entirely a BROKEN-PRIOR RESCUE effect, not evidence of genuine
query-conditioned condition transfer: across the 63/117 splits where the
corrector actually fired (delta != 0), corr(prior_only_log_mae, delta) =
+0.84 -- the corrector's apparent "help" tracks almost linearly with how bad
the zero-shot prior already was, not with whether context spans the query's
condition. Splitting on prior_only_log_mae < 0.15 ("good prior") vs. >= 0.15
("bad prior") among fired splits: good-prior mean delta = -0.0214 (9/28
helped), bad-prior mean delta = +0.1103 (31/35 helped) -- the corrector is
net HARMFUL, on average, exactly where the prior didn't need rescuing.
Worse: on the concentration axis, excluding vudalimab alone flips the axis
mean delta from +0.014 to -0.008; excluding vudalimab AND poly-hIgG, -0.013
-- a single protein's outsized correction (vudalimab's own kP=5.0 is the
most extreme in the dataset) was propping up the entire axis's headline
number. And `conc_high_ctx_low_target` (extrapolating DOWN in concentration)
fails `shift_validity_check` on its own terms -- it is measurably EASIER
than the random-split zero-shot baseline for 11/12 proteins, i.e. it tests
interpolation into well-covered territory, not extrapolation, and has no
business being pooled into a capability headline next to the direction that
actually does extrapolate (`conc_low_ctx_high_target`).

`stratified_summary` and `leave_one_protein_out_sensitivity` below exist so
this failure mode is visible by construction, not something that has to be
independently rediscovered by cross-referencing the raw board: every
aggregate is reported alongside its own good-prior/bad-prior/fired-only
breakdown, median (not just mean), win rate, and a per-protein exclusion
sensitivity. `axis_rollup` requires an explicit `validated` set and refuses
to run without one, so a direction that fails its own extrapolation check
(e.g. conc_high_ctx_low_target) cannot silently leak into a headline number
-- direction_validated being populated is not optional/informational, it
GATES the rollup.

TASK A.2 -- OFFSET-ONLY ABLATION (ship decision: linear+clamp, not kernel)
-----------------------------------------------------------------------------
Scoring the PRE-Task-1.1 corrector (`corrector_mode="offset_only"` --
identical fitting code with ctx_conc forced to zero, see predictor.py)
against linear (1.1) and kernel (1.2) on the SAME folds/seed, restricted to
validated directions, three-way (bad-prior is the SIGNIFICANT stratum here
-- see the MDE finding below; ship decisions must be driven by it, not by
the good-prior stratum where nothing is measurable):

    corrector          bad-prior (SIG)   good-prior (n.s.)
    offset-only (dumb)     +0.054            -0.010
    linear + A.3 clamp     +0.075            -0.005
    kernel post-A.3        +0.020            +0.008

Linear+clamp DOMINATES offset-only on both strata and is the corrector that
ships. Post-A.3 kernel captures only ~27% of linear's bad-prior gain (a real
loss, well outside that stratum's CI) in exchange for a good-prior "gain"
that sits inside the noise floor (see MDE below) -- i.e. kernel was kept
once for the wrong reason (best number in the stratum with no measurable
signal) and that was the mirror image of the broken-prior-rescue mistake
this module's Task A.1 section exists to catch. Kernel stays in the
codebase (`corrector_mode="kernel"`) as an explicit, non-default research
branch -- it is not the shipped/default corrector.

TASK A.2 ADDENDUM -- THE EVAL IS UNDERPOWERED BY CONSTRUCTION
-----------------------------------------------------------------------------
`minimum_detectable_effect` quantifies why a small/null good-prior result
must not be read as "no capability": at the board's real unit of
independence (proteins, not rows), between-protein SD on the good-prior
stratum is ~0.02 at n=11 clusters, giving MDE ~= 0.017 log MAE (95%
confidence / 80% power). Every corrector variant tested differs from every
other by LESS than that on the good-prior stratum -- this board could not
have resolved a real, meaningfully-sized corrector improvement there even
if one existed. This is why Phase 2's physics-pretraining task (2.4) is not
only a prior fix: unlimited synthetic proteins give hundreds of independent
clusters instead of 11, which is the only realistic route to an MDE small
enough to actually A/B correctors on the good-prior stratum. Report
`minimum_detectable_effect` alongside every future condition-shift
comparison -- a corrector A/B on this board should always state its MDE, so
a small/null good-prior difference reads as "undetectable at n=11", not
"no effect."
"""

from __future__ import annotations

import itertools
import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd

from visqai.eval.cnp_logo import _train_fold_model
from visqai.eval.constants import SHEAR_COLS
from visqai.eval.data_prep import prepare_df
from visqai.eval.logo_splits import INGREDIENT_COLS, _NULL_CATEGORIES, _norm, build_groups
from visqai.eval.metrics import calc_metrics
from visqai.inference.predictor import ViscosityPredictorCNP

# Minimum valid (non-null Protein_conc) rows a held-out protein needs before
# the concentration axis attempts a median split -- below this a "lower
# half" / "upper half" split is too degenerate to mean anything (e.g. could
# leave one side with a single formulation).
MIN_ROWS_FOR_CONCENTRATION_SPLIT: int = 4


def _pooled_metrics(target_df: pd.DataFrame, pred_df: pd.DataFrame) -> Optional[dict]:
    """calc_metrics pooled across every shear column -- same pooling
    visqai.eval.cnp_logo._shot_metrics uses, so prior_only_log_mae/
    corrected_log_mae here are directly comparable to cnp_logo's
    zero_shot_log_mae/fewshot_k*_log_mae columns."""
    all_true, all_pred = [], []
    for sc in SHEAR_COLS:
        pc = f"Pred_{sc}"
        if sc not in target_df.columns or pc not in pred_df.columns:
            continue
        true = pd.to_numeric(target_df[sc], errors="coerce")
        pred = pd.to_numeric(pred_df[pc], errors="coerce")
        mask = true.notna()
        if mask.any():
            all_true.append(true[mask].values)
            all_pred.append(pred[mask].values)
    if not all_true:
        return None
    return calc_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def _n_real_points(df: pd.DataFrame) -> int:
    cols = [c for c in SHEAR_COLS if c in df.columns]
    if not cols:
        return 0
    return int(df[cols].notna().sum().sum())


def concentration_split(held_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """[(direction, ctx_df, target_df), ...] for the concentration axis:
    lower half (below the protein's own median Protein_conc) as context vs.
    upper half as target, and the reverse. Ties sit in the upper half so a
    heavily-repeated concentration value can't starve that side."""
    if "Protein_conc" not in held_df.columns:
        return []
    conc = pd.to_numeric(held_df["Protein_conc"], errors="coerce")
    valid = held_df[conc.notna()].reset_index(drop=True)
    conc = pd.to_numeric(valid["Protein_conc"], errors="coerce")
    if len(valid) < MIN_ROWS_FOR_CONCENTRATION_SPLIT:
        return []
    median = conc.median()
    lower_mask = conc < median
    lower = valid[lower_mask].reset_index(drop=True)
    upper = valid[~lower_mask].reset_index(drop=True)
    if lower.empty or upper.empty:
        return []
    return [
        ("conc_low_ctx_high_target", lower, upper),
        ("conc_high_ctx_low_target", upper, lower),
    ]


def ingredient_splits(held_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """[(direction, ctx_df, target_df), ...] for the ingredient axis: one
    entry per INGREDIENT_COLS column the protein has both an absent- and
    present-subset for. context = absent (the corrector has never seen this
    ingredient's effect), target = present -- tests whether context that
    lacks an ingredient can still predict the effect of adding it, which it
    structurally cannot (that's the point: this should show near-zero delta
    until Task 1.2's kernel-weighted corrector or Phase 2's cross-attention
    give the mechanism something to condition on)."""
    out = []
    for col in INGREDIENT_COLS:
        if col not in held_df.columns:
            continue
        norm = _norm(held_df[col])
        has_mask = ~norm.isin(_NULL_CATEGORIES)
        present = held_df[has_mask].reset_index(drop=True)
        absent = held_df[~has_mask].reset_index(drop=True)
        if present.empty or absent.empty:
            continue
        out.append((f"{col}_absent_ctx_present_target", absent, present))
    return out


def buffer_splits(held_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """[(direction, ctx_df, target_df), ...] for the buffer axis: every
    ORDERED pair of distinct Buffer_type values the protein has real rows
    in (direction matters -- context=histidine/target=citrate is a
    different extrapolation than the reverse)."""
    if "Buffer_type" not in held_df.columns:
        return []
    norm = _norm(held_df["Buffer_type"])
    types = sorted(set(norm) - _NULL_CATEGORIES)
    if len(types) < 2:
        return []
    out = []
    for a, b in itertools.permutations(types, 2):
        ctx = held_df[norm == a].reset_index(drop=True)
        target = held_df[norm == b].reset_index(drop=True)
        if ctx.empty or target.empty:
            continue
        out.append((f"buffer_{a}_ctx_{b}_target", ctx, target))
    return out


def _score_split(predictor: ViscosityPredictorCNP, ctx_df: pd.DataFrame, target_df: pd.DataFrame):
    """Returns (prior_only_metrics, corrected_metrics) for one (ctx, target)
    split -- prior_only is the literal zero-shot path (memory_vector=None),
    corrected calls .learn(ctx_df) before predicting the SAME target_df.
    Always resets memory_vector back to None afterward so folds/splits never
    leak context into each other."""
    predictor.memory_vector = None
    predictor.context_t = None
    prior_pred = predictor.predict(target_df)
    prior_metrics = _pooled_metrics(target_df, prior_pred)

    predictor.learn(ctx_df)
    corrected_pred = predictor.predict(target_df)
    corrected_metrics = _pooled_metrics(target_df, corrected_pred)

    predictor.memory_vector = None
    predictor.context_t = None
    return prior_metrics, corrected_metrics


def run_condition_shift_fold(
    train_df: pd.DataFrame,
    held_df: pd.DataFrame,
    protein_key: str,
    work_dir: str,
    max_epochs: int = 500,
    patience: int = 80,
    params=None,
    seed: int = 0,
    corrector_mode: str = "linear",
) -> list[dict]:
    """Trains one fold's model (exactly like cnp_logo._train_fold_model) and
    scores every applicable axis/direction split for this held-out protein.
    Returns a list of row-dicts matching this module's documented output
    schema (empty list if the protein has too few held-out rows to split at
    all).

    `corrector_mode` ("linear" or "kernel") selects which few-shot corrector
    the predictor uses -- "linear" is Task 1.1's default; "kernel" opts into
    Task 1.2's kernel-weighted local residual model (the one this module's
    ingredient/buffer axes are meant to evaluate)."""
    held_df = prepare_df(held_df, drop_bad_rows=True)
    if len(held_df) < 2:
        return []

    _train_fold_model(train_df, work_dir, max_epochs, patience, params=params, seed=seed, held_df=held_df)
    predictor = ViscosityPredictorCNP(work_dir, verbose=False)
    predictor.corrector_mode = corrector_mode

    axis_splits = [
        ("concentration", concentration_split(held_df)),
        ("ingredient", ingredient_splits(held_df)),
        ("buffer", buffer_splits(held_df)),
    ]

    rows = []
    for axis, splits in axis_splits:
        for direction, ctx_df, target_df in splits:
            prior_m, corrected_m = _score_split(predictor, ctx_df, target_df)
            if prior_m is None or corrected_m is None:
                continue
            rows.append(
                {
                    "protein": protein_key,
                    "axis": axis,
                    "direction": direction,
                    "n_ctx_formulations": len(ctx_df),
                    "n_ctx_points": _n_real_points(ctx_df),
                    "n_target_formulations": len(target_df),
                    "n_target_points": _n_real_points(target_df),
                    "prior_only_log_mae": prior_m["log_mae"],
                    "corrected_log_mae": corrected_m["log_mae"],
                    "delta": prior_m["log_mae"] - corrected_m["log_mae"],
                }
            )
    return rows


def run_condition_shift(
    df: pd.DataFrame,
    base_work_dir: str,
    proteins: Optional[list[str]] = None,
    min_rows: int = 2,
    max_epochs: int = 500,
    patience: int = 80,
    params=None,
    seed: int = 0,
    keep_fold_dirs: bool = False,
    corrector_mode: str = "linear",
) -> pd.DataFrame:
    """Runs the condition-shift harness over every real protein (or a
    caller-supplied subset via `proteins`, for smoke tests). Each fold
    trains a fresh model in its own subdirectory of `base_work_dir`, deleted
    after scoring unless `keep_fold_dirs`. Deterministic given `seed` (no
    random context sub-sampling here -- every split is the FULL context/
    target side, unlike cnp_logo's k-shot draws).

    A fold that errors produces a single `axis="error"` row (message in
    `direction`) instead of aborting the whole run, mirroring
    visqai.eval.cnp_logo.run_cnp_logo's per-fold try/except -- this harness
    trains 11+ models and a single bad fold (e.g. an unstable optimization
    run) should not lose every other protein's results.

    `corrector_mode` ("linear" or "kernel") is forwarded to every fold's
    predictor -- see run_condition_shift_fold's docstring. Task 1.2's
    ingredient/buffer capability criteria are meant to be evaluated with
    corrector_mode="kernel".
    """
    fold_groups = build_groups(df, "protein", min_rows=min_rows)
    if proteins is not None:
        wanted = set(proteins)
        fold_groups = [g for g in fold_groups if g.key in wanted]

    rows = []
    os.makedirs(base_work_dir, exist_ok=True)
    for g in fold_groups:
        train_df, held_df = g.split(df)
        if held_df.empty or train_df.empty:
            continue
        fold_dir = os.path.join(base_work_dir, f"condshift__{g.key}")
        os.makedirs(fold_dir, exist_ok=True)
        try:
            fold_rows = run_condition_shift_fold(
                train_df,
                held_df,
                g.key,
                fold_dir,
                max_epochs=max_epochs,
                patience=patience,
                params=params,
                seed=seed,
                corrector_mode=corrector_mode,
            )
        except Exception as e:  # keep the harness going across folds
            fold_rows = [{"protein": g.key, "axis": "error", "direction": str(e)}]
        rows.extend(fold_rows)
        if not keep_fold_dirs:
            shutil.rmtree(fold_dir, ignore_errors=True)

    return pd.DataFrame(rows)


def shift_validity_check(condition_shift_df: pd.DataFrame, random_split_zero_shot: pd.DataFrame) -> dict:
    """Task 0.1's REQUIRED sanity check: confirms the concentration-shift
    split actually induces extrapolation (not interpolation) by comparing
    prior_only_log_mae against the SAME proteins' zero-shot log MAE from a
    random-split protein LOGO run (visqai.eval.cnp_logo.run_cnp_logo's
    `zero_shot_log_mae` column, axis="protein" -- pass that board's `group`
    column as the index key).

    Checked PER DIRECTION, not pooled across both. A real acceptance run
    (12/12 proteins, full training budget) showed a large, consistent,
    physically-expected asymmetry: extrapolating UP in concentration
    (direction="conc_low_ctx_high_target", context=protein's lower half,
    target=upper half) was dramatically harder than the random-split
    zero-shot baseline (mean prior_only_log_mae 0.237 vs. 0.190 baseline,
    12/12 proteins individually harder in that direction) -- higher
    concentrations sit in the more nonlinear, shear-thinning-dominated
    regime, which a model has less reason to extrapolate into cleanly.
    Extrapolating DOWN (direction="conc_high_ctx_low_target") was, just as
    consistently, EASIER than baseline (mean 0.119) -- the network's
    population-level training already handles the more Newtonian low-conc
    regime well, so this direction is not a meaningful stress test on its
    own. POOLING the two directions into a single mean (0.178) makes them
    cancel and the check fail spuriously, masking a real and robust signal
    behind an average that answers a question nobody asked ("is either
    direction typical") instead of the one that matters ("does at least one
    real condition-shift direction induce genuine extrapolation"). Reported
    per direction; `ok` is true if ANY direction clears the bar (a
    physically-easier direction failing it is expected, not a defect in the
    split -- see above), so a caller can also inspect exactly which
    direction(s) are the real stress test before trusting Task 1.x/2.x
    results against this axis."""
    conc = condition_shift_df[condition_shift_df["axis"] == "concentration"]
    if conc.empty:
        return {"ok": False, "reason": "no concentration-shift rows to check", "n_proteins": 0, "per_direction": {}}

    zshot = random_split_zero_shot.set_index("group")["zero_shot_log_mae"]

    per_direction = {}
    for direction, g in conc.groupby("direction"):
        shift_by_protein = g.groupby("protein")["prior_only_log_mae"].mean()
        common = [p for p in shift_by_protein.index if p in zshot.index and pd.notna(zshot[p])]
        if not common:
            continue
        shift_mean = float(shift_by_protein.loc[common].mean())
        random_mean = float(zshot.loc[common].mean())
        per_direction[direction] = {
            "ok": shift_mean > random_mean,
            "shift_mean_log_mae": shift_mean,
            "random_split_mean_log_mae": random_mean,
            "n_proteins": len(common),
        }

    if not per_direction:
        return {
            "ok": False,
            "reason": "no overlapping proteins with the random-split board",
            "n_proteins": 0,
            "per_direction": {},
        }

    return {
        "ok": any(v["ok"] for v in per_direction.values()),
        "per_direction": per_direction,
        "n_proteins": max(v["n_proteins"] for v in per_direction.values()),
    }


# Task A.1: log MAE below this = "the zero-shot prior was already healthy."
# Splitting on this threshold is what surfaced the broken-prior-rescue
# effect (see module docstring) -- a corrector that helps almost
# exclusively in the bad-prior stratum is not doing the query-conditioned
# condition-transfer job Weakness #1 asks for, whatever its aggregate mean
# delta says.
GOOD_PRIOR_THRESHOLD: float = 0.15


def validated_directions(condition_shift_df: pd.DataFrame, random_split_zero_shot: pd.DataFrame) -> set:
    """The set of concentration-axis directions shift_validity_check
    confirms actually induce extrapolation (not interpolation) -- e.g.
    `conc_low_ctx_high_target` but NOT `conc_high_ctx_low_target` in the
    real acceptance run (see module docstring). Every OTHER axis's
    directions (ingredient/buffer) are returned as validated by
    construction -- Task 0.1 does not define an analogous validity check
    for them (only the concentration axis has an unambiguous "harder
    direction" a priori)."""
    all_directions = set(condition_shift_df["direction"].unique())
    validity = shift_validity_check(condition_shift_df, random_split_zero_shot)
    conc_directions = set(condition_shift_df.loc[condition_shift_df["axis"] == "concentration", "direction"])
    valid_conc = {d for d, v in validity.get("per_direction", {}).items() if v.get("ok")}
    return (all_directions - conc_directions) | valid_conc


def stratified_summary(board: pd.DataFrame, validated: Optional[set] = None) -> pd.DataFrame:
    """Task A.1: re-cuts the condition-shift board so a single outlier
    protein or an unvalidated (interpolation, not extrapolation) direction
    cannot silently dominate or flip a headline number -- see module
    docstring for the concrete failure this responds to.

    One row per (axis, direction, stratum), where stratum in
    {"all", "good_prior", "bad_prior", "fired_only"} ("good"/"bad" is
    GOOD_PRIOR_THRESHOLD on prior_only_log_mae; "fired_only" restricts to
    delta != 0, i.e. excludes rows where the corrector's own gate abstained).
    Reports n, mean_delta, median_delta (outliers move the mean far more
    than the median -- see vudalimab), win_rate (delta > 0), and
    non_regress_rate (delta >= 0, the plan's literal "corrected <=
    prior_only" bar). `direction_validated` (True/False/None if `validated`
    wasn't supplied) flags directions shift_validity_check does NOT confirm
    induce real extrapolation -- these should not be pooled into a headline
    capability number without saying so."""
    df = board.copy()
    df["prior_band"] = np.where(df["prior_only_log_mae"] >= GOOD_PRIOR_THRESHOLD, "bad_prior", "good_prior")
    df["fired"] = df["delta"] != 0.0

    rows = []
    for (axis, direction), g in df.groupby(["axis", "direction"]):
        strata = {
            "all": g,
            "good_prior": g[g["prior_band"] == "good_prior"],
            "bad_prior": g[g["prior_band"] == "bad_prior"],
            "fired_only": g[g["fired"]],
        }
        for stratum_name, sub in strata.items():
            if sub.empty:
                continue
            rows.append(
                {
                    "axis": axis,
                    "direction": direction,
                    "direction_validated": (direction in validated) if validated is not None else None,
                    "stratum": stratum_name,
                    "n": len(sub),
                    "mean_delta": float(sub["delta"].mean()),
                    "median_delta": float(sub["delta"].median()),
                    "win_rate": float((sub["delta"] > 0).mean()),
                    "non_regress_rate": float((sub["delta"] >= 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def leave_one_protein_out_sensitivity(
    board: pd.DataFrame, axis: str, direction: Optional[str] = None
) -> pd.DataFrame:
    """Task A.1: for `axis` (optionally restricted to one `direction`),
    recomputes the aggregate mean delta with EACH protein excluded in turn
    -- surfaces exactly the failure mode where one protein's outsized
    correction (e.g. vudalimab, kP=5.0, the most extreme in the dataset) is
    propping up an entire axis's headline number (see module docstring: the
    concentration axis's mean delta flips sign, +0.014 -> -0.008, when
    vudalimab alone is excluded). Sorted by `mean_delta_excluding` so the
    single most load-bearing protein sorts to one end."""
    sub = board[board["axis"] == axis]
    if direction is not None:
        sub = sub[sub["direction"] == direction]
    full_mean = float(sub["delta"].mean()) if not sub.empty else float("nan")

    rows = []
    for p in sub["protein"].unique():
        rest = sub[sub["protein"] != p]
        rows.append(
            {
                "excluded_protein": p,
                "full_mean_delta": full_mean,
                "mean_delta_excluding": float(rest["delta"].mean()) if not rest.empty else float("nan"),
                "protein_own_mean_delta": float(sub.loc[sub["protein"] == p, "delta"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_delta_excluding").reset_index(drop=True)


def axis_rollup(summary: pd.DataFrame, validated: Optional[set] = None, strata=("all", "good_prior", "bad_prior")) -> pd.DataFrame:
    """Rolls `stratified_summary`'s per-direction rows up into ONE row per
    (axis, stratum), counting ONLY directions in `validated` -- the
    headline number that's actually safe to compare across corrector
    variants. Filters directly on `validated` (NOT the summary's own
    `direction_validated` column) so a stale/None flag can't silently pass
    a bad direction through.

    REQUIRES `validated` -- raises ValueError otherwise. A run made with
    --skip-validity-check has no way to know which directions are real
    extrapolation, and rolling THAT up would let a direction that fails its
    own validity check (e.g. conc_high_ctx_low_target -- measurably EASIER
    than the random-split baseline, i.e. interpolation, not extrapolation)
    leak back into a headline number, exactly the failure Task A.1 exists
    to prevent. Compute `validated_directions` first (see
    cli.condition_shift_eval's --random-split-board option to reuse a
    previously-saved random-split board instead of retraining one)."""
    if validated is None:
        raise ValueError(
            "axis_rollup requires `validated` (from validated_directions) -- refusing to roll up "
            "a summary with unknown direction validity, since that would silently let a direction "
            "that fails shift_validity_check (e.g. conc_high_ctx_low_target) leak into the "
            "headline aggregate. Compute validated_directions first (a real or cached "
            "random-split board is required)."
        )
    valid = summary[summary["direction"].isin(validated) & summary["stratum"].isin(strata)]
    rows = []
    for (axis, stratum), g in valid.groupby(["axis", "stratum"]):
        n = int(g["n"].sum())
        rows.append(
            {
                "axis": axis,
                "stratum": stratum,
                "n": n,
                "mean_delta": float((g["mean_delta"] * g["n"]).sum() / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["axis", "stratum"]).reset_index(drop=True)


# Standard one-sample MDE combo (95% two-sided confidence, 80% power) --
# see minimum_detectable_effect's docstring. Not a claim that this is the
# only reasonable choice, just the conventional default so MDE numbers are
# comparable run-to-run without re-deriving z-values each time.
MDE_ALPHA: float = 0.05
MDE_POWER: float = 0.80


def minimum_detectable_effect(board: pd.DataFrame, alpha: float = MDE_ALPHA, power: float = MDE_POWER) -> dict:
    """The smallest per-protein-clustered mean delta `board` could reliably
    distinguish from zero, at the board's ACTUAL unit of independence
    (proteins) rather than its much larger, correlated number of rows/
    splits. Standard one-sample MDE: `MDE = (z_(alpha/2) + z_power) * SE`,
    `SE = between-protein SD / sqrt(n_proteins)` -- callers should pass in
    `board` already filtered to whatever axis/stratum/validated-direction
    set the comparison is about (this function only pools to one row per
    protein via `board.groupby("protein")["delta"].mean()`, it doesn't
    filter anything itself).

    THIS EVAL IS UNDERPOWERED BY CONSTRUCTION: on the real 12-real-protein
    (11 with >=1 concentration split) acceptance board's good-prior
    stratum, between-protein SD~=0.02, giving MDE~=0.017 log MAE at n=11
    clusters -- and every corrector variant tested (offset-only vs. linear
    vs. kernel) differed from the others by LESS than that on the
    good-prior stratum. That does not mean there is no real difference
    between them; it means THIS BOARD CANNOT TELL, at this sample size,
    whatever the true difference is. Report this number alongside every
    condition-shift comparison so a null/small result on the good-prior
    stratum is never read as "no effect" when it may just be "undetectable
    at n=11" -- and see Task 2.4's reframing (unlimited synthetic proteins
    -> hundreds of independent clusters -> an MDE small enough to actually
    resolve corrector differences)."""
    from scipy import stats

    per_protein = board.groupby("protein")["delta"].mean()
    n = len(per_protein)
    if n < 2:
        return {
            "n_proteins": n,
            "between_protein_sd": float("nan"),
            "se": float("nan"),
            "mde": float("nan"),
            "observed_mean_delta": float(per_protein.mean()) if n else float("nan"),
        }
    sd = float(per_protein.std(ddof=1))
    se = sd / (n**0.5)
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    return {
        "n_proteins": n,
        "between_protein_sd": sd,
        "se": se,
        "mde": z * se,
        "observed_mean_delta": float(per_protein.mean()),
    }
