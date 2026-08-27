from __future__ import annotations

import argparse
import logging
import os
import shutil
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from visqai import constants, paths
from visqai.constants import DEFAULT_PARAMS, SHEAR_COLS, SHEAR_RATES
from visqai.eval.metrics import calc_metrics
from visqai.features.dataprocessor import build_feature_frame, prepare_df
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging
from visqai.training.data import load_and_preprocess
from visqai.training.run import train_final_model

logger = logging.getLogger("LogoEval")

AXES = ["protein", "ingredient", "protein_class"]


PROTEIN_DESCRIPTOR_RAW_COLS = ["Whole_Antibody_Charge_at_Buffer_pH", "Whole_Charge", "PI_mean"]


def zero_protein_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PROTEIN_DESCRIPTOR_RAW_COLS:
        if col in out.columns:
            out[col] = np.nan
    return out


INGREDIENT_COLS = [
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]
_NULL_CATEGORIES = {"none", "unknown", "nan", "na", "n/a", ""}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


@dataclass(frozen=True)
class LogoGroup:
    axis: str
    key: str
    column: str
    value: str

    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series(False, index=df.index)
        return _norm(df[self.column]) == self.value

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        held_mask = self.mask(df)
        held = df[held_mask].copy().reset_index(drop=True)
        train = df[~held_mask].copy().reset_index(drop=True)
        return train, held


def protein_groups(df: pd.DataFrame, min_rows: int = 2) -> list[LogoGroup]:
    counts = _norm(df["Protein_type"]).value_counts()
    groups = []
    for val, n in counts.items():
        if val in _NULL_CATEGORIES or n < min_rows:
            continue
        groups.append(LogoGroup(axis="protein", key=val, column="Protein_type", value=val))
    return sorted(groups, key=lambda g: g.key)


def protein_class_groups(df: pd.DataFrame, min_rows: int = 2) -> list[LogoGroup]:
    counts = _norm(df["Protein_class_type"]).value_counts()
    groups = []
    for val, n in counts.items():
        if val in _NULL_CATEGORIES or n < min_rows:
            continue
        groups.append(
            LogoGroup(axis="protein_class", key=val, column="Protein_class_type", value=val)
        )
    return sorted(groups, key=lambda g: g.key)


def ingredient_groups(df: pd.DataFrame, min_rows: int = 2) -> list[LogoGroup]:
    groups = []
    for col in INGREDIENT_COLS:
        if col not in df.columns:
            continue
        counts = _norm(df[col]).value_counts()
        for val, n in counts.items():
            if val in _NULL_CATEGORIES or n < min_rows:
                continue
            groups.append(LogoGroup(axis="ingredient", key=f"{col}={val}", column=col, value=val))
    return sorted(groups, key=lambda g: g.key)


AXIS_BUILDERS = {
    "protein": protein_groups,
    "ingredient": ingredient_groups,
    "protein_class": protein_class_groups,
}


def build_groups(df: pd.DataFrame, axis: str, min_rows: int = 2) -> list[LogoGroup]:
    if axis not in AXIS_BUILDERS:
        raise ValueError(f"Unknown axis '{axis}' (expected one of {list(AXIS_BUILDERS)})")
    return AXIS_BUILDERS[axis](df, min_rows=min_rows)


def zero_ingredient_properties(df: pd.DataFrame, group: LogoGroup) -> pd.DataFrame:
    if group.axis != "ingredient":
        raise ValueError("zero_ingredient_properties only applies to the ingredient axis")
    out = df.copy()
    out[group.column] = "none"
    return out


def _melt_long(df: pd.DataFrame) -> pd.DataFrame:
    feat_df, num_cols, cat_cols = build_feature_frame(df)
    blocks = []
    for shear_col, shear_val in zip(SHEAR_COLS, SHEAR_RATES):
        if shear_col not in feat_df.columns:
            continue
        v = pd.to_numeric(feat_df[shear_col], errors="coerce")
        valid = v.notna() & (v > 0)
        if not valid.any():
            continue
        block = feat_df.loc[valid, num_cols + cat_cols].copy()
        block["log10_shear"] = np.log10(shear_val)
        block["_log10_visc"] = np.log10(v[valid].values)
        blocks.append(block)
    if not blocks:
        return pd.DataFrame(), [], []
    return pd.concat(blocks, ignore_index=True), num_cols + ["log10_shear"], cat_cols


def _make_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_depth=6, max_iter=300, learning_rate=0.05, random_state=0
    )
    return Pipeline([("pre", pre), ("model", model)])


def fit_baseline(train_df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    train_long, num_cols, cat_cols = _melt_long(train_df)
    feat_cols = num_cols + cat_cols
    pipe = _make_pipeline(num_cols, cat_cols)
    pipe.fit(train_long[feat_cols], train_long["_log10_visc"])
    return pipe, feat_cols


def score_baseline(pipe: Pipeline, feat_cols: list[str], held_df: pd.DataFrame) -> dict:
    held_long, _, _ = _melt_long(held_df)
    if held_long.empty:
        return dict(
            n=0,
            mae=np.nan,
            mape=np.nan,
            rmse=np.nan,
            r2=np.nan,
            log_mae=np.nan,
            log_rmse=np.nan,
            log_bias=np.nan,
            within_2x=np.nan,
        )
    pred_log = pipe.predict(held_long[feat_cols])
    true_cp = np.power(10, held_long["_log10_visc"].values)
    pred_cp = np.power(10, pred_log)
    return calc_metrics(true_cp, pred_cp)


def run_baseline_fold(
    train_df: pd.DataFrame, held_df: pd.DataFrame, group: LogoGroup | None = None
) -> dict:
    pipe, feat_cols = fit_baseline(train_df)
    m = score_baseline(pipe, feat_cols, held_df)
    m["n_train_rows"] = len(train_df)
    m["n_held_rows"] = len(held_df)

    if group is not None and group.axis == "ingredient":
        ablated_held = zero_ingredient_properties(held_df, group)
        m_ablated = score_baseline(pipe, feat_cols, ablated_held)
        m["ablation_log_mae"] = m_ablated["log_mae"]
        m["ablation_delta"] = m_ablated["log_mae"] - m["log_mae"]

    return m


def run_baseline_logo(df: pd.DataFrame, axis: str, min_rows: int = 2, groups=None) -> pd.DataFrame:
    fold_groups = groups if groups is not None else build_groups(df, axis, min_rows=min_rows)
    rows = []
    for g in fold_groups:
        train_df, held_df = g.split(df)
        if held_df.empty or train_df.empty:
            continue
        m = run_baseline_fold(train_df, held_df, group=g)
        rows.append({"axis": g.axis, "group": g.key, **m})
    return pd.DataFrame(rows)


FOLD_RANGE_N_SIGMA: float = 5.0


def _check_fold_feature_range(work_dir, held_df, n_sigma: float = FOLD_RANGE_N_SIGMA) -> dict:
    import joblib

    preprocessor = joblib.load(os.path.join(work_dir, "preprocessor.pkl"))
    num_cols = list(preprocessor.transformers_[0][2])
    scaler = preprocessor.named_transformers_["num"]

    held_built, _num_cols, _cat_cols = build_feature_frame(held_df)

    violations = {"zero_variance": [], "out_of_range": {}}
    for i, col in enumerate(num_cols):
        if col not in held_built.columns:
            continue
        if scaler.var_[i] <= 1e-12:
            violations["zero_variance"].append(col)

        vals = pd.to_numeric(held_built[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        scale = scaler.scale_[i] if scaler.scale_[i] > 0 else 1.0
        z = np.abs((vals - scaler.mean_[i]) / scale)
        n_bad = int((z > n_sigma).sum())
        if n_bad:
            violations["out_of_range"][col] = {
                "n_bad": n_bad,
                "n_total": len(vals),
                "max_abs_z": float(z.max()),
            }

    if violations["zero_variance"] or violations["out_of_range"]:
        print(
            "  [logo_eval] FOLD RANGE GUARD fired -- "
            f"zero-variance train columns={violations['zero_variance']}; "
            f"held-out values beyond {n_sigma}sigma of train="
            f"{violations['out_of_range']}"
        )

    return violations


def _train_fold_model(
    train_df, work_dir, max_epochs, patience, params=None, seed=None, held_df=None
):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    csv_path = os.path.join(work_dir, "train_fold.csv")
    train_df.to_csv(csv_path, index=False)
    samples, static_dim, physics_scaler, protected_indices = load_and_preprocess(
        csv_path, save_dir=work_dir
    )
    n_groups = len(set(s["group"] for s in samples))
    if n_groups < 2:
        raise ValueError(
            f"Fold has only {n_groups} protein group(s) after the split -- "
            "CNP needs >=2 to learn any cross-protein structure."
        )

    if held_df is not None:
        _check_fold_feature_range(work_dir, held_df)

    return train_final_model(
        samples,
        static_dim,
        physics_scaler,
        protected_indices,
        out_dir=work_dir,
        params=params or DEFAULT_PARAMS,
        max_epochs=max_epochs,
        patience=patience,
        verbose=False,
    )


REFERENCE_N_HELD_FOR_REPEATS: int = 72
MAX_N_REPEATS_MULTIPLIER: int = 6


def _effective_n_repeats(n_held: int, n_repeats: int) -> int:
    if n_held <= 0:
        return n_repeats
    scale = min(MAX_N_REPEATS_MULTIPLIER, max(1.0, REFERENCE_N_HELD_FOR_REPEATS / n_held))
    return int(round(n_repeats * scale))


def _shot_metrics(predictor, held_df, k, n_repeats, rng):
    n = len(held_df)
    if k > 0 and k >= n:
        return None

    reps = n_repeats if k > 0 else 1
    all_true, all_pred = [], []
    for _ in range(reps):
        predictor.memory_vector = None
        predictor.context_t = None
        if k == 0:
            target_df = held_df
        else:
            idx = rng.permutation(n)
            ctx_idx, tgt_idx = idx[:k], idx[k:]
            predictor.learn(held_df.iloc[ctx_idx].reset_index(drop=True))
            target_df = held_df.iloc[tgt_idx].reset_index(drop=True)

        pred_df = predictor.predict(target_df)
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


def run_cnp_fold(
    train_df: pd.DataFrame,
    held_df: pd.DataFrame,
    group: LogoGroup,
    work_dir: str,
    shots=(0, 1, 2, 4, 8),
    n_repeats=5,
    max_epochs=500,
    patience=80,
    params=None,
    seed=0,
    corrector_mode="linear",
) -> dict:
    held_df = prepare_df(held_df, drop_bad_rows=True)
    if len(held_df) < 2:
        return {
            "axis": group.axis,
            "group": group.key,
            "n_held": len(held_df),
            "error": "too few held-out rows",
        }

    _train_fold_model(
        train_df, work_dir, max_epochs, patience, params=params, seed=seed, held_df=held_df
    )
    predictor = ViscosityPredictorCNP(work_dir, verbose=False)
    predictor.corrector_mode = corrector_mode

    rng = np.random.RandomState(seed)
    effective_n_repeats = _effective_n_repeats(len(held_df), n_repeats)
    row = {
        "axis": group.axis,
        "group": group.key,
        "n_held": len(held_df),
        "n_repeats_used": effective_n_repeats,
    }
    for k in shots:
        m = _shot_metrics(predictor, held_df, k, effective_n_repeats, rng)
        prefix = "zero_shot" if k == 0 else f"fewshot_k{k}"
        if m is None:
            row[f"{prefix}_log_mae"] = np.nan
        else:
            row[f"{prefix}_log_mae"] = m["log_mae"]
            row[f"{prefix}_log_rmse"] = m["log_rmse"]
            row[f"{prefix}_within_2x"] = m["within_2x"]
            row[f"{prefix}_n"] = m["n"]

    if group.axis == "ingredient":
        ablated_held = zero_ingredient_properties(held_df, group)
        m_abl = _shot_metrics(predictor, ablated_held, 0, 1, rng)
        row["ablation_zero_shot_log_mae"] = m_abl["log_mae"] if m_abl else np.nan
        zshot = row.get("zero_shot_log_mae", np.nan)
        if m_abl and not np.isnan(zshot):
            # Positive => real properties beat the zeroed/"unknown" fallback
            # (the property vector is buying extrapolation, as designed).
            row["ablation_delta"] = m_abl["log_mae"] - zshot

    zshot = row.get("zero_shot_log_mae", np.nan)
    fewshot_maes = [
        row[f"fewshot_k{k}_log_mae"]
        for k in shots
        if k != 0 and f"fewshot_k{k}_log_mae" in row and not np.isnan(row[f"fewshot_k{k}_log_mae"])
    ]
    if fewshot_maes and not np.isnan(zshot):
        best_fewshot = min(fewshot_maes)
        row["best_fewshot_log_mae"] = best_fewshot
        row["lift"] = zshot - best_fewshot

    k_lifts = {}
    if not np.isnan(zshot):
        for k in shots:
            if k == 0:
                continue
            mae_col = f"fewshot_k{k}_log_mae"
            if mae_col in row and not np.isnan(row[mae_col]):
                lift_k = zshot - row[mae_col]
                row[f"fewshot_k{k}_lift"] = lift_k
                k_lifts[k] = lift_k

    if k_lifts:
        largest_k = max(k_lifts)
        row["all_context_lift"] = k_lifts[largest_k]
        ordered_lifts = [k_lifts[k] for k in sorted(k_lifts)]
        row["monotone_context"] = bool(
            all(
                b >= a - MONOTONE_CONTEXT_TOLERANCE
                for a, b in zip(ordered_lifts, ordered_lifts[1:])
            )
        )

    return row


CONTEXT_GATE_TOLERANCE: float = 0.01

MEAN_LIFT_FLOOR: float = 0.0

MONOTONE_CONTEXT_TOLERANCE: float = 0.01


def _assert_context_gate(
    rows: list[dict], tolerance: float = CONTEXT_GATE_TOLERANCE, mean_floor: float = MEAN_LIFT_FLOOR
) -> None:
    lifts = [r["lift"] for r in rows if "lift" in r and not pd.isna(r["lift"])]

    violations = [
        (r.get("axis"), r.get("group"), "lift", r["lift"])
        for r in rows
        if "lift" in r and not pd.isna(r["lift"]) and r["lift"] < -tolerance
    ]
    violations += [
        (r.get("axis"), r.get("group"), "all_context_lift", r["all_context_lift"])
        for r in rows
        if "all_context_lift" in r
        and not pd.isna(r["all_context_lift"])
        and r["all_context_lift"] < -tolerance
    ]
    if violations:
        detail = "; ".join(
            f"{axis}/{group} [{metric}]: lift={lift:+.4f}"
            for axis, group, metric, lift in violations
        )
        raise AssertionError(
            f"Context gate failed: few-shot scored worse than zero-shot by more than "
            f"{tolerance} log MAE on {len(violations)} group(s)/metric(s) -- {detail}"
        )

    if lifts:
        mean_lift = sum(lifts) / len(lifts)
        if mean_lift < mean_floor:
            raise AssertionError(
                f"Context gate failed: mean lift across {len(lifts)} group(s) is "
                f"{mean_lift:+.4f}, below the {mean_floor:+.4f} floor -- context is "
                f"hurting more often than it helps even though no single group "
                f"breached the per-group {tolerance} tolerance."
            )


def run_cnp_logo(
    df: pd.DataFrame,
    axis: str,
    base_work_dir: str,
    min_rows: int = 2,
    groups=None,
    shots=(0, 1, 2, 4, 8),
    n_repeats=5,
    max_epochs=500,
    patience=80,
    params=None,
    seed=0,
    keep_fold_dirs=False,
    enforce_context_gate: bool = True,
    ablate_protein_descriptors: bool = False,
    corrector_mode: str = "linear",
) -> pd.DataFrame:
    if ablate_protein_descriptors:
        df = zero_protein_descriptors(df)
    fold_groups = groups if groups is not None else build_groups(df, axis, min_rows=min_rows)
    rows = []
    os.makedirs(base_work_dir, exist_ok=True)
    for g in fold_groups:
        train_df, held_df = g.split(df)
        if held_df.empty or train_df.empty:
            continue
        fold_key = g.key.replace("/", "_").replace("=", "-")
        fold_dir = os.path.join(base_work_dir, f"{axis}__{fold_key}")
        os.makedirs(fold_dir, exist_ok=True)
        try:
            row = run_cnp_fold(
                train_df,
                held_df,
                g,
                fold_dir,
                shots=shots,
                n_repeats=n_repeats,
                max_epochs=max_epochs,
                patience=patience,
                params=params,
                seed=seed,
                corrector_mode=corrector_mode,
            )
        except Exception as e:  # keep the harness going across folds
            row = {"axis": g.axis, "group": g.key, "error": str(e)}
        rows.append(row)
        if not keep_fold_dirs:
            shutil.rmtree(fold_dir, ignore_errors=True)

    if enforce_context_gate:
        _assert_context_gate(rows)
    return pd.DataFrame(rows)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase 0 leave-one-GROUP-out scoreboard.")
    ap.add_argument(
        "--data",
        default=None,
        help=(
            "Master CSV/XLSX containing every protein/ingredient/class (NOT a pre-filtered "
            "training split). Defaults to the newest file in data/latest."
        ),
    )
    ap.add_argument("--axis", choices=AXES + ["all"], default="all")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to write the scoreboard. Defaults to a fresh <checkpoints>/<date>/<time> directory.",
    )
    ap.add_argument(
        "--min-rows",
        type=int,
        default=2,
        dest="min_rows",
        help="Minimum held-out rows for a group to be evaluated.",
    )
    ap.add_argument(
        "--groups",
        default=None,
        help="Comma-separated subset of group keys to run (e.g. 'ibalizumab,adalimumab'). Default: every group.",
    )
    ap.add_argument(
        "--shots", default="0,1,2,4,8", help="Comma-separated context sizes ('0' = zero-shot)."
    )
    ap.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        dest="n_repeats",
        help="Random context draws averaged per shot count.",
    )
    ap.add_argument("--max-epochs", type=int, default=500, dest="max_epochs")
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        dest="baseline_only",
        help="Skip CNP training (fast; baseline reference only).",
    )
    ap.add_argument(
        "--cnp-only", action="store_true", dest="cnp_only", help="Skip the baseline regressor."
    )
    ap.add_argument(
        "--keep-fold-dirs",
        action="store_true",
        dest="keep_fold_dirs",
        help="Keep each fold's trained checkpoint on disk.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test preset: max-epochs=30, patience=8, n-repeats=2, shots=0,1,4.",
    )
    ap.add_argument(
        "--no-context-gate",
        action="store_true",
        dest="no_context_gate",
        help=(
            "Disable the hard context-gate assertion (few-shot must not score worse than "
            "zero-shot by more than CONTEXT_GATE_TOLERANCE). Undertrained/--quick "
            "runs can trip it on noise alone; leave the gate ON for a real scoreboard."
        ),
    )
    ap.add_argument(
        "--ablate-protein-descriptors",
        action="store_true",
        dest="ablate_protein_descriptors",
        help=(
            "P0 descriptor-vs-context test: null out Charge/ProtPi PI/PI_mean dataset-wide "
            "before training (zero_protein_descriptors), so zero-shot "
            "has no per-protein identity handle and any few-shot lift must come from context."
        ),
    )
    ap.add_argument(
        "--corrector-mode",
        dest="corrector_mode",
        choices=["linear", "kernel", "offset_only"],
        default="linear",
        help=(
            "Which few-shot corrector the predictor uses. 'linear' is Task 1.1's default; "
            "'kernel' opts into Task 1.2's kernel-weighted local residual model."
        ),
    )
    return ap.parse_args(argv)


def run(
    data=None,
    axis="all",
    out_dir=None,
    min_rows=2,
    groups=None,
    shots="0,1,2,4,8",
    n_repeats=5,
    max_epochs=500,
    patience=80,
    seed=0,
    baseline_only=False,
    cnp_only=False,
    keep_fold_dirs=False,
    quick=False,
    no_context_gate=False,
    ablate_protein_descriptors=False,
    corrector_mode="linear",
):
    if quick:
        max_epochs = min(max_epochs, 30)
        patience = min(patience, 8)
        n_repeats = min(n_repeats, 2)
        shots = "0,1,4"

    if data is None:
        data = paths.latest_data_file()
    if not os.path.exists(data):
        raise SystemExit(f"ERROR: data not found -- {data}")
    if out_dir is None:
        out_dir = paths.dated_run_dir(constants.CHECKPOINTS_DIR)

    shot_values = tuple(int(s) for s in str(shots).split(","))
    axes = AXES if axis == "all" else [axis]
    group_filter = set(groups.split(",")) if groups else None

    df = prepare_df(paths.load_table(data, index_col=False))
    logger.info(f"Loaded {len(df)} rows from {data}")

    os.makedirs(out_dir, exist_ok=True)
    all_scoreboards = []

    for ax in axes:
        fold_groups = build_groups(df, ax, min_rows=min_rows)
        if group_filter is not None:
            fold_groups = [g for g in fold_groups if g.key in group_filter]
        if not fold_groups:
            logger.warning(f"No groups found for axis='{ax}' (after filtering); skipping.")
            continue
        logger.info(
            f"\n{'='*70}\nAXIS: {ax}  ({len(fold_groups)} held-out group(s): {[g.key for g in fold_groups]})\n{'='*70}"
        )

        baseline_df = pd.DataFrame()
        if not cnp_only:
            logger.info(f"[{ax}] Running reference baseline (HistGBM, feature-only)...")
            baseline_df = run_baseline_logo(df, ax, groups=fold_groups)
            baseline_df = baseline_df.add_prefix("baseline_").rename(
                columns={"baseline_axis": "axis", "baseline_group": "group"}
            )

        cnp_df = pd.DataFrame()
        if not baseline_only:
            logger.info(f"[{ax}] Running CNP LOGO fold(s) (this trains one model per group)...")
            work_dir = os.path.join(out_dir, f"work_{ax}")
            cnp_df = run_cnp_logo(
                df,
                ax,
                work_dir,
                groups=fold_groups,
                shots=shot_values,
                n_repeats=n_repeats,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
                keep_fold_dirs=keep_fold_dirs,
                enforce_context_gate=not no_context_gate,
                ablate_protein_descriptors=ablate_protein_descriptors,
                corrector_mode=corrector_mode,
            )
            if not keep_fold_dirs:
                shutil.rmtree(work_dir, ignore_errors=True)

        if not baseline_df.empty and not cnp_df.empty:
            merged = pd.merge(baseline_df, cnp_df, on=["axis", "group"], how="outer")
        elif not baseline_df.empty:
            merged = baseline_df
        else:
            merged = cnp_df
        all_scoreboards.append(merged)

    if not all_scoreboards:
        raise SystemExit("No results produced -- check --axis/--groups/--min-rows.")

    scoreboard = pd.concat(all_scoreboards, ignore_index=True)

    if "baseline_log_mae" in scoreboard.columns and "zero_shot_log_mae" in scoreboard.columns:
        scoreboard["zero_shot_meets_baseline"] = (
            scoreboard["zero_shot_log_mae"] <= scoreboard["baseline_log_mae"]
        )
    fewshot_cols = [
        c for c in scoreboard.columns if c.startswith("fewshot_k") and c.endswith("_log_mae")
    ]
    if "baseline_log_mae" in scoreboard.columns and fewshot_cols:
        best_fewshot = scoreboard[fewshot_cols].min(axis=1)
        scoreboard["best_fewshot_beats_baseline"] = best_fewshot < scoreboard["baseline_log_mae"]

    csv_path = os.path.join(out_dir, "logo_scoreboard.csv")
    scoreboard.to_csv(csv_path, index=False)
    logger.info(f"\nScoreboard saved: {csv_path}")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("\n" + scoreboard.to_string(index=False))

    if "zero_shot_meets_baseline" in scoreboard.columns:
        n_ok = scoreboard["zero_shot_meets_baseline"].sum()
        n_total = scoreboard["zero_shot_meets_baseline"].notna().sum()
        logger.info(f"\nZero-shot <= baseline on {n_ok}/{n_total} held-out groups.")
    if "best_fewshot_beats_baseline" in scoreboard.columns:
        n_ok = scoreboard["best_fewshot_beats_baseline"].sum()
        n_total = scoreboard["best_fewshot_beats_baseline"].notna().sum()
        logger.info(f"Best few-shot beats baseline on {n_ok}/{n_total} held-out groups.")
    if "lift" in scoreboard.columns:
        lift = scoreboard["lift"].dropna()
        if not lift.empty:
            n_negative = int((lift < 0).sum())
            logger.info(
                f"\nContext lift (zero_shot_log_mae - best_fewshot_log_mae): "
                f"mean={lift.mean():+.4f}, min={lift.min():+.4f}, "
                f"positive-or-flat on {len(lift) - n_negative}/{len(lift)} groups."
            )
            if n_negative:
                worst = scoreboard.loc[scoreboard["lift"].notna()].nsmallest(
                    min(5, n_negative), "lift"
                )
                logger.info(
                    "Worst lift groups:\n"
                    + worst[
                        ["axis", "group", "zero_shot_log_mae", "best_fewshot_log_mae", "lift"]
                    ].to_string(index=False)
                )
    if "ablation_delta" in scoreboard.columns:
        ing = scoreboard[scoreboard["axis"] == "ingredient"]
        if not ing.empty:
            n_helps = (ing["ablation_delta"] > 0).sum()
            n_total = ing["ablation_delta"].notna().sum()
            logger.info(
                f"Property vector beats zeroed-ingredient ablation on {n_helps}/{n_total} "
                "held-out ingredient groups (positive ablation_delta = real properties win)."
            )

    logger.info("Done.")
    return scoreboard


def main(argv=None):
    args = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = paths.dated_run_dir(constants.CHECKPOINTS_DIR)
    configure_logging(log_dir=args.out_dir)
    return run(**vars(args))


if __name__ == "__main__":
    main()
