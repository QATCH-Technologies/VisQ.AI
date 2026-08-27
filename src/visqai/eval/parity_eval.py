from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from visqai import constants, paths
from visqai.constants import (
    SHEAR_COLS,
    SHEAR_LABELS,
    SHEAR_COLORS,
    SHEAR_RATES,
    N_SHEARS,
    C_DEEP_BLUE,
    C_ACCENT,
    C_TEXT,
    C_BORDER,
    C_WHITE,
    C_CONTEXT,
)
from visqai.eval.metrics import calc_metrics
from visqai.eval.style import mpl, apply_style
from visqai.features.dataprocessor import prepare_df
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger("ParityEval")

PROTEIN_KEY = "Ibalizumab"


def preprocess_pool(predictor, iba_df):
    static_t, shear_t, visc_t = predictor._preprocess(iba_df)
    ctx_t = torch.cat([shear_t, visc_t, static_t], dim=-1)
    return ctx_t, static_t, shear_t, visc_t


def _held_out_errors(model, ctx_t, held_out_idx, all_static_t, all_shear_t, all_visc_t):
    with torch.no_grad():
        memory = model.encode_memory(ctx_t)
        errs = []
        for j in held_out_idx:
            lo, hi = j * N_SHEARS, (j + 1) * N_SHEARS
            pred = model.decode_from_memory(
                memory, all_shear_t[:, lo:hi, :], all_static_t[:, lo:hi, :]
            )
            truth = all_visc_t[:, lo:hi, :]
            rmse = torch.sqrt(((pred - truth) ** 2).mean()).item()
            errs.append(rmse)
    return np.asarray(errs, dtype=float)


def _objective(errs: np.ndarray, mode: str) -> float:
    if errs.size == 0:
        return 0.0
    if mode == "mean":
        return float(errs.mean())
    if mode == "max":
        return float(errs.max())
    if mode == "tail":
        return float(0.5 * errs.mean() + 0.5 * np.percentile(errs, 90))
    raise ValueError(f"Unknown objective '{mode}'")


def _ctx_indices(sample_indices):
    out = []
    for s in sample_indices:
        out.extend(range(s * N_SHEARS, (s + 1) * N_SHEARS))
    return out


def _score_set(
    model, all_ctx_t, selected, n, pool, objective, all_static_t, all_shear_t, all_visc_t
):
    held = [i for i in pool if i not in selected]
    if not held:
        return 0.0
    ctx_t = all_ctx_t[:, _ctx_indices(selected), :]
    errs = _held_out_errors(model, ctx_t, held, all_static_t, all_shear_t, all_visc_t)
    return _objective(errs, objective)


def greedy_select(predictor, iba_df, n_select, objective="tail", refine=True, verbose=True):
    model = predictor.model
    model.eval()

    n = len(iba_df)
    if n_select > n:
        raise ValueError(f"n_select={n_select} > n_pool={n}")
    pool = list(range(n))

    if verbose:
        logger.info(f"[Preprocess] {n} samples (objective='{objective}', refine={refine}) …")
    t0 = time.perf_counter()
    all_ctx_t, all_static_t, all_shear_t, all_visc_t = preprocess_pool(predictor, iba_df)
    if verbose:
        logger.info(f"  done in {time.perf_counter()-t0:.1f}s, ctx {tuple(all_ctx_t.shape)}")

    selected = []
    step_log = []
    prev = None
    for step in range(n_select):
        remaining = [i for i in pool if i not in selected]
        best_idx, best_score = None, float("inf")
        for cand in remaining:
            score = _score_set(
                model,
                all_ctx_t,
                selected + [cand],
                n,
                pool,
                objective,
                all_static_t,
                all_shear_t,
                all_visc_t,
            )
            if score < best_score - 1e-12:
                best_idx, best_score = cand, score
        selected.append(best_idx)
        imp = (prev - best_score) if prev is not None else float("nan")
        prev = best_score
        sid = iba_df.iloc[best_idx]["ID"]
        step_log.append(
            dict(
                step=step + 1, sample_idx=best_idx, sample_id=sid, score=best_score, improvement=imp
            )
        )
        if verbose:
            istr = f"  Δ={imp:+.5f}" if np.isfinite(imp) else ""
            logger.info(f"[{step+1}/{n_select}] +{sid:>6} | score={best_score:.5f}{istr}")

    if refine:
        selected, swaps = _swap_refine(
            model,
            all_ctx_t,
            selected,
            pool,
            objective,
            all_static_t,
            all_shear_t,
            all_visc_t,
            iba_df,
            verbose,
        )
        if verbose:
            logger.info(f"[Refine] {swaps} swap(s) improved the set.")

    final_score = _score_set(
        model, all_ctx_t, selected, n, pool, objective, all_static_t, all_shear_t, all_visc_t
    )
    return selected, step_log, final_score


def _swap_refine(
    model,
    all_ctx_t,
    selected,
    pool,
    objective,
    all_static_t,
    all_shear_t,
    all_visc_t,
    iba_df,
    verbose,
    max_passes=3,
):
    selected = list(selected)
    cur = _score_set(
        model,
        all_ctx_t,
        selected,
        len(pool),
        pool,
        objective,
        all_static_t,
        all_shear_t,
        all_visc_t,
    )
    total_swaps = 0
    for _pass in range(max_passes):
        improved = False
        for si in range(len(selected)):
            non_members = [i for i in pool if i not in selected]
            best_repl, best_score = None, cur
            for cand in non_members:
                trial = list(selected)
                trial[si] = cand
                score = _score_set(
                    model,
                    all_ctx_t,
                    trial,
                    len(pool),
                    pool,
                    objective,
                    all_static_t,
                    all_shear_t,
                    all_visc_t,
                )
                if score < best_score - 1e-9:
                    best_repl, best_score = cand, score
            if best_repl is not None:
                old = selected[si]
                selected[si] = best_repl
                cur = best_score
                total_swaps += 1
                improved = True
                if verbose:
                    logger.info(
                        f"    swap {iba_df.iloc[old]['ID']} -> {iba_df.iloc[best_repl]['ID']} | score={cur:.5f}"
                    )
        if not improved:
            break
    return selected, total_swaps


def build_long(pred_df, is_context):
    rows = []
    for _, row in pred_df.iterrows():
        for sc in SHEAR_COLS:
            act = row.get(sc, np.nan)
            prd = row.get(f"Pred_{sc}", np.nan)
            valid = pd.notna(act) and pd.notna(prd) and act > 0 and prd > 0
            rows.append(
                {
                    "ID": row["ID"],
                    "shear_col": sc,
                    "shear_label": SHEAR_LABELS[sc],
                    "actual_cP": act,
                    "pred_cP": prd,
                    "is_context": is_context,
                    "Protein_conc": row.get("Protein_conc", np.nan),
                    "Buffer_pH": row.get("Buffer_pH", np.nan),
                    "log10_error": (np.log10(prd) - np.log10(act)) if valid else np.nan,
                    "fold_error": (prd / act) if valid else np.nan,
                }
            )
    return rows


def make_parity_plot(long_df, shear_subset, title, out_path, single_shear=False, context_ids=None):
    plt, ticker, Line2D = mpl()
    apply_style(plt)

    sub = long_df[long_df["shear_col"].isin(shear_subset)].copy()
    sub = sub[(sub["actual_cP"] > 0) & (sub["pred_cP"] > 0)].dropna(subset=["actual_cP", "pred_cP"])
    if sub.empty:
        logger.warning(f"No valid data for {out_path} - skipping.")
        return

    m = calc_metrics(sub["actual_cP"].values, sub["pred_cP"].values)
    all_vals = np.concatenate([sub["actual_cP"].values, sub["pred_cP"].values])
    all_vals = all_vals[all_vals > 0]
    log_min, log_max = np.log10(all_vals.min()), np.log10(all_vals.max())
    pad = (log_max - log_min) * 0.04
    lo, hi = 10 ** (log_min - pad), 10 ** (log_max + pad)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)
    ax.plot(
        np.linspace(lo, hi, 400),
        np.linspace(lo, hi, 400),
        color=C_DEEP_BLUE,
        lw=1.8,
        ls="--",
        zorder=3,
    )

    ctx_set = set(context_ids) if context_ids is not None else set()
    for sc in shear_subset:
        mask = sub["shear_col"] == sc
        if not mask.any():
            continue
        held = mask & ~sub["ID"].isin(ctx_set)
        if held.any():
            ax.scatter(
                sub.loc[held, "actual_cP"],
                sub.loc[held, "pred_cP"],
                color=SHEAR_COLORS[sc],
                s=62,
                zorder=5,
                alpha=0.88,
                edgecolors=C_WHITE,
                linewidths=0.9,
            )
        ctx = mask & sub["ID"].isin(ctx_set)
        if ctx.any():
            ax.scatter(
                sub.loc[ctx, "actual_cP"],
                sub.loc[ctx, "pred_cP"],
                color=C_CONTEXT,
                s=80,
                zorder=6,
                alpha=0.92,
                edgecolors=C_WHITE,
                linewidths=0.9,
                marker="D",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="major", zorder=0, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Measured Viscosity (cP)", fontsize=15, labelpad=12, color=C_TEXT)
    ax.set_ylabel("Predicted Viscosity (cP)", fontsize=15, labelpad=12, color=C_TEXT)

    metrics_text = (
        f"MAE   {m['mae']:.2f} cP\nRMSE  {m['rmse']:.2f} cP\n"
        f"R²    {m['r2']:.3f}\n<=2x   {m['within_2x']:.0f}%   (n={m['n']})"
    )
    ax.text(
        0.04,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        ha="left",
        color=C_TEXT,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor=C_WHITE,
            edgecolor=C_BORDER,
            linewidth=0.8,
            alpha=0.93,
        ),
    )

    parity_handle = Line2D([0], [0], color=C_DEEP_BLUE, lw=1.8, ls="--")
    context_handle = Line2D(
        [0], [0], marker="D", color="w", markerfacecolor=C_CONTEXT, markersize=8
    )
    if single_shear:
        handles = [parity_handle] + ([context_handle] if ctx_set else [])
        labels = ["Perfect parity"] + (["Context"] if ctx_set else [])
    else:
        shear_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SHEAR_COLORS[sc], markersize=8)
            for sc in shear_subset
        ]
        handles = [parity_handle] + shear_handles + ([context_handle] if ctx_set else [])
        labels = (
            ["Perfect parity"]
            + [SHEAR_LABELS[sc] for sc in shear_subset]
            + (["Context"] if ctx_set else [])
        )
    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        fontsize=11,
        framealpha=0.92,
        edgecolor=C_BORDER,
        borderpad=0.9,
        handlelength=2.0,
    )

    ax.set_title(title, fontsize=16, pad=14, color=C_TEXT, loc="left", fontweight="semibold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  saved: {out_path}")


def make_profile_plot(results_df, sample_id, out_path):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    row = results_df[results_df["ID"] == str(sample_id)]
    if row.empty:
        logger.warning(f"Profile: ID '{sample_id}' not found - skipping.")
        return
    row = row.iloc[0]
    measured = [row.get(sc, np.nan) for sc in SHEAR_COLS]
    predicted = [row.get(f"Pred_{sc}", np.nan) for sc in SHEAR_COLS]

    conc, ph = row.get("Protein_conc", "?"), row.get("Buffer_pH", "?")
    parts = []
    for key in ("Salt_type", "Stabilizer_type", "Surfactant_type"):
        v = row.get(key, "none")
        if str(v).lower() not in ("none", "nan", ""):
            parts.append(str(v))
    subtitle = f"{conc} mg/mL  |  pH {ph}  |  {', '.join(parts) if parts else 'no excipients'}"

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    valid = [i for i in range(N_SHEARS) if pd.notna(measured[i]) and pd.notna(predicted[i])]
    if valid:
        xs = [SHEAR_RATES[i] for i in valid]
        ax.fill_between(
            xs,
            [measured[i] for i in valid],
            [predicted[i] for i in valid],
            color=C_DEEP_BLUE,
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )
    ax.plot(
        SHEAR_RATES,
        measured,
        color=C_DEEP_BLUE,
        lw=2.2,
        marker="o",
        markersize=7,
        markeredgecolor=C_WHITE,
        markeredgewidth=0.8,
        zorder=4,
        label="Measured",
    )
    ax.plot(
        SHEAR_RATES,
        predicted,
        color=C_ACCENT,
        lw=2.2,
        ls="--",
        marker="s",
        markersize=7,
        markeredgecolor=C_WHITE,
        markeredgewidth=0.8,
        zorder=4,
        label="Predicted",
    )
    ax.set_xscale("log")
    yvals = [v for v in measured + predicted if pd.notna(v)]
    ax.set_ylim(0, max(yvals) * 1.15 if yvals else 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, which="major", zorder=0, linewidth=0.6)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Shear Rate (s⁻¹)", fontsize=14, labelpad=10, color=C_TEXT)
    ax.set_ylabel("Viscosity (cP)", fontsize=14, labelpad=10, color=C_TEXT)
    ax.set_title(
        f"Ibalizumab - Viscosity Profile  (ID {sample_id})\n{subtitle}",
        fontsize=14,
        pad=12,
        color=C_TEXT,
        loc="left",
        fontweight="semibold",
    )
    ax.legend(
        loc="upper right",
        fontsize=12,
        framealpha=0.92,
        edgecolor=C_BORDER,
        borderpad=0.8,
        handlelength=2.0,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  saved: {out_path}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Combined Ibalizumab CNP context select + evaluate.")
    ap.add_argument(
        "--model_dir",
        default=None,
        help="Trained checkpoint directory to evaluate. Defaults to the most recently produced checkpoint.",
    )
    ap.add_argument(
        "--data",
        default=None,
        help="Master formulation CSV/XLSX to filter down to --protein_key. Defaults to the newest file in data/latest.",
    )
    ap.add_argument("--n_select", type=int, default=8)
    ap.add_argument(
        "--objective",
        choices=["mean", "tail", "max"],
        default="tail",
        help="Selection objective. 'tail'/'max' demonstrate range coverage.",
    )
    ap.add_argument(
        "--refine",
        action="store_true",
        default=True,
        help="Run swap-refinement after greedy (default on).",
    )
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--protein_key", default=PROTEIN_KEY)
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to write eval results. Defaults to <model_dir>/benchmarks.",
    )
    ap.add_argument(
        "--context_csv",
        default=None,
        help="Skip selection; load context IDs from this CSV (col ID/cnp_rank).",
    )
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap.parse_args(argv)


def run(
    model_dir=None,
    data=None,
    n_select=8,
    objective="tail",
    refine=True,
    protein_key=PROTEIN_KEY,
    out_dir=None,
    context_csv=None,
    plot=True,
):
    if model_dir is None:
        model_dir = paths.latest_checkpoint_dir(constants.CHECKPOINTS_DIR)
    if data is None:
        data = paths.latest_data_file()
    for path in (model_dir, data):
        if not os.path.exists(path):
            sys.exit(f"ERROR: path not found - {path}")
    if out_dir is None:
        out_dir = os.path.join(model_dir, "benchmarks")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    df = prepare_df(paths.load_table(data, index_col=False))
    iba_df = (
        df[df["Protein_type"].astype(str).str.lower() == protein_key.lower()]
        .copy()
        .reset_index(drop=True)
    )
    n_total = len(iba_df)
    logger.info(f"{protein_key} samples: {n_total}")
    if n_total < n_select:
        sys.exit(f"ERROR: only {n_total} samples, cannot select {n_select}.")

    logger.info(f"Loading model: {model_dir}")
    predictor = ViscosityPredictorCNP(model_dir, verbose=False)
    logger.info(
        f"  static_dim={predictor.static_dim}, hidden_dim={predictor.config['hidden_dim']}, latent_dim={predictor.config['latent_dim']}"
    )

    if context_csv and os.path.exists(context_csv):
        meta = prepare_df(paths.load_table(context_csv))
        rank_col = (
            "cnp_rank"
            if "cnp_rank" in meta.columns
            else ("rank" if "rank" in meta.columns else None)
        )
        if rank_col:
            meta = meta.sort_values(rank_col)
        context_ids = meta["ID"].astype(str).head(n_select).tolist()
        logger.info(f"Loaded {len(context_ids)} context IDs from {context_csv}")
    else:
        t0 = time.perf_counter()
        selected_idx, step_log, final_score = greedy_select(
            predictor,
            iba_df,
            n_select=n_select,
            objective=objective,
            refine=refine,
            verbose=True,
        )
        context_ids = [str(iba_df.iloc[i]["ID"]) for i in selected_idx]
        logger.info(
            f"Selection complete in {time.perf_counter()-t0:.1f}s (final {objective} score={final_score:.5f})"
        )

        sel_path = os.path.join(out_dir, "context_selection.csv")
        full_sel = df[df["ID"].isin(context_ids)].copy()
        rank_map = {sid: k + 1 for k, sid in enumerate(context_ids)}
        full_sel["cnp_rank"] = full_sel["ID"].map(rank_map)
        full_sel.sort_values("cnp_rank").to_csv(sel_path, index=False)
        logger.info(f"Context selection saved: {sel_path}")

    context_id_set = set(context_ids)
    n_ctx = len(context_id_set)
    context_df = iba_df[iba_df["ID"].isin(context_id_set)].copy()
    held_out_df = iba_df[~iba_df["ID"].isin(context_id_set)].copy().reset_index(drop=True)
    n_held = len(held_out_df)
    logger.info(f"Context: {n_ctx} | Held-out: {n_held}")

    predictor.memory_vector = None
    predictor.context_t = None
    predictor.learn(context_df)

    results_df = predictor.predict(held_out_df)
    context_pred_df = predictor.predict(context_df)

    long_df = pd.DataFrame(
        build_long(results_df, is_context=False) + build_long(context_pred_df, is_context=True)
    )
    csv_path = os.path.join(out_dir, "ibalizumab_parity_results.csv")
    long_df.to_csv(csv_path, index=False)
    logger.info(f"Parity results saved: {csv_path}")

    held_long = long_df[~long_df["is_context"]]
    logger.info("\n" + "=" * 65)
    logger.info(f"PER-SHEAR-RATE SUMMARY  (held-out {n_held} samples, context excluded)")
    logger.info("=" * 65)
    logger.info(
        f"{'Shear Rate':>18}  {'N':>4}  {'MAE log10':>10}  {'RMSE log10':>10}  {'Bias':>8}  {'<=2x%':>7}"
    )
    logger.info("-" * 65)
    for sc in SHEAR_COLS:
        sub = held_long[(held_long["shear_col"] == sc) & held_long["log10_error"].notna()]
        if sub.empty:
            continue
        m = calc_metrics(sub["actual_cP"], sub["pred_cP"])
        logger.info(
            f"{sub['shear_label'].iloc[0]:>18}  {m['n']:>4}  {m['log_mae']:>10.4f}  "
            f"{m['log_rmse']:>10.4f}  {m['log_bias']:>+8.4f}  {m['within_2x']:>6.0f}%"
        )
    logger.info("-" * 65)
    valid_all = held_long[(held_long["actual_cP"] > 0) & (held_long["pred_cP"] > 0)]
    m_all = calc_metrics(valid_all["actual_cP"], valid_all["pred_cP"])
    logger.info(
        f"{'All shear rates':>18}  {m_all['n']:>4}  {m_all['log_mae']:>10.4f}  "
        f"{m_all['log_rmse']:>10.4f}  {m_all['log_bias']:>+8.4f}  {m_all['within_2x']:>6.0f}%"
    )
    logger.info("=" * 65)

    if not plot:
        logger.info("Plots suppressed.")
        return long_df

    logger.info("Generating plots …")
    subtitle = f"{n_ctx} context  |  {n_held} held-out"
    make_parity_plot(
        long_df,
        SHEAR_COLS,
        f"Ibalizumab - All Shear Rates\n{subtitle}",
        os.path.join(out_dir, "parity_ibal_all_shears.png"),
        single_shear=False,
        context_ids=context_id_set,
    )
    make_parity_plot(
        long_df,
        ["Viscosity_1000"],
        f"Viscosity @ 1 000 s⁻¹ - Ibalizumab\n{subtitle}",
        os.path.join(out_dir, "parity_ibal_1000.png"),
        single_shear=True,
        context_ids=context_id_set,
    )

    v1000 = pd.to_numeric(held_out_df["Viscosity_1000"], errors="coerce").fillna(0)
    if v1000.max() > 0:
        pid = str(held_out_df.loc[v1000.idxmax(), "ID"])
        prof = predictor.predict(held_out_df[held_out_df["ID"] == pid].copy())
        make_profile_plot(prof, pid, os.path.join(out_dir, f"profile_ibal_{pid}.png"))

    logger.info("Done.")
    return long_df


def main(argv=None):
    args = parse_args(argv)
    if args.model_dir is None:
        args.model_dir = paths.latest_checkpoint_dir(constants.CHECKPOINTS_DIR)
    if args.out_dir is None:
        args.out_dir = os.path.join(args.model_dir, "benchmarks")
    configure_logging(log_dir=args.out_dir)
    return run(**vars(args))


if __name__ == "__main__":
    main()
