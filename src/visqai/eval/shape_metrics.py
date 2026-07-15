"""
shape_metrics.py
================
Shape-fidelity metrics for viscosity shear-rate profiles: how well a predicted
5-point shear profile reproduces the PATTERN of the actual curve (thinning
direction, plateau shape, total thinning magnitude), independent of level
offset. Moved verbatim from ml/cnp_mk2/learning_curve_ibal.py — the only file
where this suite existed; no logic changes.
"""

from __future__ import annotations

import numpy as np

# Per-segment slope below this magnitude (log10 units across a shear decade-step)
# is treated as "flat" for slope-sign agreement. ~0.005 ≈ 1% change.
SHAPE_FLAT_EPS = 0.005


def _classify_slopes(log_visc: np.ndarray, eps: float = SHAPE_FLAT_EPS) -> np.ndarray:
    """Per-segment direction: -1 thinning, 0 flat, +1 thickening."""
    d = np.diff(log_visc)
    out = np.zeros_like(d)
    out[d < -eps] = -1.0
    out[d > eps] = 1.0
    return out


def compute_shape_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """
    Shape-fidelity metrics for one profile (actual vs predicted, linear cP).

    Returns
    -------
    shape_rmse_log10 : float
        Level-invariant shape error. Each curve's mean log10-viscosity is
        subtracted before comparison, so only the PATTERN of variation across
        shear is scored. This is the headline shape metric — it is small only
        when the predicted curve has the same ups/downs/plateaus as the truth,
        regardless of vertical offset.
    slope_sign_agree : float
        Fraction of SLOPED actual segments where predicted and actual thinning
        DIRECTION match (down/up). Segments where the ACTUAL is exactly flat are
        EXCLUDED — on this instrument ~46% of segments report identical values at
        adjacent shear rates due to 2–3 sig-fig quantization, and the model
        predicting a small physical decline there is not a shape error (it may be
        more physically correct than the quantized-flat reading). Including those
        segments artificially caps the metric near ~50%. This version measures
        only the segments where the truth genuinely moves, so 1.0 = the model got
        the thinning direction right on every segment that actually thins. NaN if
        no actual segment is sloped (the whole curve is a quantized plateau).
    slope_n_sloped : int
        Number of actual segments that were sloped (the denominator of
        slope_sign_agree). Low values mean the sample is mostly quantized and the
        agreement number is based on few segments — read it with that caveat.
    plateau_err_log10 : float
        |actual low-shear drop − predicted low-shear drop| over 100→10k s⁻¹.
        Targets the specific failure here: reproducing (or failing to reproduce)
        the Newtonian low-shear plateau before high-shear roll-off. 0 = the model
        matched the plateau flatness; large = it sloped where the truth was flat.
        NOTE: also partly quantization-sensitive — trend is informative, absolute
        value is not. Prefer shape_rmse_log10 / thin_ratio_log_err as the clean
        shape signals on this data.
    thin_ratio_log_err : float
        |log10(actual η_low/η_high) − log10(pred η_low/η_high)|. Total thinning
        magnitude mismatch across the full shear range, in log space. This is the
        CLEANEST shape signal for the over-thinning failure mode — it is not
        segment-level and so is immune to per-segment quantization plateaus.
    Any metric is NaN if fewer than 3 (or, for ratios, 2) valid points exist.
    """
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p) & (a > 0) & (p > 0)
    nan = {
        "shape_rmse_log10": np.nan,
        "slope_sign_agree": np.nan,
        "slope_n_sloped": 0,
        "plateau_err_log10": np.nan,
        "thin_ratio_log_err": np.nan,
    }
    if m.sum() < 2:
        return nan

    la = np.log10(a[m])
    lp = np.log10(p[m])

    # Total thinning magnitude (needs ≥2 points).
    thin_ratio_log_err = float(abs((la[0] - la[-1]) - (lp[0] - lp[-1])))

    if m.sum() < 3:
        return {**nan, "thin_ratio_log_err": thin_ratio_log_err}

    # Level-invariant shape RMSE: remove each curve's mean.
    shape_rmse = float(np.sqrt(np.mean(((la - la.mean()) - (lp - lp.mean())) ** 2)))

    # Slope-sign agreement across segments — EXCLUDING quantization-flat actual
    # segments. A segment counts only if the ACTUAL moved (|Δlog10| >= eps); the
    # instrument's 2-3 sig-fig resolution makes ~46% of segments exactly flat, and
    # penalising the model for predicting a small slope there measures quantization
    # noise, not morphology. We score the model's direction (sign of its slope)
    # against the actual direction on the sloped segments only.
    da = np.diff(la)
    dp = np.diff(lp)
    sloped = np.abs(da) >= SHAPE_FLAT_EPS  # only segments where truth genuinely moves
    n_sloped = int(sloped.sum())
    if n_sloped > 0:
        # Direction match on sloped segments: do model and actual share sign?
        slope_sign_agree = float((np.sign(da[sloped]) == np.sign(dp[sloped])).mean())
    else:
        slope_sign_agree = np.nan  # entire curve is a quantized plateau

    # Plateau error over the low-shear region (first to third valid point).
    # When all 5 rates are present these are 100, 1k, 10k s⁻¹.
    plateau_err = float(abs((la[0] - la[2]) - (lp[0] - lp[2])))

    return {
        "shape_rmse_log10": shape_rmse,
        "slope_sign_agree": slope_sign_agree,
        "slope_n_sloped": n_sloped,
        "plateau_err_log10": plateau_err,
        "thin_ratio_log_err": thin_ratio_log_err,
    }


def _aggregate_shape(profs: dict) -> dict:
    """
    Aggregate per-sample shape metrics over a set of profiles.

    shape_rmse_log10 / plateau_err_log10 / thin_ratio_log_err are simple means
    over samples with finite values. slope_sign_agree is SEGMENT-weighted (each
    sample's score weighted by its sloped-segment count) so the aggregate equals
    the true fraction of correctly-directed sloped segments across the set, and
    samples that are mostly quantization plateaus (few sloped segments) do not
    dominate. Returns NaNs when no usable data.
    """
    agg = {
        "shape_rmse_log10": np.nan,
        "slope_sign_agree": np.nan,
        "plateau_err_log10": np.nan,
        "thin_ratio_log_err": np.nan,
    }
    if not profs:
        return agg
    for key in ["shape_rmse_log10", "plateau_err_log10", "thin_ratio_log_err"]:
        vals = [p[key] for p in profs.values() if np.isfinite(p.get(key, np.nan))]
        if vals:
            agg[key] = float(np.mean(vals))
    num = den = 0.0
    for p in profs.values():
        ns = p.get("slope_n_sloped", 0)
        sa = p.get("slope_sign_agree", np.nan)
        if ns > 0 and np.isfinite(sa):
            num += sa * ns
            den += ns
    if den > 0:
        agg["slope_sign_agree"] = float(num / den)
    return agg
