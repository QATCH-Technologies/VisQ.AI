"""
test_coverage.py
=================
Task 8 tests for visqai.analysis.coverage.

Tests 1-4 and 6 use small synthetic frames so they exercise the counting
primitives (count_bin, kish_design_effect, bin_series) directly rather than
the dataset-pinned census wrappers (class_a_marginal etc. assert their
cluster counts sum to coverage.N_CLUSTERS_TOTAL == 12, which is specific to
the real CSV and would fail by construction on a synthetic frame with a
different cluster count -- that assertion is itself covered by test 5's
round-trip against the real data).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from visqai.analysis import coverage as cov

REAL_CSV = cov.DEFAULT_CSV_PATH


def _synthetic_single_protein_frame(n_rows: int = 500) -> pd.DataFrame:
    """500 rows, all the same non-placebo protein -- every class-A axis is
    trivially single-valued/single-cluster by construction."""
    return pd.DataFrame(
        {
            "Protein_type": ["synth_mab"] * n_rows,
            "MW": [150.0] * n_rows,
            "PI_mean": [7.5] * n_rows,
            "PI_range": [0.3] * n_rows,
            "kP": [3.0] * n_rows,
            "HCI": [1.0] * n_rows,
            "C_Class": [1.0] * n_rows,
            "Viscosity_1000": np.linspace(10.0, 200.0, n_rows),
        }
    )


def _synthetic_equal_clusters_frame(k: int) -> pd.DataFrame:
    """k clusters of size k each (n = k^2 rows) -- the specific construction
    under which Kish deff(icc=1) reduces exactly to k (see coverage.py's
    kish_design_effect docstring: deff(icc=1) = size-weighted mean cluster
    size, which equals k only when every cluster's size is itself k)."""
    rows = []
    for i in range(k):
        rows.extend([f"cluster_{i}"] * k)
    return pd.DataFrame({"Protein_type": rows})


# --- Test 1 -------------------------------------------------------------


def test_single_protein_synthetic_frame_bounds_n_clusters_and_n_eff():
    df = _synthetic_single_protein_frame(n_rows=500)
    for axis in cov.CLASS_A_AXES:
        binned = cov.bin_series(df[axis], cov.CLASS_A_AXES[axis])
        for label in binned.unique():
            if label == "NaN":
                continue
            bc = cov.count_bin(df, binned == label, scope=cov.SCOPE_BETWEEN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)
            assert bc.n_clusters <= 1
            # n_eff for a single cluster is bounded by 1/icc as its row
            # count grows (BinCount docstring) -- never anywhere near
            # n_rows=500.
            assert bc.n_eff <= (1.0 / cov.ICC_BETWEEN_PROTEIN) + 1e-6

    all_rows_bc = cov.count_bin(df, pd.Series(True, index=df.index), scope=cov.SCOPE_BETWEEN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)
    assert all_rows_bc.n_rows == 500
    assert all_rows_bc.n_clusters == 1
    assert all_rows_bc.n_eff < 3.0  # nowhere near 500 -- this is the whole point of the module


# --- Test 2 -------------------------------------------------------------


def test_kish_design_effect_icc_zero_and_icc_one():
    rng = np.random.default_rng(0)
    unequal_sizes = rng.integers(1, 50, size=9)
    assert cov.kish_design_effect(unequal_sizes, icc=0.0) == pytest.approx(1.0)

    k = 4  # k clusters of size k -> deff(icc=1) == k (see fixture docstring)
    df = _synthetic_equal_clusters_frame(k)
    sizes = df["Protein_type"].value_counts().to_numpy()
    assert cov.kish_design_effect(sizes, icc=1.0) == pytest.approx(k)
    assert cov.kish_design_effect(sizes, icc=0.0) == pytest.approx(1.0)


# --- Test 3 -------------------------------------------------------------


def test_count_bin_rejects_icc_under_within_protein_scope():
    df = _synthetic_single_protein_frame(n_rows=10)
    mask = pd.Series(True, index=df.index)
    with pytest.raises(ValueError, match="within_protein"):
        cov.count_bin(df, mask, scope=cov.SCOPE_WITHIN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)
    # scope='within_protein' with icc omitted is fine.
    bc = cov.count_bin(df, mask, scope=cov.SCOPE_WITHIN_PROTEIN)
    assert bc.scope == cov.SCOPE_WITHIN_PROTEIN


# --- Test 4 -------------------------------------------------------------


def test_logo_residual_leverage_missing_artifact_raises_not_falls_back():
    df = _synthetic_single_protein_frame(n_rows=20)
    with pytest.raises(FileNotFoundError):
        cov.build_gap_report(df, leverage="logo_residual", logo_residual_path="does/not/exist.csv")


# --- Test 5 -------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real dataset not present in this checkout")
def test_class_b_census_reproduces_task_0_cluster_counts_on_real_csv():
    df = cov.load_dataset(REAL_CSV)
    assert len(df) == 520

    fv = cov.class_b_fv_pseudo_level(df)
    assert int(fv.loc[fv["level"] == "fv_defined", "n_clusters"].iloc[0]) == 6

    excipient = cov.class_b_levels(df, "Excipient_type").set_index("level")
    assert int(excipient.loc["arginine", "n_clusters"]) == 8
    assert int(excipient.loc["lysine", "n_clusters"]) == 4
    assert int(excipient.loc["proline", "n_clusters"]) == 5

    stabilizer = cov.class_b_levels(df, "Stabilizer_type").set_index("level")
    assert int(stabilizer.loc["sucrose", "n_clusters_incl_placebo"]) == 13
    assert int(stabilizer.loc["trehalose", "n_clusters_incl_placebo"]) == 5

    surfactant = cov.class_b_levels(df, "Surfactant_type").set_index("level")
    assert int(surfactant.loc["tween-80", "n_clusters_incl_placebo"]) == 11
    assert int(surfactant.loc["tween-20", "n_clusters_incl_placebo"]) == 5

    buffer = cov.class_b_levels(df, "Buffer_type").set_index("level")
    assert int(buffer.loc["histidine", "n_clusters_incl_placebo"]) == 13
    assert int(buffer.loc["pbs", "n_clusters_incl_placebo"]) == 10
    assert int(buffer.loc["acetate", "n_clusters_incl_placebo"]) == 10


# --- Test 6 -------------------------------------------------------------


def test_bin_series_never_silently_drops_a_row():
    """occupied + explicit-NaN bucket == total, for every binned axis this
    module defines, including on real data with actual NaNs (e.g.
    Whole_Antibody_Charge_at_Buffer_pH is null for every placebo row)."""
    rng = np.random.default_rng(1)
    values = pd.Series(np.concatenate([rng.uniform(-5, 200, size=300), [np.nan] * 10]))
    edges = [0.0, 25.0, 50.0, 100.0, 150.0]  # deliberately narrower than the data range, on both ends
    binned = cov.bin_series(values, edges)
    assert len(binned) == len(values)
    assert binned.isna().sum() == 0  # no bucket is ever a real NaN -- out-of-range/missing become the literal "NaN" label
    assert (binned != "NaN").sum() + (binned == "NaN").sum() == len(values)

    if os.path.exists(REAL_CSV):
        df = cov.load_dataset(REAL_CSV)
        for axis, edges in {**cov.CLASS_A_AXES, **cov.CLASS_C_AXES}.items():
            binned = cov.bin_series(df[axis], edges)
            assert len(binned) == len(df), f"{axis}: row count changed under binning"
            assert binned.isna().sum() == 0, f"{axis}: a real NaN leaked out of bin_series's explicit-NaN bucket"


# =============================================================================
# v2 Task 15 -- tests 7-12
# =============================================================================

# --- Test 7 (Requirement S1-b / C-9) -------------------------------------


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real dataset not present in this checkout")
def test_threshold_scope_mismatch_raises_on_real_csv():
    """C-9's exact error class: REQUIRED_CLUSTERS_GENERALIZATION's native
    unit_of_comparison is feature_presence. Comparing it against a class-A
    partition-cell statistic (class_a_marginal's n_clusters, which sums to
    N_CLUSTERS_TOTAL by construction) must raise, not silently produce the
    10k-12 pattern."""
    df = cov.load_dataset(REAL_CSV)
    marginal = cov.class_a_marginal(df, "kP")
    n_clusters = int(marginal["n_clusters"].iloc[0])
    with pytest.raises(ValueError, match="scope-mismatched"):
        cov.compare_to_threshold(n_clusters, cov.REQUIRED_CLUSTERS_GENERALIZATION, unit_of_comparison=cov.UNIT_PARTITION_CELL)
    # The matching unit (feature_presence, as class-B levels legitimately use) does not raise.
    cov.compare_to_threshold(n_clusters, cov.REQUIRED_CLUSTERS_GENERALIZATION, unit_of_comparison=cov.UNIT_FEATURE_PRESENCE)


# --- Test 8 (C-13) --------------------------------------------------------


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real dataset not present in this checkout")
def test_n_eff_not_monotone_in_row_count():
    """C-13's regression test: dropping poly-hIgG raises n_eff (11.94 ->
    18.68) despite REMOVING rows (434 -> 281), because Kish's design effect
    penalizes imbalance, not row count directly. If a later 'fix' to n_eff
    makes it monotone in row count again, this must fail rather than pass
    silently."""
    df = cov.load_dataset(REAL_CSV)
    non_placebo = df.loc[~cov._is_placebo(df["Protein_type"])]
    bc_all = cov.count_bin(non_placebo, pd.Series(True, index=non_placebo.index), scope=cov.SCOPE_BETWEEN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)
    excl = non_placebo.loc[cov._normalize(non_placebo["Protein_type"]) != "poly-higg"]
    bc_excl = cov.count_bin(excl, pd.Series(True, index=excl.index), scope=cov.SCOPE_BETWEEN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)

    assert bc_excl.n_rows < bc_all.n_rows  # fewer rows...
    assert bc_all.n_eff < bc_excl.n_eff  # ...yet HIGHER n_eff (C-13's non-monotonicity)
    assert bc_all.n_eff == pytest.approx(11.94, abs=0.01)
    assert bc_excl.n_eff == pytest.approx(18.68, abs=0.01)


# --- Test 9 (C-10, rescoped to class-A; Task 10.1 restated) --------------


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real dataset not present in this checkout")
def test_score_guard_against_historical_class_a_range_leverage():
    """Retrospective check of C-10, evaluated against the population it
    actually diagnosed: v1's class-A deficit*range-leverage scalar (retired
    in v2 -- class_a_deficit now raises).

    E11-equivalent, expected-null result, NOT tuned to force a trip:
    check_score_non_degeneracy does NOT raise here (tau_score_deficit=0.468,
    tau_score_leverage=0.262 -- both well under KENDALL_TAU_MAX=0.9). This
    does not mean C-10 was wrong: leverage's IQR on this population is
    genuinely tiny (0.086, see coverage_report.md), but deficit ALSO only
    takes 5 distinct values (2/5/7/8/9) with heavy ties (11 of 26 bins share
    deficit=7) -- Kendall's tau is tie-corrected and gets damped by exactly
    that kind of low-cardinality clustering on EITHER side, so it doesn't
    reach 0.9 even though deficit's IQR (1.0) is >10x leverage's. The
    rank-based reformulation catches a different failure mode (one factor's
    *rank order* fully determines score's rank order) than the original
    IQR check did (one factor has no *spread* to discriminate with) -- they
    are not the same guard wearing a different statistic, and this is the
    honest result of swapping one for the other, not a bug in either.
    Reported per the plan's own instruction: don't lower KENDALL_TAU_MAX to
    force a trip."""
    df = cov.load_dataset(REAL_CSV)
    non_placebo = df.loc[~cov._is_placebo(df["Protein_type"])].copy()
    non_placebo["log_v1000"] = np.log10(non_placebo["Viscosity_1000"].where(non_placebo["Viscosity_1000"] > 0))
    global_min, global_max = non_placebo["log_v1000"].min(), non_placebo["log_v1000"].max()

    deficits, leverages = [], []
    for axis in list(cov.CLASS_A_AXES) + [cov.CLASS_A_CATEGORICAL_AXIS]:
        marginal = cov.class_a_marginal(df, axis)
        bin_col = cov._class_a_bin_column(non_placebo, axis)
        for _, row in marginal.iterrows():
            deficit = max(0.0, cov.REQUIRED_CLUSTERS_GENERALIZATION.value - row["n_clusters"])
            if deficit == 0:
                continue
            lev = cov._leverage_range(non_placebo.loc[bin_col == row["bin"], "log_v1000"], global_min, global_max)
            deficits.append(deficit)
            leverages.append(lev)

    deficits = pd.Series(deficits)
    leverages = pd.Series(leverages)
    score = deficits * leverages
    result = cov.check_score_non_degeneracy(score, deficits, leverages)  # does not raise -- see docstring
    assert result["tau_score_deficit"] < cov.KENDALL_TAU_MAX
    assert result["tau_score_leverage"] < cov.KENDALL_TAU_MAX
    # The signal C-10 actually reported (leverage has ~no spread to work
    # with) still holds and is reproducible independent of the tau guard:
    q1, q3 = np.percentile(leverages, [25, 75])
    assert (q3 - q1) < 0.15


# --- Test 10 (Task 9.2) ---------------------------------------------------


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real dataset not present in this checkout")
def test_collinearity_dedupe_merges_kp_hci_c_class():
    df = cov.load_dataset(REAL_CSV)
    groups = cov.class_a_collinearity_groups(df)
    kp_group = groups.loc[groups["members"].apply(lambda m: "kP" in m)].iloc[0]
    assert set(kp_group["members"]) == {"kP", "HCI", "C_Class"}
    assert kp_group["group_size"] == 3


# --- Test 11 (Task 11) -----------------------------------------------------


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real dataset not present in this checkout")
def test_both_remedy_categories_non_empty_on_real_csv():
    df = cov.load_dataset(REAL_CSV)
    report = cov.build_gap_report(df)
    assert len(report["acquire_cluster"]) > 0
    assert len(report["acquire_rows_within_cluster"]) > 0
    assert set(report["acquire_cluster"]["remedy"].unique()) == {cov.REMEDY_ACQUIRE_CLUSTER}
    assert set(report["acquire_rows_within_cluster"]["remedy"].unique()) == {cov.REMEDY_ACQUIRE_ROWS_WITHIN_CLUSTER}

    # E13 (expected null): the six clusters with no rows below 25 mg/mL are
    # unchanged from v1 -- they appear as acquire_rows_within_cluster entries
    # on the Protein_conc axis, first bin.
    rows = report["acquire_rows_within_cluster"]
    protein_conc_gaps = rows.loc[rows["axis"] == "Protein_conc", "bin"]
    assert len(protein_conc_gaps) > 0


# --- Test 12 (C-9 guard) ---------------------------------------------------


def test_class_a_deficit_raises_on_call():
    with pytest.raises(RuntimeError, match="retired"):
        cov.class_a_deficit()
    with pytest.raises(RuntimeError, match="retired"):
        cov.class_a_deficit(1, 2, foo="bar")
