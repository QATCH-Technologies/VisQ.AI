"""
coverage.py
===========
Dataset coverage & sparsity census for
data/raw/formulation_data_072726_zp_descriptors_charges_added.csv.

Bins the formulation dataset along the axes the model must generalize over
(Class A: protein-constant molecular descriptors: MW/PI_mean/PI_range/kP/
HCI/C_Class/Protein_class_type; Class B: chemical-categorical level support;
Class C: within-cluster condition grid), quantifies how thin each bin is in
the units that govern generalization (cluster count / effective-n, not raw
row count -- Requirement S1/S2 below), and ranks the resulting gaps by
deficit x leverage.

SCOPE DISCIPLINE (why this module exists)
------------------------------------------
"Wrong-scope statistics" -- a valid quantity applied at the wrong scope --
is a recurring error class in this project (aggregate MDE used as a
per-fold threshold; single-measurement SD used as a difference band;
cross-protein noise applied to a within-protein contrast). Every statistic
this module emits carries an explicit scope tag from
{"row", "within_protein", "between_protein"}:
  - in return values      -> BinCount.scope
  - in DataFrame metadata  -> df.attrs["column_scope"]
  - in the CLI report      -> reports/coverage/coverage_report.md

Two hard rules enforce this mechanically rather than by convention:
  S1. BinCount has no __int__/__float__/scalar accessor -- callers must pick
      a field explicitly.
  S2. count_bin() raises if handed an ICC argument under scope=
      "within_protein" -- C-4's ICC (0.49) is a between-protein quantity;
      applying it within a protein is exactly the wrong-scope error this
      module is written against.

TASK 0 GATE -- reproduced against the current CSV, with two corrections
-------------------------------------------------------------------------
13 of 15 Task-0 gate quantities reproduced exactly (see the conversation
that produced this module for the full table). Two did not, and per the
plan's own instruction ("do not reconcile, do not adjust the constant") they
are recorded here as corrections rather than silently substituted:

  C-7. "Rows with all 5 shear channels populated" -- the plan's expected
       value (434) equals the non-placebo row count, but the actual current
       CSV has *every* row (520/520, placebo included) shear-complete. If
       "434" meant "all non-placebo rows are shear-complete," that's
       confirmed (434/434 = 100%); if it meant "434 of 520 total," that's
       contradicted. This module treats shear-channel completeness as a
       live per-row/per-cluster computation (class_c_shear_channel_grid),
       not a hardcoded constant, so the ambiguity has no effect on any
       function's behavior -- only on the historical Task-0 table.
  C-8. "Clusters with any log10 v1000 > 1.0" -- the plan expected 7;
       reproduced value is 9 (etanercept, nivolumab, bsa, poly-higg,
       adalimumab, pembrolizumab, ibalizumab, bgg, bevacizumab all clear
       1.0; only vudalimab/belatacept/trastuzumab do not). The > 1.5 cut
       matches the plan exactly (6 clusters), and LOG_VISC_1000_THRESHOLDS
       below is used for live display/annotation (P4, E7), never hardcoded
       to "7" -- so this correction, like C-7, affects only the historical
       gate table, not this module's behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from visqai.features.categorical import CHEM_CATEGORICALS
from visqai.training.data import NON_PROTEIN_GROUPS, _drop_blank_rows

# =============================================================================
# Task 1 -- Pre-registered constants. Every threshold this module uses is
# declared here: (a) value, (b) scope tag, (c) derivation. No numeric literal
# may be introduced later in a function body (Task 1 acceptance: grep for
# numeric literals in function bodies returns only array indices and 0/1).
# =============================================================================

DEFAULT_CSV_PATH = "data/raw/formulation_data_072726_zp_descriptors_charges_added.csv"

# Requirement S1: the closed set of valid scope tags.
SCOPE_ROW = "row"
SCOPE_WITHIN_PROTEIN = "within_protein"
SCOPE_BETWEEN_PROTEIN = "between_protein"
VALID_SCOPES = {SCOPE_ROW, SCOPE_WITHIN_PROTEIN, SCOPE_BETWEEN_PROTEIN}

# Requirement S1-b (v2 Task 12.2, C-9): a threshold carries a scope AND a
# unit_of_comparison, and may only be compared against a statistic declaring
# the same unit. C-9 was exactly this mismatch: REQUIRED_CLUSTERS_GENERALIZATION
# (native unit: "does this feature/level appear in enough distinct clusters,"
# i.e. feature_presence) applied against class-A partition-cell counts, which
# sum to N_CLUSTERS_TOTAL by construction (class_a_marginal's own assertion)
# -- a unit that threshold was never validated against.
UNIT_FEATURE_PRESENCE = "feature_presence"  # "is this feature/level present in >=k of the 12 clusters"
UNIT_PARTITION_CELL = "partition_cell"  # a partition-of-12 cell count (class-A marginals/pairs) -- never comparable to a feature_presence threshold
UNIT_CONDITION_BIN = "condition_bin"  # "how many of this axis's within-cluster condition bins does one protein occupy"
VALID_COMPARISON_UNITS = {UNIT_FEATURE_PRESENCE, UNIT_PARTITION_CELL, UNIT_CONDITION_BIN}


@dataclass(frozen=True)
class ScopedThreshold:
    """Requirement S1-b: value + scope + unit_of_comparison, bundled so a
    threshold cannot be compared against a statistic of the wrong kind by
    accident (C-9's failure mode)."""

    value: float
    scope: str
    unit_of_comparison: str

    def __post_init__(self):
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"ScopedThreshold: invalid scope {self.scope!r}")
        if self.unit_of_comparison not in VALID_COMPARISON_UNITS:
            raise ValueError(f"ScopedThreshold: invalid unit_of_comparison {self.unit_of_comparison!r}")


def compare_to_threshold(observed: float, threshold: ScopedThreshold, *, unit_of_comparison: str) -> float:
    """Requirement S1-b / Task 12.2: deficit = threshold.value - observed,
    computed ONLY if the caller's declared unit_of_comparison matches the
    threshold's own. A class-A partition-cell statistic compared against
    REQUIRED_CLUSTERS_GENERALIZATION (unit_of_comparison="feature_presence")
    raises here rather than silently producing C-9's `10k - 12` pattern --
    this makes C-9 unrepresentable, not just diagnosed (Test 7)."""
    if unit_of_comparison != threshold.unit_of_comparison:
        raise ValueError(
            f"scope-mismatched threshold comparison: threshold declares "
            f"unit_of_comparison={threshold.unit_of_comparison!r}, comparison requested "
            f"unit_of_comparison={unit_of_comparison!r} -- these are not the same kind of "
            "quantity (Requirement S1-b; this is C-9's exact error class)."
        )
    return threshold.value - observed

# C-3: LOGO grouping unit / placebo stratum. Scope: between_protein (defines
# what a "cluster" is). Reuses visqai.training.data.NON_PROTEIN_GROUPS
# directly (rather than redeclaring the literal "none") so the two constants
# cannot drift apart.
PLACEBO_GROUPS = NON_PROTEIN_GROUPS  # == {"none"}
N_CLUSTERS_TOTAL: int = 12  # Task 0 gate row 2, reproduced from the current CSV.

# C-4: per-protein residual ICC for a between-protein contrast. Scope:
# between_protein. This module CONSUMES the constant (its downstream design
# effect 36.4 / n_eff 11.9 were reproduced exactly against the current CSV
# in the Task 0 gate) -- it does not re-derive ICC itself.
ICC_BETWEEN_PROTEIN: float = 0.49

# Task 6 / Requirement S2: cluster-count adequacy bar for a between-protein
# generalization claim (e.g. "does this ingredient/descriptor bin have
# enough distinct protein clusters behind it"). Derivation: NOT the
# aggregate MDE. Sourced from visqai.features.fv_charge's own documented
# ship criterion -- "a protein-clustered CI needs >=10 clusters to mean
# anything" (fv_charge.py module docstring, C-5) -- the same underlying
# statistical need (a clustered CI over between-protein variance) as every
# class-A/B level censused here. Scope: between_protein.
#
# v2/C-9: native unit_of_comparison is feature_presence -- "does this
# feature/level show up in enough of the 12 clusters." class-B levels and the
# Fv pseudo-level are legitimately feature_presence (a protein can carry
# multiple class-B levels across its rows, so "how many clusters have >=1
# row with level=X" is a real count). class-A partition cells are NOT this
# unit -- they're a partition of the 12 clusters (mutually exclusive,
# summing to N_CLUSTERS_TOTAL by class_a_marginal's own assertion), so
# comparing them against this threshold is comparing a partition_cell count
# against a feature_presence bar (C-9). Wrapped in ScopedThreshold so that
# mismatch raises instead of silently producing a number (Requirement S1-b).
REQUIRED_CLUSTERS_GENERALIZATION = ScopedThreshold(
    value=10.0, scope=SCOPE_BETWEEN_PROTEIN, unit_of_comparison=UNIT_FEATURE_PRESENCE
)

# Task 1 bin edges (Class A -- protein-constant descriptors, C-1). Each edge
# set was chosen so all 12 non-placebo proteins land in exactly one bin --
# enforced by the Task 3 acceptance assertion (sum of 1-D cluster counts ==
# N_CLUSTERS_TOTAL). Scope: between_protein (binning a per-protein-constant
# value is binning protein identity).
MW_BIN_EDGES = [0.0, 100.0, 140.0, 160.0]
PI_MEAN_BIN_EDGES = [0.0, 6.0, 7.0, 8.0, 10.0]
PI_RANGE_BIN_EDGES = [0.0, 0.4, 0.7, 1.1]
KP_BIN_EDGES = [0.0, 2.5, 3.25, 4.0, 5.5]
HCI_BIN_EDGES = [0.0, 0.95, 1.05, 1.2, 1.6]
C_CLASS_BIN_EDGES = [0.0, 0.95, 1.15, 1.45, 1.6]

CLASS_A_AXES: dict[str, list[float]] = {
    "MW": MW_BIN_EDGES,
    "PI_mean": PI_MEAN_BIN_EDGES,
    "PI_range": PI_RANGE_BIN_EDGES,
    "kP": KP_BIN_EDGES,
    "HCI": HCI_BIN_EDGES,
    "C_Class": C_CLASS_BIN_EDGES,
}
CLASS_A_CATEGORICAL_AXIS = "Protein_class_type"  # nominal -- no bin edges, grouped by value directly.
CLASS_A_PAIRS: list[tuple[str, str]] = [
    ("PI_mean", "kP"),
    ("PI_mean", CLASS_A_CATEGORICAL_AXIS),
]

# Class B (chemical-categorical) axes: CHEM_CATEGORICALS minus
# Protein_class_type, which is Class A (C-1's "protein-constant" list).
CLASS_B_AXES: list[str] = [c for c in CHEM_CATEGORICALS if c != CLASS_A_CATEGORICAL_AXIS]

# Category values meaning "this ingredient is absent," normalized the same
# way visqai.features.categorical._normalize_category does, so a class-B
# "none" level here means the same thing it means in the production feature
# pipeline. Scope: row (a per-row string normalization rule, not a
# statistic).
NULL_CATEGORY_VALUES = {"none", "unknown", "na", "n/a", "nan", ""}

# Task 4: Fv-descriptor availability, treated as a pseudo-level of class-B
# support -- must reproduce C-5's 6 Fv-bearing clusters as an ordinary
# output, not a special case.
FV_CHARGE_COL = "Fv_Charge_at_Buffer_pH"
FV_PSEUDO_AXIS = "Fv_availability"

# Task 4: degenerate-continuous flag -- a nominally numeric column where
# almost every row sits on a handful of distinct values is, in practice, a
# categorical wearing a numeric dtype. Scope: row. Pre-registered to trip on
# Salt_conc and Temperature (E5). Purely a detection/display threshold, not
# a modeling claim -- deliberately NOT derived from the aggregate MDE.
DEGENERATE_CANDIDATE_COLS = [
    "Protein_conc",
    "Temperature",
    "Buffer_pH",
    "Buffer_conc",
    "Salt_conc",
    "Stabilizer_conc",
    "Surfactant_conc",
    "Excipient_conc",
    "Whole_Antibody_Charge_at_Buffer_pH",
]
DEGENERATE_MAX_DISTINCT: int = 2
DEGENERATE_MIN_ROW_FRACTION: float = 0.95

# Task 5 (Class C -- within-cluster condition grid, where row collection is
# a valid remedy). Bin edges chosen from the observed column ranges (see the
# conversation that produced this module for the underlying per-column
# min/max query); each set brackets the full observed range with margin so
# no in-range row is ever binned to the explicit-NaN bucket (Task 8 test 6).
# Scope: within_protein (these bin a genuinely row-varying quantity, never a
# cluster count).
PROTEIN_CONC_BIN_EDGES = [0.0, 25.0, 50.0, 100.0, 200.0, 450.0]  # mg/mL-scale; observed range 0-431
BUFFER_PH_BIN_EDGES = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]  # 0.5-pH-unit steps; observed range 5.0-7.4
WHOLE_CHARGE_BIN_EDGES = [-20.0, -5.0, 5.0, 15.0, 25.0, 50.0]  # zero-centered; observed range -16.95..49.15 (non-placebo)
TEMPERATURE_BIN_EDGES = [10.0, 20.0, 24.0, 26.0, 30.0, 40.0]  # brackets the 25/27.5 modes plus stress extremes; observed range 15-36.5

CLASS_C_AXES: dict[str, list[float]] = {
    "Protein_conc": PROTEIN_CONC_BIN_EDGES,
    "Buffer_pH": BUFFER_PH_BIN_EDGES,
    "Whole_Antibody_Charge_at_Buffer_pH": WHOLE_CHARGE_BIN_EDGES,
    "Temperature": TEMPERATURE_BIN_EDGES,
}

# Shear-rate response channels -- a fixed categorical set, not a numeric
# binning. Matches visqai.eval.constants.SHEAR_COLS.
SHEAR_CHANNEL_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]

# Response (log10 Viscosity_1000) decision thresholds, for P4's reference
# lines and the E7 high-viscosity-region check. Scope: row (cuts on a
# per-row response value). Requirement S2: never reused as a cluster-count
# requirement -- purely a display/annotation cut, independent of
# REQUIRED_CLUSTERS_GENERALIZATION.
LOG_VISC_1000_THRESHOLDS = [1.0, 1.5]

# Task 7: heatmap cells below this raw row count are hatched instead of
# colored, so "thin" and "zero" read as visually distinct. Scope: row. Pure
# legibility threshold, not a statistical claim.
MIN_CELL_FOR_DISPLAY: int = 3

# Task 6: LOGO residual artifact schema, for leverage="logo_residual".
LOGO_RESIDUAL_REQUIRED_COLS = {"axis", "bin", "log_mae"}

# Task 11 (C-11): remedy taxonomy collapsed from three values to two.
# v1 had a third, acquire_condition, that never actually differed from
# acquire_rows_within_cluster in practice -- both were assigned purely from
# axis class (A/B -> acquire_cluster, C -> acquire_condition), and "a new
# formulation of an existing protein" is the same remedy whether the empty
# cell is a condition bin or a shear channel. Assignment is now by what
# would close the gap, not by axis class: acquire_cluster needs a protein
# not currently in the dataset (class B/Fv); acquire_rows_within_cluster is
# satisfiable with new formulations of an existing protein (class C,
# including shear-channel gaps).
REMEDY_ACQUIRE_CLUSTER = "acquire_cluster"
REMEDY_ACQUIRE_ROWS_WITHIN_CLUSTER = "acquire_rows_within_cluster"

# =============================================================================
# v2 Task 9 -- class-A descriptor-space void metric (replaces class-A deficit
# scoring entirely; C-9). The six class-A axes are a re-partition of the same
# 12 points, so "which bin is short of REQUIRED_CLUSTERS_GENERALIZATION" is
# unanswerable at that scope (deficit is fixed at 10k-12 for any k-bin axis
# by construction) -- the answerable question is a space-filling one: where
# in descriptor space would a new protein be most informative.
# =============================================================================

# 9.1: PCA retained-variance threshold for the whitened void-map subspace.
CLASS_A_VOID_VARIANCE_THRESHOLD: float = 0.90

# 9.2: single-linkage collinearity dedupe threshold on the 12-protein
# descriptor correlation matrix. A class-A output reports only the group
# representative, never every collinear member.
CLASS_A_COLLINEARITY_THRESHOLD: float = 0.85

# 9.3: void-map search parameters, in the whitened (unit-variance) retained-
# PCA subspace, where Euclidean distance == Mahalanobis distance in the
# original standardized descriptor space restricted to that subspace.
CLASS_A_VOID_GRID_POINTS_PER_AXIS: int = 25
CLASS_A_VOID_GRID_MARGIN: float = 0.15  # fractional padding beyond the observed min/max per retained PC
CLASS_A_VOID_MAX_REGIONS: int = 5
CLASS_A_VOID_MIN_SEPARATION: float = 1.0  # whitened-space NMS radius (~1 Mahalanobis unit)

# =============================================================================
# v2 Task 10/16 -- leverage & score non-degeneracy (C-10, C-15, C-16)
# =============================================================================

# C-15 (validity, not tuning): a "high_visc_share" leverage -- the fraction
# of a bin's OWN rows above a decision threshold -- was evaluated and
# rejected. It is not implemented as a selectable leverage mode. Two
# independent reasons, either sufficient on its own:
#   1. Circularity: leverage is meant to measure how much a gap matters: a
#      bin that has never been pushed into the high-viscosity regime scores
#      exactly 0 -- not "low value," but "no observation." A coverage census
#      that weights gaps by the response observed *within the bin being
#      scored* penalizes underobserved bins twice, once as the deficit and
#      again as near-zero leverage, for the same underlying reason (sparsity).
#   2. C-14 confound: the clusters that ever cross the high-viscosity cut are
#      disproportionately the Fv-undefined formats. Leverage built from that
#      cut would systematically deprioritize acquisitions in exactly the
#      region the census exists to surface -- worse than uninformative.
# leverage may not be a function of the response observed within the scored
# bin; only "range" (an occupancy-shape statistic, not a threshold share) and
# "logo_residual" (exogenous -- model error at the bin, not response value)
# are implemented.

# C-16: score = deficit * leverage, as a single scalar, is retired for
# "range" (endogenous, non-exogenous leverage). A two-factor product is
# governed by whichever factor has the larger relative rank spread, and
# there is no leverage definition that escapes this except by coincidence
# (C-10 for "range" alone; the deleted high_visc_share would have mirrored
# it in the opposite direction, corr(score, leverage) = 0.92 on the real
# post-Task-9 data -- see coverage_report.md). Default presentation is a
# Pareto frontier over (deficit, leverage) coordinates per remedy class
# (never across remedy classes -- Task 11: they draw on different budgets).
# leverage="logo_residual" is exogenous (per-bin model error, not a function
# of the response observed in the bin) and remains the one mode that still
# produces a defensible scalar score -- when produced, it is rank-checked
# rather than trusted blindly (KENDALL_TAU_MAX below).
DEFAULT_LEVERAGE_MODE: str = "range"

# Task 10.1, restated (rank-based, two-sided): for a scalar score (logo_residual
# mode only -- Pareto-frontier mode has no scalar to check), Kendall's tau
# between score and each of its two factors must not exceed this in absolute
# value. IQR alone doesn't catch rank-domination (a factor can have plenty of
# spread and still govern the *order*); rank correlation does. Chosen as a
# round, non-data-tuned bar before checking what either factor's tau actually
# is (E11-equivalent: report a non-trip plainly rather than lowering this).
KENDALL_TAU_MAX: float = 0.9

# =============================================================================
# v2 Task 12.1 -- class-C design-variation gate (C-12)
# =============================================================================

# An axis is admitted to the acquire_rows_within_cluster ranking only if it
# was deliberately varied: not degenerate-continuous at row scope AND
# showing genuine within-cluster variation, not just float-logging noise.
# Pre-registered: Temperature fails via the second clause (raw distinct
# values per cluster), not the first -- its row-scope top-2-value fraction
# (85.4%) sits under DEGENERATE_MIN_ROW_FRACTION (95%), but only 3 of 12
# clusters ever record >=3 distinct raw Temperature values; the rest sit
# almost entirely on one of the two standard setpoints (E12).
DESIGN_VARIATION_MIN_DISTINCT_VALUES: int = 3
DESIGN_VARIATION_MIN_CLUSTERS: int = 6

# Task 12.3 (C-13): n_eff is not monotone in row count (Kish's design effect
# penalizes imbalance) and is not cross-bin comparable without a balance
# statistic alongside it. BALANCE_TOLERANCE flags when two n_eff values being
# compared come from meaningfully different balance regimes.
BALANCE_TOLERANCE: float = 0.15

# =============================================================================
# v3 Task 17 -- class-C collinearity dedupe (C-18)
# =============================================================================

# Same numeric bar as the class-A dedupe (Task 9.2), reused rather than
# re-derived -- there is no principled reason for class C to need a
# different threshold, and the plan's own S1-discipline prefers reuse over a
# second constant that could drift from the first.
CLASS_C_COLLINEARITY_THRESHOLD: float = CLASS_A_COLLINEARITY_THRESHOLD

# 17.2: representative selection prefers the axis an experimenter sets
# directly. Buffer_pH is set; Whole_Antibody_Charge_at_Buffer_pH is a
# deterministic function of protein + pH (C-2) -- derived, not manipulable.
# Protein_conc is also directly set. Any class-C axis not listed here is
# treated as derived (conservative -- an unlisted axis never wins a
# representative tie against a listed one).
CLASS_C_MANIPULABLE_AXES: set[str] = {"Protein_conc", "Buffer_pH"}

# A 2-variable Pearson correlation on n points has n-2 residual degrees of
# freedom; below this many points a correlation estimate is not stable
# enough to decide a merge on -- e.g. mAb_IgG4's 3 members show
# corr(MW, PI_mean) = -0.93 purely because 3 points in 2D are nearly always
# close to a line, not because MW and PI_mean are related within that
# format class. Below this count, class_a_collinearity_groups reports every
# axis as its own singleton rather than merging on an unstable estimate.
MIN_N_FOR_COLLINEARITY_DEDUPE: int = 5

# =============================================================================
# v3 Task 16 (restructured) -- MW/PI_mean/PI_range are the only genuinely
# continuous class-A descriptors (C-25). kP/HCI/C_Class are a deterministic
# per-format lookup, not measured per molecule -- see class_a_descriptor_space
# for the verified bijection. They stay in CLASS_A_AXES (Task 3's 1-D/2-D
# marginal census is still a valid count of "how many clusters have kP in
# bin X," independent of what kP represents) but are EXCLUDED from the
# continuous PCA/void-map machinery, which now runs on these three only, and
# is stratified by Protein_class_type rather than pooled across all 12
# proteins (a point interpolated between two format classes' kP/HCI/C_Class
# values is not a request for a real molecule -- C-17).
# =============================================================================

CLASS_A_CONTINUOUS_AXES: list[str] = ["MW", "PI_mean", "PI_range"]

# kP/HCI/C_Class: kept as a named constant so any future code reaching for
# "the class-A descriptors" for a continuous-space purpose is pointed at
# CLASS_A_CONTINUOUS_AXES instead of CLASS_A_AXES's full six.
PROTEIN_CLASS_ENCODING_AXES: list[str] = ["kP", "HCI", "C_Class"]

# 16.1: minimum members a format class needs before a continuous void map
# within it means anything (a nearest-neighbour/PCA search needs an interior
# to search). Below this, the class itself -- not a sub-region of it -- is
# the reportable gap (class_a_categorical_coverage).
MIN_CLASS_SIZE_FOR_VOID_MAP: int = 3

# 16.1: required, no default, deliberately absent from this module -- see
# class_a_void_regions's docstring. Values are supplied by the caller
# (coverage_plots.py / the CLI / tests), never inferred from the observed
# data range.
FEASIBILITY_ENVELOPE_REQUIRED_AXES: list[str] = list(CLASS_A_CONTINUOUS_AXES)


# =============================================================================
# Shared helpers
# =============================================================================


def _is_placebo(protein_type: pd.Series) -> pd.Series:
    return protein_type.astype(str).str.strip().str.lower().isin(PLACEBO_GROUPS)


def _normalize(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _normalize_level(raw: pd.Series) -> pd.Series:
    """Class-B category normalization, matching
    visqai.features.categorical._normalize_category's null-handling so a
    'none' level here means what it means in the production feature
    pipeline."""
    s = raw.astype(str).str.strip().str.lower()
    return s.where(~s.isin(NULL_CATEGORY_VALUES), "none")


def _fmt_edge(x: float) -> str:
    return f"{x:g}"


def bin_series(values: pd.Series, edges: list[float]) -> pd.Series:
    """pd.cut wrapper with an explicit 'NaN' bucket for missing/out-of-range
    values, so no row is ever silently dropped by a binning (Task 8 test 6:
    occupied + explicit-NaN bucket == total, always)."""
    edges = list(edges)
    labels = [f"[{_fmt_edge(edges[i])}, {_fmt_edge(edges[i + 1])})" for i in range(len(edges) - 1)]
    cut = pd.cut(values, bins=edges, labels=labels, right=False, include_lowest=True)
    out = cut.astype(object)
    return out.where(cut.notna(), "NaN")


def _attach_scope(df: pd.DataFrame, column_scope: dict[str, str]) -> pd.DataFrame:
    """Requirement S1: stamp every emitted census table with an explicit,
    per-column scope tag in DataFrame.attrs -- never inferred from the
    column name."""
    bad = set(column_scope.values()) - VALID_SCOPES
    if bad:
        raise ValueError(f"_attach_scope: invalid scope tag(s) {bad}")
    df = df.copy()
    df.attrs["column_scope"] = dict(column_scope)
    return df


def load_dataset(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return _drop_blank_rows(df)


# =============================================================================
# Task 2 -- The three counting statistics
# =============================================================================


@dataclass(frozen=True)
class BinCount:
    """The three (now four) counting statistics for one bin. Deliberately no
    __int__/__float__/scalar accessor -- every consumer must select a field
    explicitly (mechanical enforcement of Requirement S1)."""

    scope: str
    n_rows: int  # scope: row
    n_clusters: int  # scope: between_protein (always excludes the placebo stratum, C-3)
    n_eff: float  # scope: between_protein
    balance: float  # scope: between_protein (Task 12.3/C-13). Kish balance =
    # (Sum n_i)^2 / (k * Sum n_i^2) in [0, 1]: 1.0 for perfectly equal
    # cluster sizes, -> 0 as one cluster dominates. n_eff is NOT monotone in
    # row count or cross-bin comparable without this alongside it (see
    # compare_n_eff). NaN when n_clusters == 0 (undefined).


def kish_design_effect(cluster_sizes, icc: float) -> float:
    """deff = 1 + (Sum(n_i^2)/Sum(n_i) - 1) * icc (Task 2). Reduces to 1.0 at
    icc=0 for any cluster-size distribution; at icc=1, for a synthetic frame
    of k equal-size-k clusters, reduces to k (Task 8 test 2)."""
    sizes = np.asarray(list(cluster_sizes), dtype=float)
    sizes = sizes[sizes > 0]
    n = sizes.sum()
    if n == 0:
        return 1.0
    return 1.0 + (float(np.sum(sizes**2)) / n - 1.0) * icc


def count_bin(df: pd.DataFrame, mask, *, scope: str, icc: float | None = None) -> BinCount:
    """The three counting statistics for the rows `mask` selects out of
    `df` (Task 2).

    n_rows counts every selected row (placebo included -- it's a row-scope
    count, not a cluster count). n_clusters/n_eff always exclude the
    placebo stratum (C-3): a bin containing only poly-hIgG rows returns
    n_clusters == 1 regardless of n_rows, and n_eff for a single large
    cluster is bounded by 1/icc as that cluster's row count grows (Task 8
    test 1) -- not by n_rows itself.

    icc: pass ICC_BETWEEN_PROTEIN explicitly to get a clustering-corrected
    n_eff for scope="between_protein"; omit for scope="row". Passing icc
    under scope="within_protein" raises: C-4's ICC is a between-protein
    quantity, and applying it to a within-protein cell is exactly the
    wrong-scope error this module exists to prevent (Task 5 guard, Task 8
    test 3).
    """
    if scope not in VALID_SCOPES:
        raise ValueError(f"count_bin: invalid scope {scope!r}, must be one of {sorted(VALID_SCOPES)}")
    if scope == SCOPE_WITHIN_PROTEIN and icc is not None:
        raise ValueError(
            "count_bin: icc must not be supplied for scope='within_protein' -- "
            "ICC_BETWEEN_PROTEIN (C-4) is a between-protein quantity; applying it "
            "to a within-protein cell is the wrong-scope error class this module "
            "is written against (Task 5 guard)."
        )

    sub = df.loc[mask]
    n_rows = int(len(sub))
    non_placebo = sub.loc[~_is_placebo(sub["Protein_type"])]
    sizes = _normalize(non_placebo["Protein_type"]).value_counts()
    n_clusters = int(len(sizes))

    if icc is None or n_clusters == 0:
        n_eff = float(n_rows)
    else:
        deff = kish_design_effect(sizes.to_numpy(), icc)
        n_eff = float(n_rows) / deff if deff > 0 else float(n_rows)

    if n_clusters == 0:
        balance = float("nan")
    else:
        sizes_arr = sizes.to_numpy(dtype=float)
        balance = float((sizes_arr.sum() ** 2) / (n_clusters * float(np.sum(sizes_arr**2))))

    return BinCount(scope=scope, n_rows=n_rows, n_clusters=n_clusters, n_eff=n_eff, balance=balance)


def compare_n_eff(a: BinCount, b: BinCount) -> dict:
    """Task 12.3 (C-13): n_eff is not monotone in row count -- dropping
    poly-hIgG raises n_eff from 11.94 to 18.68 on FEWER rows, because Kish's
    design effect penalizes imbalance through Sum(n_i^2)/Sum(n_i), not row
    count directly. A cross-bin n_eff comparison is only meaningful alongside
    a balance statistic; this helper requires both BinCounts' balance fields
    and flags when they differ by more than BALANCE_TOLERANCE, rather than
    letting 'n_eff went up' be read as 'better powered' at face value."""
    if a.scope != SCOPE_BETWEEN_PROTEIN or b.scope != SCOPE_BETWEEN_PROTEIN:
        raise ValueError("compare_n_eff: both BinCounts must be scope='between_protein' -- n_eff is a between_protein quantity.")
    balance_diff = abs(a.balance - b.balance)
    return {
        "n_eff_a": a.n_eff,
        "n_eff_b": b.n_eff,
        "balance_a": a.balance,
        "balance_b": b.balance,
        "balance_diff": balance_diff,
        "balance_comparable": bool(balance_diff <= BALANCE_TOLERANCE),
        "n_eff_higher": "a" if a.n_eff > b.n_eff else ("b" if b.n_eff > a.n_eff else "tie"),
    }


def log_viscosity_threshold_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Non-placebo cluster count with any row above each
    LOG_VISC_1000_THRESHOLDS cut (Task 0 gate row 14 / E7's high-viscosity
    region check). Scope: between_protein. Not used as a cluster-count
    requirement anywhere (Requirement S2) -- a display/reporting query
    only."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])
    non_placebo["log_v1000"] = np.log10(non_placebo["Viscosity_1000"].where(non_placebo["Viscosity_1000"] > 0))

    rows = []
    for thr in LOG_VISC_1000_THRESHOLDS:
        clusters = sorted(non_placebo.loc[non_placebo["log_v1000"] > thr, "_pt"].unique())
        rows.append({"threshold": thr, "n_clusters": len(clusters), "clusters": ", ".join(clusters)})
    out = pd.DataFrame(rows)
    return _attach_scope(out, {"n_clusters": SCOPE_BETWEEN_PROTEIN})


# =============================================================================
# Task 3 -- Class A census: group-level descriptor axes
# =============================================================================


def _class_a_bin_column(non_placebo: pd.DataFrame, axis: str) -> pd.Series:
    if axis == CLASS_A_CATEGORICAL_AXIS:
        return non_placebo[axis].astype(str).str.strip()
    return bin_series(non_placebo[axis], CLASS_A_AXES[axis])


def class_a_marginal(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """1-D class-A census for one protein-constant descriptor axis (C-1).
    Cluster counts only drive ranking; n_rows is carried as a secondary
    column. Asserts every cell has n_clusters <= N_CLUSTERS_TOTAL and that
    the 1-D marginal's cluster counts sum to exactly N_CLUSTERS_TOTAL (each
    protein lands in exactly one bin) -- a violation means either row
    leakage into a cluster count or a bin-edge gap dropping a protein to
    the NaN bucket."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_bin"] = _class_a_bin_column(non_placebo, axis)

    rows = []
    for bin_label in sorted(non_placebo["_bin"].unique()):
        bc = count_bin(non_placebo, non_placebo["_bin"] == bin_label, scope=SCOPE_BETWEEN_PROTEIN, icc=ICC_BETWEEN_PROTEIN)
        rows.append(
            {"axis": axis, "bin": bin_label, "n_clusters": bc.n_clusters, "n_rows": bc.n_rows, "n_eff": bc.n_eff, "balance": bc.balance}
        )
    out = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)

    assert out["n_clusters"].max() <= N_CLUSTERS_TOTAL, (
        f"class-A axis {axis!r}: a bin reports n_clusters > {N_CLUSTERS_TOTAL} -- row leakage into a cluster count."
    )
    total = int(out["n_clusters"].sum())
    assert total == N_CLUSTERS_TOTAL, (
        f"class-A axis {axis!r}: cluster counts sum to {total}, not {N_CLUSTERS_TOTAL} -- "
        "bin edges are dropping a protein to the NaN bucket."
    )
    return _attach_scope(
        out, {"n_rows": SCOPE_ROW, "n_clusters": SCOPE_BETWEEN_PROTEIN, "n_eff": SCOPE_BETWEEN_PROTEIN, "balance": SCOPE_BETWEEN_PROTEIN}
    )


def class_a_pair(df: pd.DataFrame, axis_x: str, axis_y: str) -> pd.DataFrame:
    """2-D class-A census over a pair of descriptor axes (Task 3)."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_bx"] = _class_a_bin_column(non_placebo, axis_x)
    non_placebo["_by"] = _class_a_bin_column(non_placebo, axis_y)

    rows = []
    for bx in sorted(non_placebo["_bx"].unique()):
        for by in sorted(non_placebo["_by"].unique()):
            mask = (non_placebo["_bx"] == bx) & (non_placebo["_by"] == by)
            bc = count_bin(non_placebo, mask, scope=SCOPE_BETWEEN_PROTEIN, icc=ICC_BETWEEN_PROTEIN)
            rows.append(
                {
                    "axis_x": axis_x,
                    "axis_y": axis_y,
                    "bin_x": bx,
                    "bin_y": by,
                    "n_clusters": bc.n_clusters,
                    "n_rows": bc.n_rows,
                    "n_eff": bc.n_eff,
                    "balance": bc.balance,
                }
            )
    out = pd.DataFrame(rows)
    assert out["n_clusters"].max() <= N_CLUSTERS_TOTAL, (
        f"class-A pair ({axis_x}, {axis_y}): a cell reports n_clusters > {N_CLUSTERS_TOTAL}."
    )
    total = int(out["n_clusters"].sum())
    assert total == N_CLUSTERS_TOTAL, (
        f"class-A pair ({axis_x}, {axis_y}): cluster counts sum to {total}, not {N_CLUSTERS_TOTAL}."
    )
    return _attach_scope(
        out, {"n_rows": SCOPE_ROW, "n_clusters": SCOPE_BETWEEN_PROTEIN, "n_eff": SCOPE_BETWEEN_PROTEIN, "balance": SCOPE_BETWEEN_PROTEIN}
    )


# =============================================================================
# Task 4 -- Class B census: categorical level support
# =============================================================================


def class_b_levels(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Per-level support census for one class-B categorical axis. Includes
    the 'none' (ingredient-absent) level -- a real, meaningful stratum, not
    something to drop."""
    levels = _normalize_level(df[axis])
    pt_norm = _normalize(df["Protein_type"])

    rows = []
    for level in sorted(levels.unique()):
        mask = levels == level
        bc = count_bin(df, mask, scope=SCOPE_BETWEEN_PROTEIN, icc=ICC_BETWEEN_PROTEIN)
        n_clusters_incl_placebo = int(pt_norm[mask].nunique())  # here "none" (placebo) counts as its own group
        rows.append(
            {
                "axis": axis,
                "level": level,
                "n_rows": bc.n_rows,
                "n_clusters": bc.n_clusters,
                "n_clusters_incl_placebo": n_clusters_incl_placebo,
                "n_eff": bc.n_eff,
                "balance": bc.balance,
            }
        )
    out = pd.DataFrame(rows)
    return _attach_scope(
        out,
        {
            "n_rows": SCOPE_ROW,
            "n_clusters": SCOPE_BETWEEN_PROTEIN,
            "n_clusters_incl_placebo": SCOPE_BETWEEN_PROTEIN,
            "n_eff": SCOPE_BETWEEN_PROTEIN,
            "balance": SCOPE_BETWEEN_PROTEIN,
        },
    )


def class_b_fv_pseudo_level(df: pd.DataFrame) -> pd.DataFrame:
    """Fv-descriptor availability as a pseudo-level of class-B support --
    must reproduce C-5's 6 Fv-bearing clusters as an ordinary output of this
    table, not a special case."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])]
    fv_defined = non_placebo[FV_CHARGE_COL].notna()

    rows = []
    for level, mask in [("fv_defined", fv_defined), ("fv_undefined", ~fv_defined)]:
        bc = count_bin(non_placebo, mask, scope=SCOPE_BETWEEN_PROTEIN, icc=ICC_BETWEEN_PROTEIN)
        rows.append(
            {
                "axis": FV_PSEUDO_AXIS,
                "level": level,
                "n_rows": bc.n_rows,
                "n_clusters": bc.n_clusters,
                "n_eff": bc.n_eff,
                "balance": bc.balance,
            }
        )
    out = pd.DataFrame(rows)
    return _attach_scope(
        out, {"n_rows": SCOPE_ROW, "n_clusters": SCOPE_BETWEEN_PROTEIN, "n_eff": SCOPE_BETWEEN_PROTEIN, "balance": SCOPE_BETWEEN_PROTEIN}
    )


def class_b_crossings(df: pd.DataFrame) -> pd.DataFrame:
    """For every pair of distinct class-B axes, the cluster count behind
    each (level_a, level_b) co-occurrence, non-placebo. A row with
    n_clusters == 0 is a combination that never occurs in the data -- kept
    visible rather than silently omitted, since a sparse/absent crossing is
    exactly what makes an interaction term unidentifiable."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    norm_cols = {axis: _normalize_level(non_placebo[axis]) for axis in CLASS_B_AXES}

    rows = []
    for i, axis_a in enumerate(CLASS_B_AXES):
        for axis_b in CLASS_B_AXES[i + 1 :]:
            for level_a in sorted(norm_cols[axis_a].unique()):
                mask_a = norm_cols[axis_a] == level_a
                for level_b in sorted(norm_cols[axis_b].unique()):
                    mask = mask_a & (norm_cols[axis_b] == level_b)
                    bc = count_bin(non_placebo, mask, scope=SCOPE_BETWEEN_PROTEIN, icc=ICC_BETWEEN_PROTEIN)
                    rows.append(
                        {
                            "axis_a": axis_a,
                            "level_a": level_a,
                            "axis_b": axis_b,
                            "level_b": level_b,
                            "n_clusters": bc.n_clusters,
                            "n_rows": bc.n_rows,
                        }
                    )
    out = pd.DataFrame(rows)
    return _attach_scope(out, {"n_rows": SCOPE_ROW, "n_clusters": SCOPE_BETWEEN_PROTEIN})


def degenerate_continuous_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flags any nominally-numeric column where >= DEGENERATE_MIN_ROW_FRACTION
    of non-null rows sit on <= DEGENERATE_MAX_DISTINCT distinct values (Task
    4). E5 pre-registers Salt_conc and Temperature to trip this."""
    rows = []
    for col in DEGENERATE_CANDIDATE_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        top = vals.value_counts().head(DEGENERATE_MAX_DISTINCT)
        frac = float(top.sum()) / len(vals)
        rows.append(
            {
                "column": col,
                "n_distinct_top": int(len(top)),
                "top_value_row_fraction": frac,
                "degenerate": bool(frac >= DEGENERATE_MIN_ROW_FRACTION),
            }
        )
    out = pd.DataFrame(rows)
    return _attach_scope(out, {"top_value_row_fraction": SCOPE_ROW})


# =============================================================================
# Task 5 -- Class C census: within-cluster condition grid
# =============================================================================


def class_c_grid(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Protein (cluster) x condition-bin row-count grid for one class-C axis
    -- the axes where row collection is a valid remedy. n_rows only (scope:
    within_protein); n_eff is never computed here (count_bin is called with
    icc omitted, scope='within_protein' -- see the Task 5 guard on
    count_bin itself)."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])
    non_placebo["_bin"] = bin_series(non_placebo[axis], CLASS_C_AXES[axis])

    rows = []
    for pt in sorted(non_placebo["_pt"].unique()):
        mask_pt = non_placebo["_pt"] == pt
        for b in sorted(non_placebo["_bin"].unique()):
            bc = count_bin(non_placebo, mask_pt & (non_placebo["_bin"] == b), scope=SCOPE_WITHIN_PROTEIN)
            rows.append({"axis": axis, "protein": pt, "bin": b, "n_rows": bc.n_rows})
    out = pd.DataFrame(rows)
    return _attach_scope(out, {"n_rows": SCOPE_WITHIN_PROTEIN})


def class_c_grid_completeness(grid: pd.DataFrame, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-protein grid completeness = fraction of the axis's condition bins
    occupied by >=1 row (the explicit-NaN bucket never counts as
    'occupied'), plus the cluster-level marginal: how many clusters occupy
    each bin."""
    n_bins = len(CLASS_C_AXES[axis]) - 1
    occ = grid.loc[(grid["bin"] != "NaN") & (grid["n_rows"] > 0)]

    per_protein = (
        occ.groupby("protein")["bin"]
        .nunique()
        .reindex(sorted(grid["protein"].unique()), fill_value=0)
        .div(n_bins)
        .rename("completeness")
        .reset_index()
    )
    per_bin = occ.groupby("bin")["protein"].nunique().rename("n_clusters_occupying").reset_index()
    per_protein = _attach_scope(per_protein, {"completeness": SCOPE_WITHIN_PROTEIN})
    per_bin = _attach_scope(per_bin, {"n_clusters_occupying": SCOPE_BETWEEN_PROTEIN})
    return per_protein, per_bin


def class_c_shear_channel_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Protein x shear-channel row-count grid (Task 5's fifth class-C axis;
    the channel set is fixed/categorical, not numerically binned)."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])

    rows = []
    for pt in sorted(non_placebo["_pt"].unique()):
        g = non_placebo.loc[non_placebo["_pt"] == pt]
        for col in SHEAR_CHANNEL_COLS:
            rows.append({"axis": "shear_channel", "protein": pt, "bin": col, "n_rows": int(g[col].notna().sum())})
    out = pd.DataFrame(rows)
    return _attach_scope(out, {"n_rows": SCOPE_WITHIN_PROTEIN})


# =============================================================================
# Task 6 -- Deficit x leverage ranking
# =============================================================================


def _leverage_range(values: pd.Series, global_min: float, global_max: float) -> float:
    """leverage='range': fraction of the observed log10(Viscosity_1000)
    range this bin's rows span (Task 6 default -- needs no model
    artifacts)."""
    values = values.dropna()
    span = global_max - global_min
    if len(values) == 0 or span <= 0:
        return 0.0
    return float((values.max() - values.min()) / span)


def _content_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_logo_residual_artifact(path: str | None) -> tuple[pd.DataFrame, str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"leverage='logo_residual' requires an existing LOGO run artifact at {path!r}. "
            "Refusing to silently fall back to leverage='range' (Task 6)."
        )
    digest = _content_hash(path)
    residual_df = pd.read_csv(path)
    missing = LOGO_RESIDUAL_REQUIRED_COLS - set(residual_df.columns)
    if missing:
        raise ValueError(f"LOGO residual artifact at {path!r} is missing required column(s) {missing}.")
    return residual_df, digest


def _logo_residual_leverage(residual_df: pd.DataFrame, axis: str, bin_label) -> float:
    match = residual_df.loc[(residual_df["axis"] == axis) & (residual_df["bin"].astype(str) == str(bin_label))]
    return float(match["log_mae"].mean()) if len(match) else 0.0


def _leverage_fn(leverage: str, residual_df: pd.DataFrame | None, global_min: float, global_max: float):
    if leverage == "range":
        return lambda values, axis, bin_label: _leverage_range(values, global_min, global_max)
    if leverage == "logo_residual":
        return lambda values, axis, bin_label: _logo_residual_leverage(residual_df, axis, bin_label)
    raise ValueError(f"unknown leverage {leverage!r}, expected 'range' or 'logo_residual'")


# =============================================================================
# v2 Task 12.1 -- class-C design-variation gate (C-12)
# =============================================================================


def class_c_design_variation_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Task 12.1 (C-12): an axis is admitted to the acquire_rows_within_cluster
    ranking only if it was deliberately varied -- it must not trip the
    row-scope degenerate-continuous flag AND must show
    >=DESIGN_VARIATION_MIN_DISTINCT_VALUES raw distinct values in
    >=DESIGN_VARIATION_MIN_CLUSTERS clusters. Pre-registered: Temperature
    fails via the second clause alone (E12) -- its row-scope top-2-value
    fraction (85.4%) sits under DEGENERATE_MIN_ROW_FRACTION so it does not
    trip the first clause, but only 3 of 12 clusters ever record >=3 distinct
    raw Temperature values; the rest sit almost entirely on one of the two
    standard setpoints."""
    degen = degenerate_continuous_flags(df).set_index("column")["degenerate"]
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])

    rows = []
    for axis in CLASS_C_AXES:
        raw_nunique = non_placebo.groupby("_pt")[axis].nunique()
        n_clusters_varied = int((raw_nunique >= DESIGN_VARIATION_MIN_DISTINCT_VALUES).sum())
        is_degenerate = bool(degen.get(axis, False))
        varied_enough = n_clusters_varied >= DESIGN_VARIATION_MIN_CLUSTERS
        rows.append(
            {
                "axis": axis,
                "degenerate_continuous": is_degenerate,
                "n_clusters_with_ge3_distinct_values": n_clusters_varied,
                "admitted": bool((not is_degenerate) and varied_enough),
            }
        )
    out = pd.DataFrame(rows)
    return _attach_scope(out, {"n_clusters_with_ge3_distinct_values": SCOPE_BETWEEN_PROTEIN})


# =============================================================================
# v3 Task 17 -- class-C collinearity dedupe (C-18)
# =============================================================================


def _pooled_within_protein_correlation(df: pd.DataFrame, axis_a: str, axis_b: str) -> float:
    """Pearson correlation of the WITHIN-protein deviations (each axis
    demeaned by its own protein's mean before pooling), not the raw
    pooled-across-proteins correlation. The latter is contaminated by
    between-protein offsets (C-4): two axes can look correlated purely
    because different proteins sit at different baseline levels on both,
    with no within-protein relationship at all. This is exactly the
    distinction Buffer_pH vs. Whole_Antibody_Charge_at_Buffer_pH needs --
    charge-at-pH is a deterministic function of (protein, pH) per C-2, so
    the within-protein relationship is where that determinism actually
    shows up."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])
    sub = non_placebo[["_pt", axis_a, axis_b]].dropna()
    a_c = sub[axis_a] - sub.groupby("_pt")[axis_a].transform("mean")
    b_c = sub[axis_b] - sub.groupby("_pt")[axis_b].transform("mean")
    return float(a_c.corr(b_c))


def class_c_collinearity_groups(df: pd.DataFrame, axes: list[str], threshold: float = CLASS_C_COLLINEARITY_THRESHOLD) -> pd.DataFrame:
    """Task 17.1/17.2: single-linkage dedupe of the class-C axes ADMITTED by
    the Task 12.1 design-variation gate, on pooled within-protein |r| (not
    Pearson on the raw pool -- see _pooled_within_protein_correlation).
    Representative selection prefers a CLASS_C_MANIPULABLE_AXES member (the
    variable an experimenter actually sets) over a derived one -- an
    acquisition instruction has to name a settable variable, not a
    downstream consequence of it. `collinear_with` lists the absorbed axes.

    On the real CSV: Buffer_pH and Whole_Antibody_Charge_at_Buffer_pH clear
    0.85 (pooled within-protein |r|=0.945) and collapse; Buffer_pH wins the
    representative slot (manipulable) over the charge axis (derived).
    Protein_conc is uncorrelated with either (|r|<0.1) and stays a
    singleton."""
    corr = pd.DataFrame(1.0, index=axes, columns=axes)
    for i, a in enumerate(axes):
        for b in axes[i + 1 :]:
            r = abs(_pooled_within_protein_correlation(df, a, b))
            corr.loc[a, b] = corr.loc[b, a] = r

    parent = {a: a for a in axes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(axes):
        for b in axes[i + 1 :]:
            if corr.loc[a, b] >= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for a in axes:
        groups.setdefault(find(a), []).append(a)

    rows = []
    for members in groups.values():
        manipulable = [m for m in members if m in CLASS_C_MANIPULABLE_AXES]
        ranked = sorted(manipulable, key=axes.index) + sorted([m for m in members if m not in manipulable], key=axes.index)
        rep = ranked[0]
        rows.append(
            {
                "representative": rep,
                "members": sorted(members, key=axes.index),
                "collinear_with": sorted(m for m in members if m != rep),
                "group_size": len(members),
            }
        )
    return pd.DataFrame(rows).sort_values("representative").reset_index(drop=True)


# =============================================================================
# v2 Task 9 -- class-A descriptor-space void metric (replaces class-A
# deficit scoring; C-9)
# =============================================================================


def class_a_descriptor_space(df: pd.DataFrame, proteins: list[str] | None = None) -> dict:
    """Task 9.1, restructured (v3 Task 16 / C-25): standardized protein-level
    descriptor PCA over CLASS_A_CONTINUOUS_AXES (MW, PI_mean, PI_range) only.

    kP, HCI, and C_Class are EXCLUDED here -- they are not measured
    per-molecule descriptors. Verified against the real CSV: the 6 values of
    Protein_class_type map to exactly 6 distinct (kP, HCI, C_Class) triples,
    with no triple shared across classes and no within-class variation
    (perfect bijection; placebo rows carry the sentinel (0.0, 0.0, 0.0)).
    The mapping is non-monotone (Bispecific has the highest kP, 5.0, but not
    the highest HCI, 1.3 vs Fc-Fusion's 1.5), which is why the v2 Pearson
    dedupe (0.86-0.91) nearly missed that this is categorical determinism,
    not a continuous relationship -- it cleared 0.85 by only 0.011. They
    remain in CLASS_A_AXES for Task 3's 1-D/2-D marginal census (a valid
    count of "how many clusters have kP in bin X" regardless of what kP
    represents) but no longer participate in the continuous PCA/void map,
    which is stratified by Protein_class_type instead (class_a_void_regions).

    `proteins`: optional subset of normalized protein identifiers to
    restrict to (Task 16 -- one format class's members). None = all 12
    (a pooled, format-blind view, kept only as descriptive context -- e.g.
    the pooled participation ratio, ~2.0 -- now that void search itself is
    per-class; see coverage_report.md).

    Any CLASS_A_CONTINUOUS_AXES column with zero variance within the
    selection is dropped before scaling (StandardScaler divides by std) --
    e.g. PI_range is constant within most single format classes. Returned
    as `dropped_axes` with each one's constant value, distinct from `axes`
    (the ones actually spanning the PCA)."""
    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])
    all_axes = list(CLASS_A_CONTINUOUS_AXES)
    per_protein = non_placebo.groupby("_pt")[all_axes].first().sort_index()
    if proteins is not None:
        per_protein = per_protein.loc[sorted(proteins)]

    axes = [a for a in all_axes if per_protein[a].std(ddof=0) > 1e-9]
    dropped_axes = {a: float(per_protein[a].iloc[0]) for a in all_axes if a not in axes}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(per_protein[axes].to_numpy(dtype=float))
    pca = PCA().fit(x_scaled)

    eigenvalues = pca.explained_variance_
    ratios = pca.explained_variance_ratio_
    participation_ratio = float((eigenvalues.sum() ** 2) / float(np.sum(eigenvalues**2))) if len(eigenvalues) else float("nan")
    cumulative = np.cumsum(ratios)
    n_components_90 = int(np.argmax(cumulative >= CLASS_A_VOID_VARIANCE_THRESHOLD) + 1) if len(ratios) else 0

    eigenspectrum = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(ratios))],
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": ratios,
            "cumulative_variance_ratio": cumulative,
        }
    )
    return {
        "axes": axes,
        "dropped_axes": dropped_axes,
        "per_protein": per_protein,
        "scaler": scaler,
        "pca": pca,
        "eigenspectrum": eigenspectrum,
        "participation_ratio": participation_ratio,
        "n_components_90": n_components_90,
    }


def class_a_categorical_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Task 16 categorical coverage statement: Protein_class_type IS the
    coverage question for kP/HCI/C_Class (C-25) -- there is no continuum
    between e.g. "Other" (kP=2.0) and "Polyclonal" (kP=3.0) to search a void
    map over. Reuses class_a_marginal's existing, already-correct count
    (Task 3) rather than recomputing it, and adds the
    eligible_for_void_map flag (Task 16.1: a class needs
    >=MIN_CLASS_SIZE_FOR_VOID_MAP members before a continuous search within
    it means anything -- below that, the class ITSELF, not a sub-region of
    it, is the reportable gap)."""
    marginal = class_a_marginal(df, CLASS_A_CATEGORICAL_AXIS)
    out = marginal[["bin", "n_clusters", "n_rows"]].rename(columns={"bin": "protein_class_type"})
    out["eligible_for_void_map"] = out["n_clusters"] >= MIN_CLASS_SIZE_FOR_VOID_MAP
    out = out.sort_values("n_clusters").reset_index(drop=True)
    return _attach_scope(out, {"n_clusters": SCOPE_BETWEEN_PROTEIN, "n_rows": SCOPE_ROW, "eligible_for_void_map": SCOPE_BETWEEN_PROTEIN})


def class_a_collinearity_groups(df: pd.DataFrame, proteins: list[str] | None = None, threshold: float = CLASS_A_COLLINEARITY_THRESHOLD) -> pd.DataFrame:
    """Task 9.2, restructured: single-linkage clustering of
    CLASS_A_CONTINUOUS_AXES (now 3, not 6 -- kP/HCI/C_Class are handled by
    class_a_categorical_coverage instead, C-25) at |r| >= threshold. `proteins`
    restricts to one format class's members, matching class_a_descriptor_space.

    On the real CSV, pooled over all 12: MW-PI_mean |r|=0.8347 (just under
    0.85 -- stays separate, not adjusted to force a merge); PI_range is
    weakly correlated with either (|r|<0.35). With kP/HCI/C_Class removed,
    this dedupe is currently a structural no-op (three singleton groups) --
    kept as a general-purpose safeguard, not because it currently merges
    anything.

    Below MIN_N_FOR_COLLINEARITY_DEDUPE points, no merge is attempted at
    all: a 2-variable correlation on that few points is not a stable
    estimate (mAb_IgG4's 3 members alone show |r(MW, PI_mean)|=0.93, purely
    because 3 points in 2D are nearly always near-collinear, not because the
    two are related within that format). Every format class in this dataset
    (max size 3) falls under this floor, so per-class calls always return
    singletons -- reported plainly rather than silently merging on noise."""
    space = class_a_descriptor_space(df, proteins=proteins)
    axes = space["axes"]
    n_points = len(space["per_protein"])
    if len(axes) < 2 or n_points < MIN_N_FOR_COLLINEARITY_DEDUPE:
        return pd.DataFrame([{"representative": a, "members": [a], "group_size": 1} for a in axes])
    corr = space["per_protein"][axes].corr(method="pearson").abs()

    parent = {a: a for a in axes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(axes):
        for b in axes[i + 1 :]:
            if corr.loc[a, b] >= threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for a in axes:
        groups.setdefault(find(a), []).append(a)

    rows = []
    for members in groups.values():
        members_sorted = sorted(members, key=axes.index)
        rows.append({"representative": members_sorted[0], "members": members_sorted, "group_size": len(members_sorted)})
    return pd.DataFrame(rows).sort_values("representative").reset_index(drop=True)


def _hull_interior_mask(hull_points: np.ndarray, query_points: np.ndarray) -> np.ndarray | None:
    """True where a query point falls inside (or on) the convex hull of
    hull_points. Returns None if the hull is degenerate (fewer points than
    dimensions+1, or collinear/coplanar -- e.g. a format class whose 3
    members happen to line up) -- callers treat None as "no interior is
    computable for this class," not an error, since that is itself an
    honest, reportable outcome (E21).

    1-D case handled directly (interval membership): scipy's Delaunay
    requires >= 2-D input and raises on a single column, which -- if routed
    through the same try/except as the degenerate-hull case below -- would
    silently report "no interior" for every k=1 class (common here: several
    format classes' 3 points project to >=90% variance on PC1 alone,
    e.g. mAb_IgG4 at 96.6%) even though a 1-D interior is perfectly
    well-defined (between the two extreme points)."""
    if hull_points.shape[1] == 1:
        lo, hi = hull_points.min(), hull_points.max()
        if hi <= lo:
            return None
        q = query_points.ravel()
        return (q >= lo) & (q <= hi)
    try:
        hull = Delaunay(hull_points)
    except Exception:
        return None
    return hull.find_simplex(query_points) >= 0


def class_a_void_regions(
    df: pd.DataFrame,
    feasibility_envelope: dict[str, tuple[float, float]] | None,
    leverage: str = DEFAULT_LEVERAGE_MODE,
    logo_residual_path: str | None = None,
    max_regions: int = CLASS_A_VOID_MAX_REGIONS,
) -> dict[str, pd.DataFrame]:
    """Task 9.3 + v3 Task 16: void map, stratified by Protein_class_type
    (C-17/C-25 -- a point interpolated across format classes in the old
    pooled 6-axis space was not a request for a real molecule) and split
    into interior/exterior tables (Task 16.3), each screened against
    `feasibility_envelope` (Task 16.1/16.2).

    `feasibility_envelope`: REQUIRED, no default, never inferred from the
    observed data range -- see FEASIBILITY_ENVELOPE_REQUIRED_AXES. Passing
    None or an incomplete mapping raises, naming the missing axes.

    Only format classes with >= MIN_CLASS_SIZE_FOR_VOID_MAP members get a
    continuous search (class_a_categorical_coverage reports the rest as
    class-level gaps). Within an eligible class: same PCA-whiten-grid-NN-NMS
    mechanism as v2's pooled version, run on that class's own points only.
    Each candidate region is classified interior/exterior via the class's
    own convex hull (Delaunay membership; a degenerate/collinear hull -- too
    few effective dimensions -- yields no interior candidates for that
    class, not an error). Regions with any back-projected descriptor range
    outside `feasibility_envelope` are discarded entirely (not down-ranked).

    Returns {'interior_gaps': df, 'extrapolation_targets': df}, each capped
    at max_regions, ranked by void_score, tagged with `protein_class_type`
    and (for extrapolation_targets) which axes exit the hull."""
    if feasibility_envelope is None:
        raise ValueError(
            "class_a_void_regions: feasibility_envelope is required (Task 16.1) -- "
            f"supply (min, max) for each of {FEASIBILITY_ENVELOPE_REQUIRED_AXES}. "
            "Never defaulted to the observed data range or left unbounded."
        )
    missing = [a for a in FEASIBILITY_ENVELOPE_REQUIRED_AXES if a not in feasibility_envelope]
    if missing:
        raise ValueError(f"class_a_void_regions: feasibility_envelope is missing bounds for {missing}.")

    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])
    non_placebo["log_v1000"] = np.log10(non_placebo["Viscosity_1000"].where(non_placebo["Viscosity_1000"] > 0))
    global_min = float(non_placebo["log_v1000"].min())
    global_max = float(non_placebo["log_v1000"].max())

    residual_df = None
    if leverage == "logo_residual":
        residual_df, _ = _load_logo_residual_artifact(logo_residual_path)
    lev_fn = _leverage_fn(leverage, residual_df, global_min, global_max)

    coverage = class_a_categorical_coverage(df)
    eligible_classes = coverage.loc[coverage["eligible_for_void_map"], "protein_class_type"].tolist()

    interior_rows: list[dict] = []
    exterior_rows: list[dict] = []

    for class_name in eligible_classes:
        class_proteins = sorted(non_placebo.loc[non_placebo["Protein_class_type"] == class_name, "_pt"].unique())
        space = class_a_descriptor_space(df, proteins=class_proteins)
        axes, per_protein, scaler, pca = space["axes"], space["per_protein"], space["scaler"], space["pca"]
        if len(axes) == 0:
            continue  # every continuous axis constant within this class -- nothing to search
        representatives = class_a_collinearity_groups(df, proteins=class_proteins)["representative"].tolist()
        k = max(1, min(space["n_components_90"], len(axes)))

        scores_full = pca.transform(scaler.transform(per_protein[axes].to_numpy(dtype=float)))
        scores = scores_full[:, :k]
        eigvals = pca.explained_variance_[:k]
        whitened = scores / np.sqrt(eigvals)

        mins = whitened.min(axis=0)
        maxs = whitened.max(axis=0)
        span = np.maximum(maxs - mins, 1e-9)
        pad = span * CLASS_A_VOID_GRID_MARGIN
        axes_grid = [np.linspace(mins[i] - pad[i], maxs[i] + pad[i], CLASS_A_VOID_GRID_POINTS_PER_AXIS) for i in range(k)]
        mesh = np.stack(np.meshgrid(*axes_grid, indexing="ij"), axis=-1).reshape(-1, k)

        interior_mask = _hull_interior_mask(whitened, mesh)  # None if degenerate hull

        nn = NearestNeighbors(n_neighbors=1).fit(whitened)
        dist, idx = nn.kneighbors(mesh)
        dist = dist.ravel()
        nearest_protein = per_protein.index.to_numpy()[idx.ravel()]

        protein_leverage = {
            pt: lev_fn(non_placebo.loc[non_placebo["_pt"] == pt, "log_v1000"], "class_a_void", pt) for pt in per_protein.index
        }
        point_leverage = np.array([protein_leverage[p] for p in nearest_protein])
        void_score = dist * point_leverage

        grid_step = np.array([axes_grid[d][1] - axes_grid[d][0] for d in range(k)])
        half_width = grid_step / 2.0

        def _back_project(center) -> dict[str, tuple[float, float]]:
            corners = np.array(list(itertools.product(*[(c - hw, c + hw) for c, hw in zip(center, half_width)])))
            full_corners = np.zeros((len(corners), len(axes)))
            full_corners[:, :k] = corners * np.sqrt(eigvals)
            descriptor_corners = scaler.inverse_transform(pca.inverse_transform(full_corners))
            ranges = {axis: (float(descriptor_corners[:, j].min()), float(descriptor_corners[:, j].max())) for j, axis in enumerate(axes)}
            for axis, value in space["dropped_axes"].items():
                ranges[axis] = (value, value)
            return ranges

        def _within_envelope(ranges: dict[str, tuple[float, float]]) -> bool:
            for axis in CLASS_A_CONTINUOUS_AXES:
                lo, hi = ranges[axis]
                env_lo, env_hi = feasibility_envelope[axis]
                if lo < env_lo or hi > env_hi:
                    return False
            return True

        for is_interior_pass, bucket, score_arr in [(True, interior_rows, void_score), (False, exterior_rows, void_score)]:
            if is_interior_pass:
                if interior_mask is None:
                    continue  # degenerate hull: no interior concept for this class
                candidate_mask = interior_mask
            else:
                # Degenerate hull -> every grid point is exterior (no interior to exclude).
                candidate_mask = ~interior_mask if interior_mask is not None else np.ones(len(mesh), dtype=bool)

            order = np.argsort(-score_arr)
            kept_idx: list[int] = []
            claimed_proteins: set[str] = set()
            for i in order:
                if not candidate_mask[i] or score_arr[i] <= 0:
                    continue
                p_id = str(nearest_protein[i])
                if p_id in claimed_proteins:
                    continue
                p = mesh[i]
                if any(np.linalg.norm(p - mesh[j]) < CLASS_A_VOID_MIN_SEPARATION for j in kept_idx):
                    continue
                ranges = _back_project(p)
                if not _within_envelope(ranges):
                    continue  # 16.2: discarded entirely, never down-ranked
                claimed_proteins.add(p_id)
                kept_idx.append(int(i))
                if len(kept_idx) >= max_regions:
                    break

            for i in kept_idx:
                ranges = _back_project(mesh[i])
                row = {
                    "protein_class_type": class_name,
                    "void_score": float(score_arr[i]),
                    "nearest_neighbour_distance": float(dist[i]),
                    "response_leverage": float(point_leverage[i]),
                    "nearest_protein": str(nearest_protein[i]),
                }
                exits_hull_on = []
                for axis in CLASS_A_CONTINUOUS_AXES:
                    lo, hi = ranges[axis]
                    if axis in representatives or axis in space["dropped_axes"]:
                        row[f"{axis}_min"] = lo
                        row[f"{axis}_max"] = hi
                    obs_lo, obs_hi = per_protein[axis].min(), per_protein[axis].max()
                    if lo < obs_lo or hi > obs_hi:
                        exits_hull_on.append(axis)
                if not is_interior_pass:
                    row["hull_status"] = "exterior"
                    row["exits_observed_range_on"] = ", ".join(exits_hull_on)
                bucket.append(row)

    def _finalize(rows: list[dict]) -> pd.DataFrame:
        out = pd.DataFrame(rows)
        if len(out):
            out = out.sort_values("void_score", ascending=False).reset_index(drop=True)
            out.insert(0, "rank", range(1, len(out) + 1))
        return _attach_scope(
            out,
            {"void_score": SCOPE_BETWEEN_PROTEIN, "nearest_neighbour_distance": SCOPE_BETWEEN_PROTEIN, "response_leverage": SCOPE_ROW},
        )

    interior_gaps = _finalize(interior_rows)
    extrapolation_targets = _finalize(exterior_rows)
    for out in (interior_gaps, extrapolation_targets):
        out.attrs["feasibility_envelope"] = dict(feasibility_envelope)
        out.attrs["eligible_classes"] = eligible_classes
    return {"interior_gaps": interior_gaps, "extrapolation_targets": extrapolation_targets}


def class_a_deficit(*_args, **_kwargs):
    """Retired (C-9): class-A ranking is degenerate at partition-cell scope
    by construction -- class-A bins partition the 12 clusters
    (class_a_marginal's own assertion), so total deficit for a k-bin axis is
    fixed at REQUIRED_CLUSTERS_GENERALIZATION*k - N_CLUSTERS_TOTAL regardless
    of the data, and within-axis ranking degenerates to -observed_clusters.
    Deleted, not deprecated: this raises so any surviving call site is
    caught immediately rather than silently reintroducing C-9. Use
    class_a_void_regions() instead."""
    raise RuntimeError(
        "class_a_deficit was retired in v2 (C-9): class-A partition-cell counts sum to "
        "N_CLUSTERS_TOTAL by construction, so REQUIRED_CLUSTERS_GENERALIZATION deficit is "
        "fixed at 10k-12 for any k-bin axis regardless of the data. Use class_a_void_regions() "
        "instead."
    )


# =============================================================================
# v2 Task 10/16 -- Pareto frontier (endogenous leverage) / rank-checked
# scalar (exogenous leverage). C-16: score = deficit * leverage as a single
# scalar is retired for "range" -- a two-factor product is governed by
# whichever factor has the larger relative RANK spread, and no leverage
# definition escapes that except by coincidence (C-10, rescoped to class-A
# in v2 -- see coverage_report.md; the deleted high_visc_share, C-15, would
# have mirrored it in the opposite direction). leverage="logo_residual" is
# exogenous (per-bin model error, not a function of the response observed in
# the bin) and remains the one mode that still produces a defensible scalar
# score -- rank-checked, not trusted blindly (check_score_non_degeneracy).
# =============================================================================


def _pareto_frontier_mask(deficit: np.ndarray, leverage: np.ndarray) -> np.ndarray:
    """True where (deficit, leverage) is NOT dominated by any other entry in
    the same set (dominated = some other entry has >= both coordinates, with
    > on at least one). The non-dominated set is the Pareto frontier."""
    n = len(deficit)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        ge = (deficit >= deficit[i]) & (leverage >= leverage[i])
        gt = (deficit > deficit[i]) | (leverage > leverage[i])
        dominates_i = ge & gt
        dominates_i[i] = False
        dominated[i] = bool(dominates_i.any())
    return ~dominated


def check_score_non_degeneracy(score: pd.Series, deficit: pd.Series, leverage: pd.Series) -> dict:
    """Task 10.1, restated as rank-based and two-sided (C-16): for a scalar
    score = deficit * leverage (logo_residual mode only -- the Pareto-
    frontier default has no scalar to check), the product is governed by
    whichever factor has the larger RANK spread; an IQR check on leverage
    alone doesn't catch that (a factor can have plenty of spread and still
    dominate the order -- this is exactly how the deleted high_visc_share
    would have failed: corr(score, leverage) = 0.92 on the real post-Task-9
    data even though IQR alone flagged it too). Computes Kendall's tau
    between score and each factor; raises if either exceeds KENDALL_TAU_MAX
    in absolute value -- score is then a monotone restatement of that one
    factor, not a joint measure of both."""
    tau_deficit, _ = _scipy_stats.kendalltau(score, deficit)
    tau_leverage, _ = _scipy_stats.kendalltau(score, leverage)
    result = {"tau_score_deficit": float(tau_deficit), "tau_score_leverage": float(tau_leverage)}
    offending = {name: v for name, v in [("deficit", tau_deficit), ("leverage", tau_leverage)] if abs(v) > KENDALL_TAU_MAX}
    if offending:
        raise ValueError(
            f"score non-degeneracy guard failed: {offending} exceed KENDALL_TAU_MAX={KENDALL_TAU_MAX} -- "
            f"score is a monotone restatement of that factor alone "
            f"(tau_score_deficit={tau_deficit:.3f}, tau_score_leverage={tau_leverage:.3f})."
        )
    return result


# =============================================================================
# v2 Task 6/11/12 -- gap report: two remedy classes, never combined into one
# scored list (Task 11: they draw on different acquisition budgets)
# =============================================================================


def _finalize_remedy_table(entries: list[dict], remedy: str, leverage: str, deficit_scope: str) -> pd.DataFrame:
    """Assembles one remedy class's gap entries. leverage='range' (or any
    endogenous mode) gets a Pareto frontier over (deficit, leverage) -- no
    scalar score (C-16). leverage='logo_residual' (exogenous) gets the
    classic score=deficit*leverage, rank-checked before being trusted
    (check_score_non_degeneracy)."""
    out = pd.DataFrame(entries)
    out["remedy"] = remedy
    scope_map = {
        "observed_clusters": deficit_scope,
        "required_clusters": deficit_scope,
        "deficit": deficit_scope,
        "leverage": SCOPE_ROW,
    }
    if len(out) == 0:
        out["on_frontier"] = pd.Series(dtype=bool)
        return _attach_scope(out, scope_map)

    if leverage == "logo_residual":
        out["score"] = out["deficit"] * out["leverage"]
        guard = check_score_non_degeneracy(out["score"], out["deficit"], out["leverage"])
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        out.attrs["score_non_degeneracy"] = guard
        scope_map["score"] = deficit_scope
    else:
        out["on_frontier"] = _pareto_frontier_mask(out["deficit"].to_numpy(dtype=float), out["leverage"].to_numpy(dtype=float))
        out = out.sort_values(["on_frontier", "deficit", "leverage"], ascending=[False, False, False]).reset_index(drop=True)
        scope_map["on_frontier"] = deficit_scope

    return _attach_scope(out, scope_map)


def build_gap_report(df: pd.DataFrame, leverage: str = DEFAULT_LEVERAGE_MODE, logo_residual_path: str | None = None) -> dict[str, pd.DataFrame]:
    """v2 Tasks 6/9/11/12/16: replaces v1's build_gap_ranking. Returns:
      - 'void_regions': class-A descriptor-space void map (Task 9.3), <=5 rows.
      - 'acquire_cluster': class-B level gaps + Fv pseudo-level (Task 11) --
        needs a protein not currently in the dataset.
      - 'acquire_rows_within_cluster': class-C condition-bin + shear-channel
        gaps (Task 11) -- satisfiable with new formulations of an existing
        protein. Restricted to axes admitted by the Task 12.1
        design-variation gate.
      - 'not_varied_by_design': class-C axes that failed that gate
        (Temperature) -- reported, never ranked.
    """
    residual_df = None
    residual_hash = None
    if leverage == "logo_residual":
        residual_df, residual_hash = _load_logo_residual_artifact(logo_residual_path)

    non_placebo = df.loc[~_is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = _normalize(non_placebo["Protein_type"])
    non_placebo["log_v1000"] = np.log10(non_placebo["Viscosity_1000"].where(non_placebo["Viscosity_1000"] > 0))
    global_min = float(non_placebo["log_v1000"].min())
    global_max = float(non_placebo["log_v1000"].max())
    lev_fn = _leverage_fn(leverage, residual_df, global_min, global_max)

    void_regions = class_a_void_regions(df, leverage=leverage, logo_residual_path=logo_residual_path)

    # --- acquire_cluster: class B + Fv pseudo-level -----------------------
    cluster_entries = []
    for axis in CLASS_B_AXES:
        levels = class_b_levels(df, axis)
        level_col = _normalize_level(non_placebo[axis])
        for _, row in levels.iterrows():
            deficit = max(0.0, compare_to_threshold(row["n_clusters"], REQUIRED_CLUSTERS_GENERALIZATION, unit_of_comparison=UNIT_FEATURE_PRESENCE))
            if deficit == 0:
                continue
            lev = lev_fn(non_placebo.loc[level_col == row["level"], "log_v1000"], axis, row["level"])
            cluster_entries.append(
                {
                    "axis": axis,
                    "bin": row["level"],
                    "observed_clusters": row["n_clusters"],
                    "required_clusters": REQUIRED_CLUSTERS_GENERALIZATION.value,
                    "deficit": deficit,
                    "leverage": lev,
                }
            )

    fv = class_b_fv_pseudo_level(df)
    fv_row = fv.loc[fv["level"] == "fv_defined"].iloc[0]
    fv_deficit = max(
        0.0, compare_to_threshold(fv_row["n_clusters"], REQUIRED_CLUSTERS_GENERALIZATION, unit_of_comparison=UNIT_FEATURE_PRESENCE)
    )
    if fv_deficit > 0:
        fv_mask = non_placebo[FV_CHARGE_COL].notna()
        lev = lev_fn(non_placebo.loc[fv_mask, "log_v1000"], FV_PSEUDO_AXIS, "fv_defined")
        cluster_entries.append(
            {
                "axis": FV_PSEUDO_AXIS,
                "bin": "fv_defined",
                "observed_clusters": fv_row["n_clusters"],
                "required_clusters": REQUIRED_CLUSTERS_GENERALIZATION.value,
                "deficit": fv_deficit,
                "leverage": lev,
            }
        )
    acquire_cluster = _finalize_remedy_table(cluster_entries, REMEDY_ACQUIRE_CLUSTER, leverage, SCOPE_BETWEEN_PROTEIN)

    # --- acquire_rows_within_cluster: class C, gated by Task 12.1 ---------
    gate = class_c_design_variation_gate(df)
    admitted_axes = gate.loc[gate["admitted"], "axis"].tolist()
    not_varied = gate.loc[~gate["admitted"]].reset_index(drop=True)

    condition_entries = []
    for axis in admitted_axes:
        grid = class_c_grid(df, axis)
        n_bins = len(CLASS_C_AXES[axis]) - 1
        bin_threshold = ScopedThreshold(value=float(n_bins), scope=SCOPE_WITHIN_PROTEIN, unit_of_comparison=UNIT_CONDITION_BIN)
        per_protein, _ = class_c_grid_completeness(grid, axis)
        for _, row in per_protein.iterrows():
            occupied = int(round(row["completeness"] * n_bins))
            deficit = max(0.0, compare_to_threshold(occupied, bin_threshold, unit_of_comparison=UNIT_CONDITION_BIN))
            if deficit <= 0:
                continue
            # Leverage of a missing condition-bin cell is the leverage of
            # the CLUSTER it belongs to (the empty cell itself has no rows
            # to measure over) -- how much this protein's existing viscosity
            # footprint matters is what makes filling its gap worthwhile.
            lev = lev_fn(non_placebo.loc[non_placebo["_pt"] == row["protein"], "log_v1000"], axis, row["protein"])
            condition_entries.append(
                {
                    "axis": axis,
                    "bin": row["protein"],
                    "observed_clusters": occupied,
                    "required_clusters": n_bins,
                    "deficit": deficit,
                    "leverage": lev,
                }
            )

    shear_grid = class_c_shear_channel_grid(df)
    n_channels = len(SHEAR_CHANNEL_COLS)
    channel_threshold = ScopedThreshold(value=float(n_channels), scope=SCOPE_WITHIN_PROTEIN, unit_of_comparison=UNIT_CONDITION_BIN)
    occupied_channels = (
        shear_grid.loc[shear_grid["n_rows"] > 0].groupby("protein")["bin"].nunique().reindex(sorted(non_placebo["_pt"].unique()), fill_value=0)
    )
    for pt, occupied in occupied_channels.items():
        deficit = max(0.0, compare_to_threshold(int(occupied), channel_threshold, unit_of_comparison=UNIT_CONDITION_BIN))
        if deficit <= 0:
            continue
        lev = lev_fn(non_placebo.loc[non_placebo["_pt"] == pt, "log_v1000"], "shear_channel", pt)
        condition_entries.append(
            {
                "axis": "shear_channel",
                "bin": pt,
                "observed_clusters": int(occupied),
                "required_clusters": n_channels,
                "deficit": deficit,
                "leverage": lev,
            }
        )

    acquire_rows_within_cluster = _finalize_remedy_table(condition_entries, REMEDY_ACQUIRE_ROWS_WITHIN_CLUSTER, leverage, SCOPE_WITHIN_PROTEIN)

    result = {
        "void_regions": void_regions,
        "acquire_cluster": acquire_cluster,
        "acquire_rows_within_cluster": acquire_rows_within_cluster,
        "not_varied_by_design": not_varied,
    }
    for table in result.values():
        table.attrs["leverage_mode"] = leverage
    if residual_hash is not None:
        for name in ("acquire_cluster", "acquire_rows_within_cluster"):
            result[name].attrs["logo_residual_artifact_hash"] = residual_hash
    return result


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--out-dir", default="reports/coverage")
    parser.add_argument("--leverage", default=DEFAULT_LEVERAGE_MODE, choices=["range", "logo_residual"])
    parser.add_argument("--logo-residual-path", default=None)
    args = parser.parse_args(argv)

    df = load_dataset(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    report = build_gap_report(df, leverage=args.leverage, logo_residual_path=args.logo_residual_path)
    for name, table in report.items():
        table.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    print(
        f"Loaded {len(df)} rows. Wrote void_regions ({len(report['void_regions'])}), "
        f"acquire_cluster ({len(report['acquire_cluster'])}), "
        f"acquire_rows_within_cluster ({len(report['acquire_rows_within_cluster'])}), "
        f"not_varied_by_design ({len(report['not_varied_by_design'])}) to {args.out_dir}/."
    )


if __name__ == "__main__":
    main()
