"""
charge_features.py
==================
Rung-2 representation upgrade: turn the two new protein-level columns

    "Charge"     -> net protein charge at the FORMULATION pH  (signed, from the
                    sequence titration curve; f(protein, pH) only)
    "ProtPi PI"  -> theoretical (sequence-derived) isoelectric point

into a small block of *physically load-bearing* features instead of feeding the
two raw columns straight into the numeric pipeline.

WHY NOT JUST ADD THE TWO RAW COLUMNS
------------------------------------
Empirically (on formulation_data_05262026_DFChg.csv, 408 protein rows):

    corr(log_visc,  signed Charge)             = -0.09   (≈ noise)
    corr(log_visc, |Charge|)                   = -0.13
    corr(log_visc, |Charge| * Protein_conc)    = +0.33
    corr(log_visc,  near_pI)                   = +0.12
    corr(log_visc,  near_pI * Protein_conc)    = +0.56   <-- almost as strong
    corr(log_visc,  Protein_conc)              = +0.78        as concentration

The physics: high-concentration viscosity is driven by protein–protein
interaction. That interaction is governed by the *magnitude* of net charge
(repulsion strength, i.e. |charge|), not its sign, and it is U-shaped in pH:
when the net charge is near zero (pH ≈ pI) charge–charge repulsion vanishes,
attraction/self-association dominates, and viscosity spikes. Crucially this only
matters when the protein is *crowded*, so the effect lives in the
charge × concentration interaction — which is exactly the term that correlates.

Handing the network the signed scalar forces it to (a) discover the absolute
value through ReLUs and (b) discover the concentration coupling, both from ~400
rows. We instead expose |charge|, a smooth near-pI indicator, and their products
with concentration directly. StandardScaler normalises everything downstream, so
only the *relations* need to be right.

WHY IT MATTERS FOR EXTRAPOLATION (the ibalizumab held-out benchmark)
--------------------------------------------------------------------
Protein_type is one-hot, so a *held-out* protein arrives as an all-zero code:
the model has no transferable handle on its identity. `theo_pI` and the charge
block ARE that handle — they are computable a-priori for any sequence at any pH
(no viscosity label involved, so no leakage), and they place a novel protein at
its true physical coordinates. This is precisely the Rung-2/3 protein descriptor
the earlier one-hot-only setup was missing, and it should help most at 0–2 shots
on the left end of the learning curve, where context is too sparse to compensate.

TWO TRAPS THIS MODULE IS BUILT TO AVOID  (both bit the earlier Rung-1 attempt)
------------------------------------------------------------------------------
1. fillna(0) is WRONG for charge. Zero net charge is not "missing" — it is the
   single highest-viscosity-risk state (pH == pI). Silently zero-filling an
   uncomputed protein injects a strong, false "at-pI" signal. We therefore
   distinguish a genuine zero (no protein present) from an unknown, emit a
   `charge_missing` flag, and only ever impute behind that flag.
2. Do NOT let these columns be masked to zero during training. For the same
   reason (0 == "at pI", a specific physical point, not a neutral null), every
   column here must join PROTECTED_FEATURE_NAMES in the trainer. See README.

USAGE
-----
    from charge_features import featurize_charge, CHARGE_FEATURE_COLS

    df = normalize_charge_columns(df)          # rename messy CSV headers
    df, charge_cols = featurize_charge(df)      # append the feature block
    num_cols.extend(charge_cols)                # add to the StandardScaler group
    PROTECTED_FEATURE_NAMES |= set(charge_cols) # never mask (see trap #2)

visqai.preprocessing.pipeline.build_feature_frame is the single shared caller
for both training and inference now -- there is no separate copy to mirror.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.features.categorical import SALT_PROPS

# ---------------------------------------------------------------------------
# Config. Defaults chosen from the data (net charge spans ~ -17 .. +50).
# ---------------------------------------------------------------------------
# Width (in units of net charge) of the "near the isoelectric point" window.
# near_pI = exp(-charge^2 / (2 * NEAR_PI_SIGMA^2)); ~8 charges ≈ the scale over
# which charge screening/attraction turns on. Configurable; StandardScaler makes
# the exact value non-critical, but it sets where the bump peaks vs saturates.
NEAR_PI_SIGMA: float = 8.0

# Whether to also emit the ionic-strength-screened charge magnitude. Salt (and
# buffer) screen electrostatic repulsion (shorter Debye length). This is a
# monotone proxy, not a real Debye calc — we lack full speciation. Optional.
ADD_SCREENED_CHARGE: bool = True

# Approximate ionic-strength valence factor per buffer species at
# formulation pH (I = 0.5 * sum(c_i * z_i^2); for a 1:z electrolyte this is
# well approximated by c * z). We lack full pKa-based speciation, so these
# are order-of-magnitude: histidine/acetate are ~monovalent near their pKa,
# phosphate/PBS carry more charge per mole (HPO4^2-/H2PO4- mix), citrate is
# trivalent. Unrecognized buffers default to 1.0 (monovalent-equivalent),
# the conservative middle case. Salt uses SALT_PROPS' salt_valence instead
# of a separate table, so both categoricals share one physical-chemistry
# source of truth.
BUFFER_IONIC_VALENCE: dict[str, float] = {
    "histidine": 1.0,
    "acetate": 1.0,
    "phosphate": 2.0,
    "pbs": 2.0,
    "citrate": 3.0,
}

# Physically-generous ceiling on total ionic strength (M): well above
# anything in the training data (max observed ~0.28 M salt+buffer combined).
# ionic_M is clipped to this before computing charge_screened so that a
# held-out ingredient's concentration (e.g. a leave-one-ingredient-out salt
# fold, where Salt_conc jumps from 0 in every training row to 32-175 mM in
# every held-out row) can't push the feature into a raw-unit range the rest
# of the pipeline never had reason to represent, on top of whatever the
# fold's own StandardScaler was fit to.
ION_STRENGTH_CAP_M: float = 0.5

# Salt's valence factor reuses SALT_PROPS' salt_valence (categorical.py) so
# salt and buffer ionic-strength contributions share one physical-chemistry
# source of truth instead of two independently-maintained tables.
SALT_PROPS_VALENCE: dict[str, float] = {k: v["salt_valence"] for k, v in SALT_PROPS.items() if k != "none"}

# Rough imputation for a protein whose charge was never computed but whose pI is
# known: sign & slope from a linear titration surrogate around the pI. ONLY used
# behind charge_missing=1, and marked so the model can discount it. Prefer
# computing the real value upstream (you already have the tool that made Charge).
IMPUTE_SLOPE: float = 6.0  # charges per pH unit away from pI (order-of-magnitude)


# Canonical output columns, in fixed order (defines position for protected-index
# bookkeeping). Screened column is appended conditionally.
CHARGE_FEATURE_COLS_BASE = [
    "net_charge",  # signed net charge @ formulation pH
    "abs_charge",  # |net_charge| — repulsion magnitude
    "charge_x_conc",  # signed charge × concentration
    "abscharge_x_conc",  # |charge| × concentration  (interaction that matters)
    "near_pI",  # smooth at-isoelectric-point indicator (peaks at 0)
    "nearpI_x_conc",  # near_pI × concentration   (strongest charge-derived)
    "theo_pI",  # sequence-derived isoelectric point (protein anchor)
    "pI_gap",  # PI_mean − theo_pI  (experimental vs theoretical gap)
    "charge_missing",  # 1.0 if net_charge was unknown & imputed, else 0.0
]


def _numeric_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """pd.to_numeric(df[col]) if present, else a `default`-filled Series over
    df's index. df.get(col, default) returns a bare scalar (not a Series)
    when `col` is entirely absent, which breaks every downstream .fillna()/
    .copy()/boolean-mask call in featurize_charge -- this guarantees a Series
    either way, which is what "degrades gracefully if absent" actually
    requires (this module's own docstring promise for older CSVs with no
    Charge/ProtPi PI columns)."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _ionic_valence(type_col, index: pd.Index, valence_table: dict[str, float]) -> pd.Series:
    """Map a categorical type column (Salt_type/Buffer_type) to an
    approximate ionic-strength valence per row via substring match against
    `valence_table` (mirrors categorical._lookup's matching). Absent/none
    -> 0.0 (no ionic contribution). A present-but-unrecognized category
    defaults to 1.0 (monovalent-equivalent), the conservative middle case."""
    if type_col is None:
        return pd.Series(0.0, index=index)
    norm = type_col.astype(str).str.strip().str.lower()

    def lookup(s: str) -> float:
        if s in ("none", "nan", "unknown", "na", "n/a", ""):
            return 0.0
        match = next((v for k, v in valence_table.items() if k in s), None)
        return match if match is not None else 1.0

    return norm.map(lookup)


def normalize_charge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw CSV headers ('Charge', 'ProtPi PI', stray 'Unnamed: *')
    to the internal names this module expects. Idempotent and safe if already
    renamed or if the columns are absent (older CSVs)."""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    ren = {}
    for cand in ("Charge", "charge", "Net_charge", "NetCharge"):
        if cand in df.columns:
            ren[cand] = "_raw_charge"
            break
    for cand in ("ProtPi PI", "ProtPi_PI", "TheoPI", "Theo_PI", "Theoretical_PI", "Theoretical PI"):
        if cand in df.columns:
            ren[cand] = "_raw_theo_pi"
            break
    return df.rename(columns=ren)


def _protein_present(df: pd.DataFrame) -> pd.Series:
    """True where a protein is actually in the well (so charge==0 is real, not
    missing). Uses Protein_type != none/unknown, backed up by Protein_conc>0."""
    if "Protein_type" in df.columns:
        pt = df["Protein_type"].astype(str).str.strip().str.lower()
        present = ~pt.isin(["none", "unknown", "nan", "na", "", "n/a"])
    else:
        present = pd.Series(True, index=df.index)
    if "Protein_conc" in df.columns:
        conc = pd.to_numeric(df["Protein_conc"], errors="coerce").fillna(0.0)
        present = present | (conc > 0)
    return present


def featurize_charge(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Append the charge/pI feature block. Expects `normalize_charge_columns` to
    have run (looks for `_raw_charge`, `_raw_theo_pi`), but degrades gracefully
    if those are absent (whole block becomes the physically-correct null).

    Returns (df_out, cols) where `cols` are the new numeric columns to add to the
    StandardScaler group AND to PROTECTED_FEATURE_NAMES.
    """
    df = df.copy()

    conc = _numeric_col(df, "Protein_conc", 0.0).fillna(0.0)
    ph = _numeric_col(df, "Buffer_pH", 7.0).fillna(7.0)
    pi_mean = _numeric_col(df, "PI_mean", np.nan)

    raw_charge = _numeric_col(df, "_raw_charge", np.nan)
    theo_pi = _numeric_col(df, "_raw_theo_pi", np.nan)

    present = _protein_present(df)

    # --- resolve net charge + missingness -------------------------------------
    # Genuine zero: no protein present -> charge is truly 0, NOT missing.
    net_charge = raw_charge.copy()
    net_charge[~present] = 0.0

    missing = present & net_charge.isna()

    # Behind the flag, optionally impute a crude signed charge from a pI so the
    # decoder still gets sign + rough slope rather than a hard (false) zero.
    pi_for_impute = theo_pi.where(theo_pi.notna(), pi_mean)
    impute_val = IMPUTE_SLOPE * (pi_for_impute - ph)  # +ve below pI, matches data
    net_charge[missing] = impute_val[missing]
    # Anything still NaN (no pI at all): fall back to 0 but keep the flag on.
    net_charge = net_charge.fillna(0.0)

    abs_charge = net_charge.abs()

    # --- theoretical pI + experimental/theoretical gap ------------------------
    # For a null/absent protein, pI at the origin (0) is the consistent null,
    # matching how the rest of the pipeline zeroes absent-ingredient properties.
    theo_pi_filled = theo_pi.copy()
    theo_pi_filled[~present] = 0.0
    theo_pi_filled = theo_pi_filled.fillna(pi_mean).fillna(0.0)

    pi_gap = pi_mean - theo_pi_filled
    pi_gap[~present] = 0.0
    pi_gap = pi_gap.fillna(0.0)

    # --- smooth near-isoelectric-point indicator (peaks at net_charge == 0) ---
    near_pI = np.exp(-(net_charge.values**2) / (2.0 * NEAR_PI_SIGMA**2))
    near_pI = pd.Series(near_pI, index=df.index)
    # A well with no protein is not "at its pI" in any meaningful sense; zero it
    # so near_pI can't fire on buffer-only rows.
    near_pI[~present] = 0.0

    # --- assemble ------------------------------------------------------------
    df["net_charge"] = net_charge.values
    df["abs_charge"] = abs_charge.values
    df["charge_x_conc"] = net_charge.values * conc.values
    df["abscharge_x_conc"] = abs_charge.values * conc.values
    df["near_pI"] = near_pI.values
    df["nearpI_x_conc"] = near_pI.values * conc.values
    df["theo_pI"] = theo_pi_filled.values
    df["pI_gap"] = pi_gap.values
    df["charge_missing"] = missing.astype(float).values

    cols = list(CHARGE_FEATURE_COLS_BASE)

    if ADD_SCREENED_CHARGE:
        # Ionic-strength proxy from salt + buffer (mM), valence-weighted
        # (I = 0.5 * sum(c_i * z_i^2) ~ c * z for a 1:z electrolyte) rather
        # than a flat 1:1 sum of raw mM -- a divalent/trivalent buffer or
        # salt contributes more ionic strength per mM than a monovalent one.
        # Real Debye screening would still need full pH-dependent speciation;
        # this valence-weighted surrogate is enough for the NN to learn "salt
        # screens the charge effect." Repulsion ↓ as sqrt(I) ↑.
        salt = _numeric_col(df, "Salt_conc", 0.0).fillna(0.0)
        buf = _numeric_col(df, "Buffer_conc", 0.0).fillna(0.0)
        salt_z = _ionic_valence(df.get("Salt_type"), df.index, SALT_PROPS_VALENCE)
        buf_z = _ionic_valence(df.get("Buffer_type"), df.index, BUFFER_IONIC_VALENCE)
        ionic_M = (salt.values * salt_z.values + buf.values * buf_z.values) / 1000.0  # mM -> M
        ionic_M = np.clip(ionic_M, 0.0, ION_STRENGTH_CAP_M)
        screened = abs_charge.values / (1.0 + np.sqrt(ionic_M))
        df["charge_screened"] = screened
        cols.append("charge_screened")

    return df, cols


def charge_coupling_index(df: pd.DataFrame, c_class_col: str = "C_Class") -> pd.Series:
    """
    Physical replacement for the hand-rolled cci used to pick the Near-pI/Mixed/
    Far regime in process_row_features / inference._calculate_cci.

        old:  cci = C_Class * exp(-|pH - PI_mean| / tau)      # pH-distance proxy
        new:  cci = C_Class * exp(-net_charge^2 / (2 sigma^2)) # real net charge

    Same shape (peaks when the protein is at its isoelectric point) but driven by
    the actual computed net charge instead of a Gaussian on |pH - pI_mean|.
    Requires featurize_charge to have run (uses `net_charge`). Falls back to the
    old proxy where net_charge is unavailable.
    """
    c_class = pd.to_numeric(df.get(c_class_col, 1.0), errors="coerce").fillna(1.0)
    if "net_charge" in df.columns:
        nc = pd.to_numeric(df["net_charge"], errors="coerce").fillna(0.0)
        coupling = np.exp(-(nc.values**2) / (2.0 * NEAR_PI_SIGMA**2))
    else:  # graceful fallback to the legacy proxy
        ph = pd.to_numeric(df.get("Buffer_pH", 7.0), errors="coerce").fillna(7.0)
        pi = pd.to_numeric(df.get("PI_mean", 7.0), errors="coerce").fillna(7.0)
        coupling = np.exp(-(ph - pi).abs().values / 1.5)
    return pd.Series(c_class.values * coupling, index=df.index)


def audit(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Quick sanity table: run the block and show coverage + ranges."""
    df = normalize_charge_columns(df_raw)
    df, cols = featurize_charge(df)
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "feature": c,
                "nonzero": int((s != 0).sum()),
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
            }
        )
    return pd.DataFrame(rows)
