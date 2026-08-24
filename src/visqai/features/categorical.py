"""
categorical_features.py
=======================
Rung-1 representation upgrade: replace one-hot categorical encodings with
physicochemical PROPERTY VECTORS so the model can interpolate (and extrapolate)
between chemically similar entities instead of treating every category as an
orthogonal, equidistant label.

WHY
---
A one-hot code makes "arginine" and "lysine" exactly as dissimilar as
"arginine" and "sucrose" — the encoding throws away all chemical structure, so
the model must memorise each category independently and cannot transfer between
them or generalise to an unseen one. Replacing the label with the *properties
that make the chemical behave the way it does* (charge, size, Hofmeister
position, hydrophobicity, ...) puts every category at a point in a continuous
physical space where distance means similarity. Similar chemicals then sit close
together and the model interpolates for free; a brand-new excipient/salt/buffer
lands at its property coordinates and inherits behaviour from its neighbours.

SCOPE
-----
This upgrades the CHEMICALLY-meaningful categoricals only:
    Buffer_type, Salt_type, Stabilizer_type, Surfactant_type, Excipient_type,
    Protein_class_type
It deliberately does NOT touch Protein_type — that is the held-out extrapolation
target and needs Rung-3 (sequence/structure descriptors), a separate experiment.
Protein_type stays one-hot for now so this change is isolated and evaluable.

DESIGN NOTES
------------
* Every property table includes a "none" / unknown row of physical zeros, so an
  absent ingredient maps to the origin of its property space (no contribution),
  which is the physically correct null.
* Unrecognised categories fall back to the "none"/zero row and emit a warning,
  rather than crashing — so new chemicals are handled gracefully (and you can
  then add their real properties to the table).
* Values are grounded in standard physical chemistry. They are intended to be
  *approximately* right and, crucially, *relationally* right (arginine closer to
  lysine than to proline; tween-80 lower CMC than tween-20; trehalose ~ sucrose).
  Exact magnitudes matter less than correct ordering because everything is
  StandardScaler-normalised downstream.

USAGE
-----
    from categorical_features import (
        CHEM_CATEGORICALS, featurize_chemical_categoricals,
    )
    # df already lower-cased its categorical columns
    df, prop_cols = featurize_chemical_categoricals(df)
    # `prop_cols` are new numeric columns to add to your numeric pipeline.
    # Remove CHEM_CATEGORICALS from the one-hot list; keep Protein_type one-hot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The six chemically-meaningful categoricals this module replaces.
# (Protein_type is intentionally excluded — see module docstring.)
CHEM_CATEGORICALS = [
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
    "Protein_class_type",
]


# ---------------------------------------------------------------------------
# Property tables. Each maps a lowercased category -> {descriptor: value}.
# Every table MUST contain a "none" row of physical zeros (the null ingredient).
# Descriptor sets differ per categorical because the relevant physics differs.
# ---------------------------------------------------------------------------

# Reference MW each table's ratio is expressed against (the most common /
# canonical member of that category). Raw Da values span two orders of
# magnitude across categories (60 - 8400), which is exactly the kind of
# unbounded, StandardScaler-dependent magnitude that blew up under a
# leave-one-ingredient-out fold (see SALT_PROPS below and the P0 salt
# regression this fixes): when the held-out category is rare/absent from a
# training fold, that fold's fitted scale for the raw-MW column can be near
# zero, so the held-out row's raw Da value passes through the scaler almost
# unwhitened. Expressing MW as a same-order-of-magnitude ratio bounds the raw
# feature value itself at construction time, before it ever reaches the
# scaler -- degenerate fold statistics can no longer turn it into an outsized
# activation.
BUFFER_MW_REF: float = 155.2  # histidine
STABILIZER_MW_REF: float = 342.3  # sucrose / trehalose
SURFACTANT_MW_REF: float = 1228.0  # tween-20 / polysorbate-20
EXCIPIENT_MW_REF: float = 174.2  # arginine

# Buffers: what matters is the pH region they buffer (pKa), their charge tendency
# at formulation pH, size, and whether they specifically interact with protein
# surfaces (histidine's imidazole can coordinate; phosphate/acetate are inert-ish).
BUFFER_PROPS: dict[str, dict[str, float]] = {
    "histidine": {
        "buf_pKa": 6.0,
        "buf_mw_ratio": 155.2 / BUFFER_MW_REF,
        "buf_charge_sign": 1.0,
        "buf_specific_interact": 1.0,
    },
    "pbs": {
        "buf_pKa": 7.2,
        "buf_mw_ratio": 141.96 / BUFFER_MW_REF,
        "buf_charge_sign": -1.0,
        "buf_specific_interact": 0.0,
    },
    "phosphate": {
        "buf_pKa": 7.2,
        "buf_mw_ratio": 141.96 / BUFFER_MW_REF,
        "buf_charge_sign": -1.0,
        "buf_specific_interact": 0.0,
    },
    "acetate": {
        "buf_pKa": 4.76,
        "buf_mw_ratio": 60.05 / BUFFER_MW_REF,
        "buf_charge_sign": -1.0,
        "buf_specific_interact": 0.0,
    },
    "citrate": {
        "buf_pKa": 5.4,
        "buf_mw_ratio": 192.12 / BUFFER_MW_REF,
        "buf_charge_sign": -1.0,
        "buf_specific_interact": 0.0,
    },
    "none": {"buf_pKa": 0.0, "buf_mw_ratio": 0.0, "buf_charge_sign": 0.0, "buf_specific_interact": 0.0},
}

# Salts: the dominant axis for protein interactions is the Hofmeister series
# (kosmotrope/charge-screening character). We encode a relative Hofmeister
# position (negative = kosmotropic/stabilising, positive = chaotropic) and the
# valence. MW is deliberately NOT encoded here (unlike the other four
# categoricals): valence + Hofmeister position already carry the salt physics
# that drives protein-protein interaction, and raw salt_mw was the P0
# regression -- a leave-one-salt-out fold (e.g. holding out nacl, the
# dominant salt in this dataset) left salt_mw near-zero-variance in training,
# so the held-out row's raw Da value blew through the fold's degenerate
# scaler as an outsized activation. Dropping it removes that failure mode
# entirely instead of merely shrinking it.
SALT_PROPS: dict[str, dict[str, float]] = {
    "nacl": {"salt_hofmeister": 0.0, "salt_valence": 1.0},
    "kcl": {"salt_hofmeister": 0.2, "salt_valence": 1.0},
    "nacitrate": {"salt_hofmeister": -1.0, "salt_valence": 3.0},
    "ammonium_sulfate": {"salt_hofmeister": -1.5, "salt_valence": 2.0},
    "none": {"salt_hofmeister": 0.0, "salt_valence": 0.0},
}

# Excipients (viscosity-modifying amino acids / osmolytes): net charge at ~pH6,
# size, hydrophobicity (logP), H-bond donor count, and a flag for the known
# viscosity-reduction class. Arginine and lysine are both +1 and large -> close;
# proline is the neutral osmolyte -> separated on charge.
EXCIPIENT_PROPS: dict[str, dict[str, float]] = {
    "arginine": {
        "exc_charge": 1.0,
        "exc_mw_ratio": 174.2 / EXCIPIENT_MW_REF,
        "exc_logP": -4.2,
        "exc_hdonor": 4.0,
        "exc_visc_reducer": 1.0,
    },
    "lysine": {
        "exc_charge": 1.0,
        "exc_mw_ratio": 146.19 / EXCIPIENT_MW_REF,
        "exc_logP": -3.05,
        "exc_hdonor": 3.0,
        "exc_visc_reducer": 1.0,
    },
    "proline": {
        "exc_charge": 0.0,
        "exc_mw_ratio": 115.13 / EXCIPIENT_MW_REF,
        "exc_logP": -2.5,
        "exc_hdonor": 1.0,
        "exc_visc_reducer": 1.0,
    },
    "glycine": {
        "exc_charge": 0.0,
        "exc_mw_ratio": 75.07 / EXCIPIENT_MW_REF,
        "exc_logP": -3.21,
        "exc_hdonor": 1.0,
        "exc_visc_reducer": 0.0,
    },
    "histidine": {
        "exc_charge": 0.1,
        "exc_mw_ratio": 155.15 / EXCIPIENT_MW_REF,
        "exc_logP": -3.32,
        "exc_hdonor": 2.0,
        "exc_visc_reducer": 1.0,
    },
    "none": {
        "exc_charge": 0.0,
        "exc_mw_ratio": 0.0,
        "exc_logP": 0.0,
        "exc_hdonor": 0.0,
        "exc_visc_reducer": 0.0,
    },
}

# Stabilizers (sugars / polyols): MW, hydroxyl count (H-bonding / preferential
# exclusion strength), and a preferential-exclusion flag. Sucrose and trehalose
# are near-identical disaccharides -> they should sit on top of each other.
STABILIZER_PROPS: dict[str, dict[str, float]] = {
    "sucrose": {"stab_mw_ratio": 342.3 / STABILIZER_MW_REF, "stab_oh": 8.0, "stab_pref_excl": 1.0},
    "trehalose": {"stab_mw_ratio": 342.3 / STABILIZER_MW_REF, "stab_oh": 8.0, "stab_pref_excl": 1.0},
    "sorbitol": {"stab_mw_ratio": 182.17 / STABILIZER_MW_REF, "stab_oh": 6.0, "stab_pref_excl": 1.0},
    "mannitol": {"stab_mw_ratio": 182.17 / STABILIZER_MW_REF, "stab_oh": 6.0, "stab_pref_excl": 1.0},
    "none": {"stab_mw_ratio": 0.0, "stab_oh": 0.0, "stab_pref_excl": 0.0},
}

# Surfactants: HLB (hydrophilic-lipophilic balance), MW, and CMC. Tween-20 vs
# Tween-80 differ mainly in CMC and HLB (tail length) -> close but distinguishable.
SURFACTANT_PROPS: dict[str, dict[str, float]] = {
    "tween-20": {"surf_hlb": 16.7, "surf_mw_ratio": 1228.0 / SURFACTANT_MW_REF, "surf_cmc": 0.06},
    "tween-80": {"surf_hlb": 15.0, "surf_mw_ratio": 1310.0 / SURFACTANT_MW_REF, "surf_cmc": 0.012},
    "polysorbate-20": {"surf_hlb": 16.7, "surf_mw_ratio": 1228.0 / SURFACTANT_MW_REF, "surf_cmc": 0.06},
    "polysorbate-80": {"surf_hlb": 15.0, "surf_mw_ratio": 1310.0 / SURFACTANT_MW_REF, "surf_cmc": 0.012},
    "poloxamer-188": {"surf_hlb": 29.0, "surf_mw_ratio": 8400.0 / SURFACTANT_MW_REF, "surf_cmc": 0.05},
    "none": {"surf_hlb": 0.0, "surf_mw_ratio": 0.0, "surf_cmc": 0.0},
}

# Protein class (format): structural descriptors of the molecular architecture —
# domain count, relative hinge flexibility, typical pI, glycosylation. These give
# the model a continuous notion of "what kind of molecule" rather than an opaque
# class label, so e.g. an unseen format lands near structurally similar ones.
PROTEIN_CLASS_PROPS: dict[str, dict[str, float]] = {
    "mab_igg1": {"pc_domains": 12.0, "pc_flex": 1.0, "pc_typ_pi": 8.5, "pc_glyc": 1.0},
    "mab_igg4": {"pc_domains": 12.0, "pc_flex": 1.2, "pc_typ_pi": 7.0, "pc_glyc": 1.0},
    "fc-fusion": {"pc_domains": 10.0, "pc_flex": 1.5, "pc_typ_pi": 6.0, "pc_glyc": 1.0},
    "fc_fusion": {"pc_domains": 10.0, "pc_flex": 1.5, "pc_typ_pi": 6.0, "pc_glyc": 1.0},
    "bispecific": {"pc_domains": 12.0, "pc_flex": 1.3, "pc_typ_pi": 7.5, "pc_glyc": 1.0},
    "trispecific": {"pc_domains": 14.0, "pc_flex": 1.4, "pc_typ_pi": 7.5, "pc_glyc": 1.0},
    "adc": {"pc_domains": 12.0, "pc_flex": 1.0, "pc_typ_pi": 8.0, "pc_glyc": 1.0},
    "polyclonal": {"pc_domains": 12.0, "pc_flex": 1.0, "pc_typ_pi": 7.5, "pc_glyc": 1.0},
    "other": {"pc_domains": 6.0, "pc_flex": 1.0, "pc_typ_pi": 5.5, "pc_glyc": 0.0},
    "none": {"pc_domains": 0.0, "pc_flex": 0.0, "pc_typ_pi": 0.0, "pc_glyc": 0.0},
}


# Registry: column name -> (property table, ordered descriptor keys).
# The ordered keys define the output column order and are used to build the
# zero-fallback row.
def _keys(table: dict[str, dict[str, float]]) -> list[str]:
    # Use the "none" row to define the canonical descriptor order.
    return list(table["none"].keys())


_TABLES: dict[str, dict[str, dict[str, float]]] = {
    "Buffer_type": BUFFER_PROPS,
    "Salt_type": SALT_PROPS,
    "Stabilizer_type": STABILIZER_PROPS,
    "Surfactant_type": SURFACTANT_PROPS,
    "Excipient_type": EXCIPIENT_PROPS,
    "Protein_class_type": PROTEIN_CLASS_PROPS,
}


def _normalize_category(raw: str) -> str:
    """Lowercase, strip, and map empty/nan-likes to 'none'."""
    s = str(raw).strip().lower()
    if s in ("", "nan", "unknown", "na", "n/a"):
        return "none"
    return s


def _lookup(
    table: dict[str, dict[str, float]], keys: list[str], raw: str, warned: set
) -> list[float]:
    """Map one category value to its ordered descriptor vector.

    Substring match is allowed (e.g. 'l-arginine hcl' -> 'arginine') so minor
    naming variants resolve. Falls back to the 'none' zero-row on miss.
    """
    cat = _normalize_category(raw)
    if cat in table:
        row = table[cat]
    else:
        # Try a substring match against known keys (handles 'l-arginine', etc.).
        match = next((k for k in table if k != "none" and k in cat), None)
        if match is not None:
            row = table[match]
        else:
            if cat not in warned:
                print(
                    f"  [categorical_features] WARNING: unknown category "
                    f"'{cat}' -> using zero (none) properties. Add it to the "
                    f"property table for proper handling."
                )
                warned.add(cat)
            row = table["none"]
    return [float(row[k]) for k in keys]


def featurize_chemical_categoricals(
    df: pd.DataFrame,
    drop_original: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Append physicochemical property columns for each chemical categorical.

    Parameters
    ----------
    df : DataFrame with the categorical columns present (any case; missing
         columns are treated as all-'none').
    drop_original : if True, drop the original categorical columns after
         featurizing. Default False — harmless to leave them (they simply
         won't be referenced by the numeric/one-hot pipeline once removed from
         the encoder lists).

    Returns
    -------
    (df_out, prop_cols)
        df_out   : df with new numeric property columns appended.
        prop_cols: ordered list of the new column names (add these to your
                   numeric pipeline / StandardScaler).
    """
    df = df.copy()
    prop_cols: list[str] = []
    warned: set = set()

    for col, table in _TABLES.items():
        keys = _keys(table)
        prop_cols.extend(keys)
        if col not in df.columns:
            # Column absent: fill all descriptors with the 'none' zero-row.
            zero = {k: float(table["none"][k]) for k in keys}
            for k in keys:
                df[k] = zero[k]
            continue
        # Build the property matrix row-by-row.
        mat = np.array(
            [_lookup(table, keys, v, warned) for v in df[col].values],
            dtype=float,
        )
        for j, k in enumerate(keys):
            df[k] = mat[:, j]
        if drop_original:
            df.drop(columns=[col], inplace=True)

    return df, prop_cols


def describe_property_space() -> pd.DataFrame:
    """Return a tidy table of every category and its descriptors (for auditing)."""
    rows = []
    for col, table in _TABLES.items():
        keys = _keys(table)
        for cat, props in table.items():
            rows.append({"categorical": col, "category": cat, **{k: props[k] for k in keys}})
    return pd.DataFrame(rows)
