"""Provide physicochemical property-vector representations for chemical categories.

This module replaces one-hot encodings for chemically meaningful categorical
features with continuous physicochemical property vectors. The representation
places chemically similar entities near one another in feature space, allowing
downstream models to interpolate between related categories and assign
reasonable representations to previously unseen categories when their
properties are added to the corresponding property table.

The transformed categorical features include buffer, salt, stabilizer,
surfactant, excipient, and protein-class categories. `Protein_type` is
intentionally excluded because it is treated as a held-out extrapolation
target for a separate representation experiment.

Each property table contains a `"none"` entry representing the absence of
the corresponding ingredient or an unknown value. Unknown categories also
fall back to this zero vector and emit a warning. Property descriptors are
defined independently for each categorical because the physically relevant
properties differ between chemical classes.

All generated property columns are numeric and are intended to be passed
through the downstream numeric preprocessing pipeline, including
standardization.

Attributes:
    CHEM_CATEGORICALS: Names of the categorical columns represented by
        physicochemical property vectors.
    BUFFER_PROPS: Physicochemical descriptors for buffer categories.
    SALT_PROPS: Physicochemical descriptors for salt categories.
    EXCIPIENT_PROPS: Physicochemical descriptors for excipient categories.
    STABILIZER_PROPS: Physicochemical descriptors for stabilizer categories.
    SURFACTANT_PROPS: Physicochemical descriptors for surfactant categories.
    PROTEIN_CLASS_PROPS: Structural descriptors for protein-class categories.

Examples:
    Featurize the chemical categorical columns and obtain the generated
    numeric feature names::

        df, prop_cols = featurize_chemical_categoricals(df)

    The returned `prop_cols` can then be incorporated into the numeric
    feature pipeline while the original chemical categorical columns are
    removed from the one-hot feature list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.validation import require_dataframe

CHEM_CATEGORICALS = [
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
    "Protein_class_type",
]


# ---------------------------------------------------------------------------
# Property tables. Each maps a lowercased category to descriptor values.
# Every table must include a `none` row representing the physical null
# ingredient. Descriptor sets vary by categorical to reflect the relevant
# physicochemical properties.
# ---------------------------------------------------------------------------

# Reference MW used to normalize each table's molecular-weight descriptor.
# Expressing MW as a ratio rather than raw Da keeps feature magnitudes bounded
# across categories and prevents poorly estimated scaler statistics in
# leave-one-ingredient-out folds from producing outsized activations.
BUFFER_MW_REF: float = 155.2  # histidine
STABILIZER_MW_REF: float = 342.3  # sucrose / trehalose
SURFACTANT_MW_REF: float = 1228.0  # tween-20 / polysorbate-20
EXCIPIENT_MW_REF: float = 174.2  # arginine

# Buffers: descriptors capture the properties most relevant to formulation
# behavior, including buffering range (pKa), charge tendency, molecular size,
# and potential protein-surface interactions.
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
    "none": {
        "buf_pKa": 0.0,
        "buf_mw_ratio": 0.0,
        "buf_charge_sign": 0.0,
        "buf_specific_interact": 0.0,
    },
}

# Salts: descriptors focus on Hofmeister behavior and valence, which capture
# the primary salt-dependent effects on protein interactions. MW is omitted to
# avoid unstable scaling in leave-one-salt-out folds, where a held-out salt
# can produce an unsupported activation from a near-zero-variance MW feature.
SALT_PROPS: dict[str, dict[str, float]] = {
    "nacl": {"salt_hofmeister": 0.0, "salt_valence": 1.0},
    "kcl": {"salt_hofmeister": 0.2, "salt_valence": 1.0},
    "nacitrate": {"salt_hofmeister": -1.0, "salt_valence": 3.0},
    "ammonium_sulfate": {"salt_hofmeister": -1.5, "salt_valence": 2.0},
    "none": {"salt_hofmeister": 0.0, "salt_valence": 0.0},
}

# Excipients: descriptors capture properties relevant to viscosity and protein
# interactions, including charge, molecular size, hydrophobicity, hydrogen
# bonding, and whether the excipient belongs to a known viscosity-reduction
# class.
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

# Stabilizers: descriptors capture molecular size, hydroxyl density, and
# preferential-exclusion behavior relevant to protein stabilization. Similar
# sugars and polyols therefore receive similar physicochemical representations.
STABILIZER_PROPS: dict[str, dict[str, float]] = {
    "sucrose": {"stab_mw_ratio": 342.3 / STABILIZER_MW_REF, "stab_oh": 8.0, "stab_pref_excl": 1.0},
    "trehalose": {
        "stab_mw_ratio": 342.3 / STABILIZER_MW_REF,
        "stab_oh": 8.0,
        "stab_pref_excl": 1.0,
    },
    "sorbitol": {
        "stab_mw_ratio": 182.17 / STABILIZER_MW_REF,
        "stab_oh": 6.0,
        "stab_pref_excl": 1.0,
    },
    "mannitol": {
        "stab_mw_ratio": 182.17 / STABILIZER_MW_REF,
        "stab_oh": 6.0,
        "stab_pref_excl": 1.0,
    },
    "none": {"stab_mw_ratio": 0.0, "stab_oh": 0.0, "stab_pref_excl": 0.0},
}

# Surfactants: descriptors capture hydrophilic-lipophilic balance, molecular
# size, and critical micelle concentration, allowing closely related
# surfactants to remain similar while preserving meaningful physicochemical
# differences.
SURFACTANT_PROPS: dict[str, dict[str, float]] = {
    "tween-20": {"surf_hlb": 16.7, "surf_mw_ratio": 1228.0 / SURFACTANT_MW_REF, "surf_cmc": 0.06},
    "tween-80": {"surf_hlb": 15.0, "surf_mw_ratio": 1310.0 / SURFACTANT_MW_REF, "surf_cmc": 0.012},
    "polysorbate-20": {
        "surf_hlb": 16.7,
        "surf_mw_ratio": 1228.0 / SURFACTANT_MW_REF,
        "surf_cmc": 0.06,
    },
    "polysorbate-80": {
        "surf_hlb": 15.0,
        "surf_mw_ratio": 1310.0 / SURFACTANT_MW_REF,
        "surf_cmc": 0.012,
    },
    "poloxamer-188": {
        "surf_hlb": 29.0,
        "surf_mw_ratio": 8400.0 / SURFACTANT_MW_REF,
        "surf_cmc": 0.05,
    },
    "none": {"surf_hlb": 0.0, "surf_mw_ratio": 0.0, "surf_cmc": 0.0},
}

# Protein class: descriptors capture molecular architecture, including domain
# count, hinge flexibility, typical pI, and glycosylation, providing a
# continuous representation of structural similarity rather than an opaque
# categorical label.
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


def _keys(table: dict[str, dict[str, float]]) -> list[str]:
    """Return the canonical descriptor order for a property table.

    The descriptor names are taken from the `"none"` row so that the null
    representation defines a stable ordering for every category in the
    corresponding table.

    Args:
        table: Property table mapping category names to descriptor dictionaries.
            The table is expected to contain a `"none"` entry.

    Returns:
        list[str]: Descriptor names in the canonical output order.
    """
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
    """Normalize a categorical value for property-table lookup.

    Values are converted to strings, stripped of surrounding whitespace, and
    lowercased. Empty values and common representations of missing or unknown
    values are mapped to `"none"`.

    Args:
        raw: Raw categorical value to normalize.

    Returns:
        str: Normalized category name, or `"none"` when the value represents
        a missing or unknown category.
    """
    s = str(raw).strip().lower()
    if s in ("", "nan", "unknown", "na", "n/a"):
        return "none"
    return s


def _lookup(
    table: dict[str, dict[str, float]], keys: list[str], raw: str, warned: set
) -> list[float]:
    """Resolve a category to its ordered physicochemical property vector.

    Exact category matches are preferred. If an exact match is not found, a
    substring match against known categories is attempted to accommodate minor
    naming variations. For example, a category containing `"arginine"` can
    resolve to the `"arginine"` property entry.

    Categories that cannot be resolved fall back to the table's `"none"`
    row, which represents a zero-valued physical contribution. Each unknown
    normalized category is reported only once through the supplied warning
    set.

    Args:
        table: Property table mapping category names to descriptor dictionaries.
        keys: Ordered descriptor names defining the output vector layout.
        raw: Raw categorical value to resolve.
        warned: Mutable set of category names for which an unknown-category
            warning has already been emitted.

    Returns:
        list[float]: Property values corresponding to `keys` in order.
    """
    cat = _normalize_category(raw)
    if cat in table:
        row = table[cat]
    else:
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
    """Append physicochemical property vectors for chemical categoricals.

    Each configured chemical categorical column is converted into the
    corresponding property representation defined in its property table.
    Generated descriptor columns are appended to a copy of the input
    DataFrame and their names are returned in deterministic order.

    If a configured categorical column is absent from the input DataFrame,
    its generated descriptors are filled with the corresponding `"none"`
    values. If a present column contains an unrecognized category, that value
    is mapped to the `"none"` property vector and a warning is emitted.

    The original categorical columns are retained by default so callers can
    explicitly control which columns are passed to subsequent categorical
    encoders. They can instead be removed by setting `drop_original=True`.

    Args:
        df: Input DataFrame containing the chemical categorical columns. Column
            values may use varying capitalization or whitespace because values
            are normalized before lookup. Missing configured columns are
            treated as containing only the `"none"` category.
        drop_original: Whether to remove each original chemical categorical
            column after its property descriptors have been generated. Defaults
            to `False`.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:

            - The copied DataFrame with physicochemical descriptor columns
              appended.
            - An ordered list of generated descriptor column names suitable
              for inclusion in a numeric preprocessing pipeline.

    Raises:
        ValueError: If `df` does not satisfy the validation requirements
            enforced by :func:`visqai.validation.require_dataframe`.
    """
    require_dataframe(df, "df")
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
    """Return all configured categories and their physicochemical descriptors.

    The result is a tidy DataFrame containing one row per category and columns
    for the categorical feature, category name, and every descriptor defined
    for that feature. This representation is useful for auditing, inspecting,
    or documenting the property space used by the featurization pipeline.

    Returns:
        pd.DataFrame: Tidy property-space table with `categorical` and
        `category` columns followed by the descriptors applicable to each
        categorical feature.
    """
    rows = []
    for col, table in _TABLES.items():
        keys = _keys(table)
        for cat, props in table.items():
            rows.append({"categorical": col, "category": cat, **{k: props[k] for k in keys}})
    return pd.DataFrame(rows)
