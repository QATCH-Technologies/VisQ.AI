"""
logo_splits.py
==============
Leave-one-GROUP-out split generators for the three held-out axes Phase 0
requires: protein, ingredient, and protein-class. parity_eval.py only ever
held out Ibalizumab; none of these axes were previously measured at all.

Each axis produces a list of `LogoGroup` descriptors and a `split()` method
that partitions a prepared DataFrame into (train_df, held_df). "Held out"
always means "every row that touches this group" -- for the ingredient axis
that's every row where the given categorical column equals the category,
regardless of which protein or other ingredients are present.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# The five chemically-meaningful ingredient columns to hold out one category
# at a time. Protein_class_type is deliberately excluded here -- it gets its
# own axis (protein_class), not folded into "ingredient".
INGREDIENT_COLS = [
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]

# Category values that mean "this ingredient is absent" -- never a valid
# held-out group (holding out "none" would strip the ingredient-free rows
# from every fold, not exercise extrapolation to an unseen ingredient).
_NULL_CATEGORIES = {"none", "unknown", "nan", "na", "n/a", ""}


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


@dataclass(frozen=True)
class LogoGroup:
    axis: str  # "protein" | "ingredient" | "protein_class"
    key: str  # human-readable group id, e.g. "ibalizumab" or "Buffer_type=histidine"
    column: str  # the DataFrame column this group is defined on
    value: str  # the (normalized) category value identifying this group

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
    """One group per Protein_type, excluding the null/buffer-only rows."""
    counts = _norm(df["Protein_type"]).value_counts()
    groups = []
    for val, n in counts.items():
        if val in _NULL_CATEGORIES or n < min_rows:
            continue
        groups.append(LogoGroup(axis="protein", key=val, column="Protein_type", value=val))
    return sorted(groups, key=lambda g: g.key)


def protein_class_groups(df: pd.DataFrame, min_rows: int = 2) -> list[LogoGroup]:
    """One group per Protein_class_type (e.g. drop all fc-fusion)."""
    counts = _norm(df["Protein_class_type"]).value_counts()
    groups = []
    for val, n in counts.items():
        if val in _NULL_CATEGORIES or n < min_rows:
            continue
        groups.append(LogoGroup(axis="protein_class", key=val, column="Protein_class_type", value=val))
    return sorted(groups, key=lambda g: g.key)


def ingredient_groups(df: pd.DataFrame, min_rows: int = 2) -> list[LogoGroup]:
    """One group per (ingredient column, category) pair actually present in
    the data, e.g. ('Buffer_type', 'histidine'), ('Excipient_type', 'arginine')."""
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
    """Ablation counterfactual for the ingredient axis: rewrite the held-out
    rows' ingredient column to 'none' BEFORE feature engineering, so
    featurize_chemical_categoricals maps them to the zero/null property row
    instead of the real physicochemical properties. Used to test whether the
    property vector is actually buying extrapolation (P0: leave-one-
    ingredient-out ablation) -- if predicting with the real properties beats
    predicting with them zeroed out, the property vector is doing work.
    """
    if group.axis != "ingredient":
        raise ValueError("zero_ingredient_properties only applies to the ingredient axis")
    out = df.copy()
    out[group.column] = "none"
    return out
