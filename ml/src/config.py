"""Configuration module for the VisQ.AI.

This module contains the static configuration definitions required for feature engineering,
target selection, and physics-informed constraints within the viscosity prediction pipeline.

Attributes:
    BASE_CATEGORICAL (List[str]): A list of column names representing categorical features,
        including protein types and formulation component identities.
    BASE_NUMERIC (List[str]): A list of column names representing continuous features,
        such as molecular weight, concentrations, pH, and hydrophobicity indices.
    TARGETS (List[str]): A list of target variables representing viscosity measurements
        at various shear rates or conditions.
    CONC_THRESHOLDS (Dict[str, float]): A dictionary mapping formulation components to
        concentration thresholds (in mM, M, or %). Used for normalization or feature splitting.
    EXCIPIENT_PRIORS (Dict[Tuple[str, str], Dict[str, float]]): A nested dictionary encoding
        domain knowledge (physics priors) regarding excipient effects.
        Key structure: `(Protein_Class, Electrostatic_Regime)`
        Value structure: `{Excipient_Name: Effect_Coefficient}`
    EXCIPIENT_TYPE_MAPPING (Dict[str, List[str]]): Maps the generic dataframe column names
        to the specific chemical keys used in `EXCIPIENT_PRIORS`.
    CONC_TYPE_PAIRS (Dict[str, str]): Maps numeric concentration columns to their
        corresponding categorical type columns.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

# --- CONFIGURATION ---

BASE_CATEGORICAL = [
    "Protein_type",
    "Protein_class_type",
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]

BASE_NUMERIC = [
    "MW",
    "Protein_conc",
    "Temperature",
    "Buffer_pH",
    "Buffer_conc",
    "Salt_conc",
    "Stabilizer_conc",
    "Surfactant_conc",
    "Excipient_conc",
    "kP",
    "C_Class",
    "HCI",
]

TARGETS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]

# --- CONCENTRATION THRESHOLDS ---

CONC_THRESHOLDS = {
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    "tween": 0.01,
    "polysorbate": 0.01,
    "sucrose": 200.0,
    "trehalose": 200.0,
}
"""Dict[str, float]: Thresholds defining significant concentration levels.
Units are assumed to match input data (e.g. mM for salts, M for sugars, % for surfactants).
"""

# --- PHYSICS PRIORS ---

EXCIPIENT_PRIORS = {
    # Format: (Protein_Class, Regime): {Excipient_Name: Effect_Value}
    # --- SPECIAL REGIME: NO PROTEIN ---
    ("noprotein", "noprotein"): {
        "nacl": 0.1,
        "arginine": 0.1,
        "lysine": 0.1,
        "proline": 0.1,
        "stabilizer": 1.0,  # Generalized from sucrose/trehalose
        "tween": 0.1,
    },
    # --- IgG1 ---
    ("mab_igg1", "near"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    ("mab_igg1", "mixed"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    ("mab_igg1", "far"): {
        "arginine": 0,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    # --- IgG4 ---
    ("mab_igg4", "near"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    ("mab_igg4", "mixed"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    ("mab_igg4", "far"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    # --- Fc-Fusion ---
    ("fc-fusion", "near"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -2,
        "tween80": -2,
    },
    ("fc-fusion", "mixed"): {
        "arginine": -1,
        "lysine": 0,
        "nacl": 0,
        "proline": -2,
        "stabilizer": 1,
        "tween20": -2,
        "tween80": -2,
    },
    ("fc-fusion", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -2,
        "stabilizer": 1,
        "tween20": -2,
        "tween80": -2,
    },
    # --- Bispecific ---
    ("bispecific", "near"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween20": -1,
        "tween80": -1,
    },
    ("bispecific", "mixed"): {
        "arginine": -1,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -2,
        "tween80": -2,
    },
    ("bispecific", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween20": -2,
        "tween80": -2,
    },
    # --- Other/Fallback ---
    ("other", "near"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween20": 0,
        "tween80": 0,
    },
    ("other", "mixed"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween20": 0,
        "tween80": 0,
    },
    ("other", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween20": 0,
        "tween80": 0,
    },
    # POLYCLONAL
    ("polyclonal", "near"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween20": 0,
        "tween80": 0,
    },
    ("polyclonal", "mixed"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween20": 0,
        "tween80": 0,
    },
    ("polyclonal", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween20": 0,
        "tween80": 0,
    },
}
"""Dict[Tuple[str, str], Dict[str, float]]: Heuristic prior values for excipient effects.

    Encodes the expected impact of specific excipients on viscosity based on the 
    protein class and the electrostatic interaction regime (e.g., 'near', 'mixed', 'far').
    Negative values generally imply viscosity reduction, while positive values imply increases.
"""

EXCIPIENT_TYPE_MAPPING = {
    "Salt_type": ["nacl"],
    "Excipient_type": ["arginine", "lysine", "proline"],
    "Stabilizer_type": ["sucrose", "trehalose"],
    "Surfactant_type": ["tween", "polysorbate"],
}
"""Dict[str, List[str]]: Mapping of DataFrame column names to specific chemical keys.
Used to link the generic type columns (e.g., 'Salt_type') to the specific keys 
defined in `EXCIPIENT_PRIORS` (e.g., 'nacl').
"""

CONC_TYPE_PAIRS = {
    "Salt_conc": "Salt_type",
    "Stabilizer_conc": "Stabilizer_type",
    "Surfactant_conc": "Surfactant_type",
    "Excipient_conc": "Excipient_type",
}
"""Dict[str, str]: Mapping of numeric concentration columns to categorical type columns.
Used for logic that requires splitting or pairing concentration values with 
their corresponding chemical identity.
"""
