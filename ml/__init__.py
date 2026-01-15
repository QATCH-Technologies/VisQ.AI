"""
Configuration module for the Viscosity model library.
Contains feature lists, target definitions, and physics priors.
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
# Units assumed to match input data (e.g. mM for salts, M for sugars, % for surfactants)
CONC_THRESHOLDS = {
    # Salts / Excipients (mM)
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    # Surfactants (%)
    "tween": 0.01,
    "polysorbate": 0.01,
    # Stabilizers (M - assuming 0.2M inputs)
    "sucrose": 200.0,  # Assuming inputs are mM. If M, change to 0.2
    "trehalose": 200.0,  # Assuming inputs are mM. If M, change to 0.2
}

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
        "tween": -1,
    },
    ("mab_igg1", "mixed"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween": -1,
    },
    ("mab_igg1", "far"): {
        "arginine": 0,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween": -1,
    },
    # --- IgG4 ---
    ("mab_igg4", "near"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween": -1,
    },
    ("mab_igg4", "mixed"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween": -1,
    },
    ("mab_igg4", "far"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween": -1,
    },
    # --- Fc-Fusion ---
    ("fc-fusion", "near"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": -1,
        "stabilizer": 1,
        "tween": -2,
    },
    ("fc-fusion", "mixed"): {
        "arginine": -1,
        "lysine": 0,
        "nacl": 0,
        "proline": -2,
        "stabilizer": 1,
        "tween": -2,
    },
    ("fc-fusion", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -2,
        "stabilizer": 1,
        "tween": -2,
    },
    # --- Bispecific ---
    ("bispecific", "near"): {
        "arginine": -2,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween": -1,
    },
    ("bispecific", "mixed"): {
        "arginine": -1,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween": -2,
    },
    ("bispecific", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween": -2,
    },
    # --- Other/Fallback ---
    ("other", "near"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween": 0,
    },
    ("other", "mixed"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween": 0,
    },
    ("other", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween": 0,
    },
    # POLYCLONAL
    ("polyclonal", "near"): {
        "arginine": -1,
        "lysine": -1,
        "nacl": -1,
        "proline": 0,
        "stabilizer": 1,
        "tween": 0,
    },
    ("polyclonal", "mixed"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween": 0,
    },
    ("polyclonal", "far"): {
        "arginine": 0,
        "lysine": 0,
        "nacl": 0,
        "proline": -1,
        "stabilizer": 1,
        "tween": 0,
    },
}

# Map specific column names to the generic types in the Prior Table
EXCIPIENT_TYPE_MAPPING = {
    "Salt_type": ["nacl"],
    "Excipient_type": ["arginine", "lysine", "proline"],
    "Stabilizer_type": ["sucrose", "trehalose"],
    "Surfactant_type": ["tween", "polysorbate"],
}

# Map conc columns to type columns for splitting logic
CONC_TYPE_PAIRS = {
    "Salt_conc": "Salt_type",
    "Stabilizer_conc": "Stabilizer_type",
    "Surfactant_conc": "Surfactant_type",
    "Excipient_conc": "Excipient_type",
}
