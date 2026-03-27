"""
constants.py
============
Global configuration for the CBM-CNP training pipeline.

Contains:
  - Protein class taxonomy (PROTEIN_CLASS_MAP, NON_PROTEIN_GROUPS)
  - Concept Bottleneck definitions (CONCEPT_DEFS and derived constants)

These are the only values that need to change when adding new proteins,
new concept dimensions, or adjusting physics-inspired priors.
"""

# ============================================================
# Protein class taxonomy
# ============================================================

PROTEIN_CLASS_MAP: dict[str, str] = {
    "adalimumab": "igg1",
    "bevacizumab": "igg1",
    "trastuzumab": "igg1",
    "pembrolizumab": "igg4",
    "ibalizumab": "igg4",
    "nivolumab": "igg4",
    "belatacept": "fc_fusion",
    "etanercept": "fc_fusion",
    "vudalimab": "bispecific",
    "poly-higg": "polyclonal",
    "bgg": "polyclonal",
    "bsa": "other",
}

# Groups that should be excluded from protein-level contrastive losses
NON_PROTEIN_GROUPS: set[str] = {"none"}


CONCEPT_DEFS = [
    # tanh used for polarity range between -1 to +1 to simulate repulsion and attraction.
    # sigmoid used for intensities to between 0 to +1.
    # +1 and -1 signal directional and inversly proportional interactions.
    # ------------------------------------------------------------------------- #
    # PROTEIN-PROTEIN CONCEPTS
    # kP measures protein-protein interaction. Multiplying it by concentration
    # (conc) gives the total interaction potential in the vial.
    # We use tanh because interactions can be attractive (-) or repulsive (+).
    # We use -1 so that the concept activates strongly for sticky, attractive proteins.
    ("self_interaction", "conc_x_kP", -1, "tanh"),
    # HCI (Hydrophobic Contact Index) estimates exposed sticky patches.
    # Multiplying by concentration scales it to the whole solution.
    # It is a sigmoid because you either have hydrophobic patches or you don't;
    # you can't have "anti-hydrophobicity."
    ("hydrophobicity", "conc_x_HCI", +1, "sigmoid"),
    # A protein's net charge interacts heavily with the salt/ions in the buffer.
    # It uses tanh because the charge state can swing from deeply acidic (-) to
    # highly basic (+).
    # ------------------------------------------------------------------------- #
    # ENVIRONMENTAL CONCEPTS
    # A protein's net charge interacts heavily with the salt/ions in the buffer.
    # It uses tanh because the charge state can swing from deeply acidic (-)
    # to highly basic (+).
    ("charge_environment", "charge_x_ionic", +1, "tanh"),
    # Salts and excipients act as a "screen" that hides protein charges from each other.
    # This targets the raw ionic strength proxy. It is a sigmoid because
    # Tying it to sqrt([SALT] + [EXCIPIENT]) allows for nonlinear scaling factor to be tracked.
    ("ionic_screening", "Total_Ionic_Proxy", +1, "sigmoid"),
    # ------------------------------------------------------------------------- #
    # VOLUME AND SPACING CONCEPTS
    # The simplest physical rule. The higher the mg/mL,
    # the less free space there is (excluded volume).
    ("crowding", "Protein_conc", +1, "sigmoid"),
    # Viscosity doesn't rise in a straight line; it curves upwards aggressively
    # at high concentrations (due to jamming and multi-body collisions).
    # The square of the concentration c^2 proxies this exponential "wall."
    ("nonlinear_conc", "conc_sq", +1, "sigmoid"),
    # ------------------------------------------------------------------------- #
    # ADDITIVE CONCEPTS
    # Excipients (like sucrose or arginine) are added to interact with the protein.
    # Because they can either bind to the protein or be preferentially excluded from it,
    # they can drive the system in two different directions (tanh).
    ("cosolute_interaction", "Stabilizer_mg_mL", -1, "tanh"),
    # This represents how much "padding" the stabilizers provide to prevent the proteins
    # from crashing into each other. Crowding between stabilizers * proteins in terms of mg/mL
    # is what is tracked here.
    ("cosolute_protection", "Crowding_Index", -1, "sigmoid"),
    # SURFACE STABILITY
    # Surfactants prevent surface-induced aggregation/viscosity.
    # Uses Loading Ratio. Direction -1: more surfactant = less attraction/stickiness.
    ("interfacial_shielding", "Surfactant_Loading", -1, "sigmoid"),
    # This concept can track the mitigation of surface-induced aggregation.
    ("surface_stability", "Surfactant_Protein_Ratio", +1, "sigmoid"),
    # ------------------------------------------------------------------------- #
    # EXCIPIENT CONCEPTS
    # Moves away from raw mass to molar interactions.
    # Uses tanh to capture both binding (shielding) and exclusion (crowding).
    ("chemical_modulation", "Excipient_Molar_Ratio", -1, "tanh"),
    # REFINED SCREENING (Update)
    # Now uses the total ionic strength from both salt and charged excipients.
    ("electrostatic_screening", "Total_Ionic_Strength", +1, "sigmoid"),
    # ------------------------------------------------------------------------- #
]

V_BAR_REGISTRY = {
    # Salts
    "NaCl": 0.31,
    "Salt_default": 0.30,
    # Amino Acids (Excipients)
    "Arginine": 0.70,
    "Lysine": 0.72,
    "Proline": 0.76,
    "Excipient_default": 0.70,
    # Sugars (Stabilizers)
    "Sucrose": 0.63,
    "Trehalose": 0.62,
    "Stabilizer_default": 0.62,
    # Surfactants
    "Tween-20": 0.89,
    "Tween-80": 0.91,
    "Surfactant_default": 0.90,
    # Protein
    "Protein_default": 0.73,
}


# Convenience aliases derived from CONCEPT_DEFS
N_CONCEPTS_SUPERVISED: int = len(CONCEPT_DEFS)
CONCEPT_NAMES: list[str] = [cd[0] for cd in CONCEPT_DEFS]
CONCEPT_ACTIVATIONS: list[str] = [cd[3] for cd in CONCEPT_DEFS]
