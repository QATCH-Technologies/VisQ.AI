# Feature provenance: kP, HCI, C_Class (Phase 0 / P1)

Phase 0 P1 asks for documented provenance of `kP`, `HCI`, and `C_Class` --
the three protein-level features carrying most of the model's generalization
-- and a check that none of them are fit from the viscosity curves they're
used to predict (which would make every held-out benchmark optimistic).

## What was checked

**1. No code path in this repository derives kP/HCI/C_Class from viscosity.**
A repo-wide search for every file referencing `kP`, `HCI`, or `C_Class`
(`git grep`-equivalent across `src/`, `tests/`, `data/`) turns up only
*consumers*:

- `src/visqai/preprocessing/pipeline.py` (`BASE_NUMERIC_COLS`, `ENGINEERED_COLS`)
  reads all three straight off the input row and fills missing values with
  `0.0` -- no computation.
- `src/visqai/physics/priors.py` (`calculate_row_priors` / `calculate_cci`)
  uses `C_Class` as a multiplicative weight in the charge-coupling-index
  formula (`cci = C_Class * exp(-net_charge^2 / (2*sigma^2))`) -- again, read,
  not derived.
- `tests/unit/test_metrics.py`, `tests/unit/test_preprocessing_pipeline.py`
  reference them as fixture columns.

No script, notebook, or function anywhere under `src/` computes `kP`, `HCI`,
or `C_Class` from `Viscosity_*`. They arrive as raw columns in every
`data/raw/formulation_data_*.csv` file (see the CSV header:
`...,PI_range,Protein_conc,Temperature,...,C_Class,HCI,Viscosity_100,...` --
they are positioned as *input* covariates, upstream of the viscosity columns,
in the raw export itself).

**2. They are protein-constant, like MW and PI_mean.**
Grouping `data/raw/formulation_data_07062026.csv` by `Protein_type` and
counting unique values per group:

| feature | max distinct values within one protein | proteins with >1 value |
|---|---|---|
| kP | 1 | 0 / 13 |
| HCI | 1 | 0 / 13 |
| C_Class | 1 | 0 / 13 |
| PI_mean | 1 | 0 / 13 |
| MW | 1 | 0 / 13 |

Every row for a given protein carries the identical value -- they behave
exactly like the other known-independent protein descriptors (MW, PI_mean),
not like a per-well measurement or a fitted residual.

**3. Value shapes are consistent with hand-assigned physicochemical classes,
not a regression output.** E.g. Adalimumab: `kP=3.0, HCI=1.0, C_Class=1.0`.
Small-integer/round values across all 13 proteins are consistent with a
categorical/ordinal characterization (e.g. an interaction-parameter class, a
hydrophobicity class, a charge class from an external assay or literature
lookup) rather than a continuous value fit to this dataset's viscosity
curves.

## What this does NOT prove

This repository has no data dictionary, ingestion script, or README
documenting what upstream process actually computed `kP`, `HCI`, and
`C_Class` before they landed in `data/raw/*.csv`. The checks above rule out
**in-repo** leakage (nothing here fits them from viscosity), and the
protein-constant, round-value pattern is consistent with an external,
a-priori characterization -- but this codebase cannot independently confirm
*what* upstream tool or dataset produced them, or whether that upstream
process ever touched this project's own viscosity measurements.

The user's own check (Spearman(kP/HCI/C_Class, per-protein mean viscosity)
~= 0) is the strongest evidence available and is consistent with the
in-repo findings above, but it was performed outside this repository and
isn't independently reproducible from what's checked in here.

## Recommendation

Until the upstream computation is documented (ideally: a short note on what
assay/tool/reference produced each of `kP`, `HCI`, `C_Class`, committed
alongside `data/raw/`), treat this as: **no leakage found in-repo; upstream
provenance unconfirmed.** If a later audit finds any of the three was ever
back-computed from this project's viscosity data, every LOGO benchmark in
`visqai.cli.logo_eval` should be re-run and flagged as optimistic in the
scoreboard.
