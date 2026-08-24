# Graph Report - VisQ.AI  (2026-07-16)

## Corpus Check
- 62 files · ~26,423 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 432 nodes · 663 edges · 41 communities (35 shown, 6 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 168 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `856aced5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- IBAL Learning-Curve Convergence Analysis
- O-Net CNP Inference (ViscosityPredictorCNP)
- Model Packaging & Signing
- ML Architecture Docs & Dependencies
- Charge Feature Engineering
- Categorical Feature Engineering
- Physics-Informed Design Rationale
- CodeQL Security Workflow
- Graphify Project Config (CLAUDE.md)
- plot_convergence
- test_charge_features.py
- run_convergence_replay
- run_cnp_fold
- ViscosityPredictorCNP
- configure_logging
- calc_metrics
- calculate_row_priors
- compute_shape_metrics
- context_selection.py
- run_baseline_fold
- ModuleSigner
- test_logging_config.py
- Feature provenance: kP, HCI, C_Class (Phase 0 / P1)
- constants.py
- visqai

## God Nodes (most connected - your core abstractions)
1. `build_feature_frame()` - 16 edges
2. `featurize_chemical_categoricals()` - 13 edges
3. `ViscosityPredictorCNP` - 13 edges
4. `plot_convergence()` - 12 edges
5. `run_convergence_replay()` - 11 edges
6. `plot_mape()` - 11 edges
7. `plot_log_convergence()` - 11 edges
8. `configure_logging()` - 11 edges
9. `main()` - 10 edges
10. `run_cnp_fold()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_prepare_df_coerces_ints_and_id_to_str()` --calls--> `prepare_df()`  [INFERRED]
  tests/unit/test_metrics.py → src/visqai/eval/data_prep.py
- `test_prepare_df_default_matches_no_drop_behavior()` --calls--> `prepare_df()`  [INFERRED]
  tests/unit/test_metrics.py → src/visqai/eval/data_prep.py
- `test_prepare_df_drop_bad_rows_filters_invalid_viscosity_and_numerics()` --calls--> `prepare_df()`  [INFERRED]
  tests/unit/test_metrics.py → src/visqai/eval/data_prep.py
- `test_calc_metrics_empty_after_masking_returns_nans()` --calls--> `calc_metrics()`  [INFERRED]
  tests/unit/test_metrics.py → src/visqai/eval/metrics.py
- `test_calc_metrics_masks_non_positive_and_nonfinite()` --calls--> `calc_metrics()`  [INFERRED]
  tests/unit/test_metrics.py → src/visqai/eval/metrics.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GitHub Actions CI Pipeline (CodeQL, Pylint, Python App)** — _github_workflows_codeql_workflow, _github_workflows_pylint_workflow, _github_workflows_python_app_workflow [INFERRED 0.85]

## Communities (41 total, 6 thin omitted)

### Community 0 - "IBAL Learning-Curve Convergence Analysis"
Cohesion: 0.16
Nodes (18): describe_property_space(), featurize_chemical_categoricals(), _keys(), _lookup(), _normalize_category(), categorical_features.py ======================= Rung-1 representation upgrade:, Lowercase, strip, and map empty/nan-likes to 'none'., Map one category value to its ordered descriptor vector.      Substring match (+10 more)

### Community 1 - "O-Net CNP Inference (ViscosityPredictorCNP)"
Cohesion: 0.14
Nodes (8): AttentionPool, CrossSampleCNP, cnp.py ====== The Cross-Sample Conditional Neural Process architecture (Attentio, _discover_checkpoints(), Path, Regression gate for the AttentionPool/CrossSampleCNP merge (previously two indep, test_attention_pool_output_matches_latent_dim(), test_merged_cnp_loads_existing_checkpoint_strict()

### Community 5 - "ML Architecture Docs & Dependencies"
Cohesion: 0.50
Nodes (4): Pylint Workflow, Python Application Workflow, ml/requirements.txt, Root requirements.txt

### Community 8 - "Charge Feature Engineering"
Cohesion: 0.09
Nodes (23): _build_ctx_tensor(), _build_tgt_tensors(), compute_viscosity_weights(), load_and_preprocess(), data.py ======= Training-time data loading: load_and_preprocess builds a fitted, Build a context tensor [1, N_points, 2+static_dim] from sample indices., Build query tensors for target samples., Compute per-point loss weights based on viscosity magnitude.      Points above ` (+15 more)

### Community 9 - "Categorical Feature Engineering"
Cohesion: 0.18
Nodes (17): _all_charge_columns(), _all_property_columns(), build_feature_frame(), _get_mw(), protected_feature_indices(), DataFrame, Series, pipeline.py =========== The single row-level feature-engineering pipeline shared (+9 more)

### Community 19 - "plot_convergence"
Cohesion: 0.13
Nodes (29): _run_and_plot(), plot_convergence(), plot_log_convergence(), plot_mape(), plot_sample_profile(), plot_shape_convergence(), DataFrame, convergence.py =============== Convergence-replay plots (linear cP, MAPE, log10 (+21 more)

### Community 20 - "test_charge_features.py"
Cohesion: 0.12
Nodes (32): Index, audit(), charge_coupling_index(), featurize_charge(), _ionic_valence(), normalize_charge_columns(), _numeric_col(), _protein_present() (+24 more)

### Community 21 - "run_convergence_replay"
Cohesion: 0.12
Nodes (21): _init_clean_predictor(), encode_context(), DataFrame, convergence_replay.py ======================= Step-by-step context-addition repl, Adds ibalizumab samples one-by-one in `ordered_ids` order. At each     step: enc, Reset memory, apply diverse context selection (if the engine exposes     it), th, Build one _index.csv row including shape metrics for triage/sorting., At a single replay step, render predicted-vs-actual profiles for every     sampl (+13 more)

### Community 22 - "run_cnp_fold"
Cohesion: 0.14
Nodes (23): DataFrame, cnp_logo.py =========== CNP side of the Phase 0 leave-one-GROUP-out harness: t, Train one fold's model and score zero-shot + few-shot log10 error on     the he, Run the CNP LOGO harness over every group for `axis` (or a     caller-supplied, Average calc_metrics (pooled across all shear columns) over     n_repeats rando, run_cnp_fold(), run_cnp_logo(), _shot_metrics() (+15 more)

### Community 23 - "ViscosityPredictorCNP"
Cohesion: 0.11
Nodes (20): DataFrame, ndarray, predictor.py ============ ViscosityPredictorCNP: loads a trained checkpoint + fi, Adapts the predictor to a new protein group by encoding its context         samp, Predicts using the cached memory (calibrated state)., Estimates the model's predictive uncertainty via MC Dropout., Inverse-scales a decoder output tensor to log10 viscosity values., ViscosityPredictorCNP (+12 more)

### Community 24 - "configure_logging"
Cohesion: 0.10
Nodes (24): _load_order_ids(), _log_summary(), main(), parse_args(), learning_curve.py ================== Replays the optimal (and a random-baselin, Load the sample-addition order from `order_csv`. Accepts either:      - a `Sam, main(), parse_args() (+16 more)

### Community 25 - "calc_metrics"
Cohesion: 0.09
Nodes (26): main(), parse_args(), parity_eval.py =============== Combined Ibalizumab CNP context experiment: selec, prepare_df(), DataFrame, data_prep.py ============ prepare_df: int->float coercion + ID->str, with an opt, calc_metrics(), compute_metrics() (+18 more)

### Community 26 - "calculate_row_priors"
Cohesion: 0.16
Nodes (17): calculate_cci(), calculate_regime(), calculate_row_priors(), priors.py ========= Physics-prior lookup tables and the charge-coupling-index, Charge-coupling index: peaks (i.e. -> C_Class) when the protein sits at     its, Map a CCI value to a Near-pI/Mixed/Far regime, with per-protein-class     thres, Per-row prior/concentration-split features. Matches process_row_features     (t, This is the direct regression test for the charge-features bug fix:     inferen (+9 more)

### Community 27 - "compute_shape_metrics"
Cohesion: 0.17
Nodes (15): _aggregate_shape(), _classify_slopes(), compute_shape_metrics(), ndarray, shape_metrics.py ================ Shape-fidelity metrics for viscosity shear-rat, Aggregate per-sample shape metrics over a set of profiles.      shape_rmse_log10, Per-segment direction: -1 thinning, 0 flat, +1 thickening., Shape-fidelity metrics for one profile (actual vs predicted, linear cP).      Re (+7 more)

### Community 28 - "context_selection.py"
Cohesion: 0.20
Nodes (14): _ctx_indices(), greedy_select(), _held_out_errors(), _objective(), preprocess_pool(), ndarray, context_selection.py ===================== Greedy forward selection (+ optional, Try replacing each selected member with each non-member; keep improvements. (+6 more)

### Community 29 - "run_baseline_fold"
Cohesion: 0.27
Nodes (13): Pipeline, fit_baseline(), _make_pipeline(), _melt_long(), DataFrame, baseline.py =========== Phase 0 reference baseline: a plain feature-only regre, Run the baseline over every LOGO group for `axis` (or a caller-supplied     sub, One row per (sample, shear rate): every engineered static feature,     plus log (+5 more)

### Community 30 - "ModuleSigner"
Cohesion: 0.07
Nodes (21): Any, main(), parse_args(), package_model.py ================= Build a signed deployment package from the la, get_latest_checkpoints(), packager.py =========== SecurePredictorPackager: builds a cryptographically sign, Finds .pt files in the most recently modified directory within experiments_dir., Package a visqai model with the runtime-inference source modules it needs. (+13 more)

### Community 31 - "test_logging_config.py"
Cohesion: 0.15
Nodes (6): Each test gets a clean, unconfigured state and leaves loguru without     danglin, file_level defaults more verbose (DEBUG) than console (INFO) -- a     DEBUG mess, The InterceptHandler is the mechanism that lets every existing     logging.getLo, _reset_logging_state(), test_configure_logging_respects_file_level_below_console(), test_stdlib_logging_is_routed_into_loguru_file_sink()

### Community 34 - "Feature provenance: kP, HCI, C_Class (Phase 0 / P1)"
Cohesion: 0.40
Nodes (4): Feature provenance: kP, HCI, C_Class (Phase 0 / P1), Recommendation, What this does NOT prove, What was checked

## Knowledge Gaps
- **9 isolated node(s):** `visqai`, `graphify`, `What was checked`, `What this does NOT prove`, `Recommendation` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_feature_frame()` connect `Categorical Feature Engineering` to `IBAL Learning-Curve Convergence Analysis`, `Charge Feature Engineering`, `test_charge_features.py`, `ViscosityPredictorCNP`, `calculate_row_priors`, `run_baseline_fold`?**
  _High betweenness centrality (0.312) - this node is a cross-community bridge._
- **Why does `ViscosityPredictorCNP` connect `ViscosityPredictorCNP` to `calc_metrics`, `run_convergence_replay`, `run_cnp_fold`, `O-Net CNP Inference (ViscosityPredictorCNP)`?**
  _High betweenness centrality (0.280) - this node is a cross-community bridge._
- **Why does `configure_logging()` connect `configure_logging` to `calc_metrics`, `ModuleSigner`?**
  _High betweenness centrality (0.197) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `build_feature_frame()` (e.g. with `_melt_long()` and `._preprocess()`) actually correct?**
  _`build_feature_frame()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `featurize_chemical_categoricals()` (e.g. with `build_feature_frame()` and `test_all_chem_categoricals_covered_in_output()`) actually correct?**
  _`featurize_chemical_categoricals()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `ViscosityPredictorCNP` (e.g. with `_init_clean_predictor()` and `main()`) actually correct?**
  _`ViscosityPredictorCNP` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `plot_convergence()` (e.g. with `_run_and_plot()` and `annotate_best()`) actually correct?**
  _`plot_convergence()` has 9 INFERRED edges - model-reasoned connections that need verification._