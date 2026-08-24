# Graph Report - VisQ.AI  (2026-08-17)

## Corpus Check
- 97 files · ~111,062 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 998 nodes · 1773 edges · 57 communities (48 shown, 9 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 352 edges (avg confidence: 0.79)
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
- ViscosityPredictorCNP
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
- CrossSampleCNP
- ModuleSigner
- test_logging_config.py
- Feature provenance: kP, HCI, C_Class (Phase 0 / P1)
- test_predictor_local_residual.py
- constants.py
- visqai
- fv_charge_diagnostics.py
- load_heldout_panel
- test_predictor_kernel_corrector.py
- zero_shot_new_vs_existing_eda.py
- _effective_n_repeats
- load_and_preprocess
- _drop_blank_rows
- clinical_vs_preclinical_umap.py
- _fix_zero_variance_scale
- objective_cv
- run_bgg_ablation.sh
- run_logo_before_after.sh
- run_logo_multiseed.sh
- __init__.py
- objective_cv

## God Nodes (most connected - your core abstractions)
1. `ViscosityPredictorCNP` - 35 edges
2. `build_feature_frame()` - 23 edges
3. `build_gap_report()` - 19 edges
4. `fit_fv_regime_params()` - 19 edges
5. `CrossSampleCNP` - 17 edges
6. `_attach_scope()` - 16 edges
7. `_df()` - 16 edges
8. `featurize_chemical_categoricals()` - 15 edges
9. `configure_logging()` - 15 edges
10. `_make_predictor_stub()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `check_against_noise_band()`  [INFERRED]
  analysis/bgg_ablation_eval.py → src/visqai/eval/metrics.py
- `criterion_3_lobo_gated_encoding()` --calls--> `load_heldout_panel()`  [INFERRED]
  analysis/fv_regime_real_data_eval.py → src/visqai/eval/heldout_panels.py
- `test_belatacept_sized_group_reaches_the_empirically_verified_30_repeats()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py
- `test_large_group_is_unchanged_not_reduced()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py
- `test_reference_sized_group_is_unchanged()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GitHub Actions CI Pipeline (CodeQL, Pylint, Python App)** — _github_workflows_codeql_workflow, _github_workflows_pylint_workflow, _github_workflows_python_app_workflow [INFERRED 0.85]

## Communities (57 total, 9 thin omitted)

### Community 0 - "IBAL Learning-Curve Convergence Analysis"
Cohesion: 0.12
Nodes (23): describe_property_space(), featurize_chemical_categoricals(), _keys(), _lookup(), _normalize_category(), DataFrame, categorical_features.py ======================= Rung-1 representation upgrade:, Lowercase, strip, and map empty/nan-likes to 'none'. (+15 more)

### Community 1 - "O-Net CNP Inference (ViscosityPredictorCNP)"
Cohesion: 0.05
Nodes (61): _assert_context_gate(), _check_fold_feature_range(), _effective_n_repeats(), DataFrame, cnp_logo.py =========== CNP side of the Phase 0 leave-one-GROUP-out harness: t, Scales `n_repeats` up for held-out groups smaller than     REFERENCE_N_HELD_FOR, Average calc_metrics (pooled across all shear columns) over     n_repeats rando, Train one fold's model and score zero-shot + few-shot log10 error on     the he (+53 more)

### Community 5 - "ML Architecture Docs & Dependencies"
Cohesion: 0.50
Nodes (4): Pylint Workflow, Python Application Workflow, ml/requirements.txt, Root requirements.txt

### Community 8 - "Charge Feature Engineering"
Cohesion: 0.18
Nodes (12): log_flatness(), log_latent_variance(), loop.py ======= Training/validation loops for CrossSampleCNP: train_epoch (the c, Pure zero-shot validation: predicts every held-out point using the     EXACT r=0, Randomized-context validation: n_repeats random splits per group,     averaged,, Mean pairwise L2 distance between PROTEIN-ONLY group latent centroids     (buffe, Diagnoses collapse-to-population-mean directly: for a sample of     protein grou, validate() (+4 more)

### Community 9 - "Categorical Feature Engineering"
Cohesion: 0.08
Nodes (39): calculate_cci(), calculate_regime(), calculate_row_priors(), priors.py ========= Physics-prior lookup tables and the charge-coupling-index, Charge-coupling index: peaks (i.e. -> C_Class) when the protein sits at     its, Map a CCI value to a Near-pI/Mixed/Far regime, with per-protein-class     thres, Per-row prior/concentration-split features. Matches process_row_features     (t, _all_charge_columns() (+31 more)

### Community 10 - "ViscosityPredictorCNP"
Cohesion: 0.07
Nodes (34): DataFrame, ndarray, predictor.py ============ ViscosityPredictorCNP: loads a trained checkpoint + fi, Query-conditioned delta-corrector prediction (Task 1.1):         prior(x) + offs, The back half of predict(): prior + delta-corrector terms +         results-fram, Estimates the model's predictive uncertainty via MC Dropout., Inverse-scales a decoder output tensor to log10 viscosity values., `predictor.memory_vector = None` is the established "reset to         zero-shot" (+26 more)

### Community 19 - "plot_convergence"
Cohesion: 0.07
Nodes (49): main(), parse_args(), parity_eval.py =============== Combined Ibalizumab CNP context experiment: selec, _overlay_panel(), plot_charge_ablation_comparison(), DataFrame, charge_ablation.py =================== Side-by-side comparison plots for the cha, Render one figure overlaying full-vs-ablated for MAE/RMSE (linear cP),     MAPE, (+41 more)

### Community 20 - "test_charge_features.py"
Cohesion: 0.07
Nodes (57): Index, audit(), charge_coupling_index(), _check_and_log_join_match_rates(), featurize_charge(), FvChargeJoinError, _ionic_valence(), join_fv_charge_columns() (+49 more)

### Community 21 - "run_convergence_replay"
Cohesion: 0.07
Nodes (38): encode_context(), DataFrame, convergence_replay.py ======================= Step-by-step context-addition repl, Adds ibalizumab samples one-by-one in `ordered_ids` order. At each     step: enc, Reset memory, apply diverse context selection (if the engine exposes     it), th, Build one _index.csv row including shape metrics for triage/sorting., At a single replay step, render predicted-vs-actual profiles for every     sampl, render_step_profiles() (+30 more)

### Community 22 - "run_cnp_fold"
Cohesion: 0.27
Nodes (13): Pipeline, fit_baseline(), _make_pipeline(), _melt_long(), DataFrame, baseline.py =========== Phase 0 reference baseline: a plain feature-only regre, Run the baseline over every LOGO group for `axis` (or a caller-supplied     sub, One row per (sample, shear rate): every engineered static feature,     plus log (+5 more)

### Community 23 - "ViscosityPredictorCNP"
Cohesion: 0.06
Nodes (69): _apply_quick_preset(), main(), parse_args(), condition_shift_eval.py ======================== Task 0.1 scoreboard (issue1_que, axis_rollup(), buffer_splits(), concentration_split(), ingredient_splits() (+61 more)

### Community 24 - "configure_logging"
Cohesion: 0.07
Nodes (38): LogRecord, main(), parse_args(), learning_curve_charge_ablation.py ================================== Two-pass va, _init_clean_predictor(), _load_order_ids(), _log_summary(), main() (+30 more)

### Community 25 - "calc_metrics"
Cohesion: 0.07
Nodes (40): _drop_targets(), _importance_for_arm(), main(), parse_args(), _permute_and_score(), DataFrame, charge_feature_importance.py ============================= Permutation feature i, Shared permutation loop for both tiers: `predict_fn(df) -> results_df`     is th (+32 more)

### Community 26 - "calculate_row_priors"
Cohesion: 0.08
Nodes (46): _assign_block(), criterion_2_cluster_count(), criterion_3_lobo_gated_encoding(), main(), ndarray, Task 4 acceptance criteria 2-4 (real-data evaluation). Criterion 1 lives in fv_r, Follow-on plan Task H: for an extrapolation fold, verify the     determinism a c, Count Fv-bearing PROTEIN clusters in the training-side source (the     unit a pr (+38 more)

### Community 27 - "compute_shape_metrics"
Cohesion: 0.07
Nodes (72): _attach_scope(), bin_series(), BinCount, build_gap_report(), check_score_non_degeneracy(), _class_a_bin_column(), class_a_collinearity_groups(), class_a_deficit() (+64 more)

### Community 28 - "context_selection.py"
Cohesion: 0.20
Nodes (14): _ctx_indices(), greedy_select(), _held_out_errors(), _objective(), preprocess_pool(), ndarray, context_selection.py ===================== Greedy forward selection (+ optional, Try replacing each selected member with each non-member; keep improvements. (+6 more)

### Community 29 - "CrossSampleCNP"
Cohesion: 0.10
Nodes (17): AttentionPool, CrossSampleCNP, cnp.py ====== The Cross-Sample Conditional Neural Process architecture (Attentio, Returns (prior, correction) separately -- training         (visqai.training.loop, Same as forward(), but returns (prior, correction) unsummed., _discover_checkpoints(), Path, Regression gate for the AttentionPool/CrossSampleCNP merge (previously two indep (+9 more)

### Community 30 - "ModuleSigner"
Cohesion: 0.07
Nodes (21): Any, main(), parse_args(), package_model.py ================= Build a signed deployment package from the la, get_latest_checkpoints(), packager.py =========== SecurePredictorPackager: builds a cryptographically sign, Finds .pt files in the most recently modified directory within experiments_dir., Package a visqai model with the runtime-inference source modules it needs. (+13 more)

### Community 31 - "test_logging_config.py"
Cohesion: 0.15
Nodes (6): Each test gets a clean, unconfigured state and leaves loguru without     danglin, file_level defaults more verbose (DEBUG) than console (INFO) -- a     DEBUG mess, The InterceptHandler is the mechanism that lets every existing     logging.getLo, _reset_logging_state(), test_configure_logging_respects_file_level_below_console(), test_stdlib_logging_is_routed_into_loguru_file_sink()

### Community 34 - "Feature provenance: kP, HCI, C_Class (Phase 0 / P1)"
Cohesion: 0.11
Nodes (27): determine_control_treated_groups(), load_one(), main(), Series, Aggregates the per-protein logo_scoreboard.csv files produced by analysis/run_lo, Welch's t-test (unequal variances) for the treated-vs-control     difference in, Programmatically classify each protein as 'control' (net_charge     bit-for-bit, welch_contrast() (+19 more)

### Community 38 - "test_predictor_local_residual.py"
Cohesion: 0.18
Nodes (21): _make_ctx(), _make_predictor_stub(), test_predictor_local_residual.py ================================= Task 1.1 (iss, Reproduces T-R3.2's old `_shrink_offset` formula by hand and confirms     Task 1, The generalized whole-model LOO transfer check     (_transfer_check_passes) must, Rule 2 (never regress the safe path): offset_hat=conc_hat=slope_hat=0     (the m, Even a degenerate [0, 0] support range (the reset/no-context default)     must n, One formulation per entry of `conc_values`, each with a real point at     every (+13 more)

### Community 41 - "fv_charge_diagnostics.py"
Cohesion: 0.23
Nodes (17): _assign_block(), check1_coverage(), check2_determinism(), check3_legacy_vs_new(), check4_zero_shot_blocks(), check5_lobo_probe(), check_composite_key_duplication(), gate_summary() (+9 more)

### Community 42 - "load_heldout_panel"
Cohesion: 0.18
Nodes (16): _calling_module(), HeldoutPanel, HeldoutPanelAccessError, load_heldout_panel(), _log_access(), DataFrame, RuntimeError, heldout_panels.py ================== Registry + load guard for the Fv/whole-anti (+8 more)

### Community 43 - "test_predictor_kernel_corrector.py"
Cohesion: 0.20
Nodes (16): _make_predictor_stub(), test_predictor_kernel_corrector.py =================================== Task 1.2, When residuals are pure noise (no real cluster structure), the     kernel-weight, Rule 2: corrector_mode='kernel' with no fitted kernel state (gate     never fire, 8 formulations in two well-separated clusters (in a single kernel     feature di, test_bandwidth_selection_prefers_smallest_passing_over_lowest_mae(), test_clean_two_cluster_signal_passes_gate_and_picks_small_bandwidth(), test_fewer_than_two_formulations_gate_fails() (+8 more)

### Community 44 - "zero_shot_new_vs_existing_eda.py"
Cohesion: 0.33
Nodes (9): build_feature_matrix(), load(), main(), nearest_neighbor_report(), plot_group_embedding(), plot_viscosity_context(), Compares the 24 new samples in "Zero-shot testdata 1.csv" (no Viscosity_1000 lab, Existing points colored by log(Viscosity_1000) magnitude; new points overlaid (+1 more)

### Community 45 - "_effective_n_repeats"
Cohesion: 0.23
Nodes (20): generate_all(), _hatch_thin_cells(), _label_points_with_collision_avoidance(), plot_p1_logo_support_matrix(), plot_p2_descriptor_occupancy(), plot_p2b_whitened_void_space(), plot_p3_concentration_grid(), plot_p4_response_coverage() (+12 more)

### Community 46 - "load_and_preprocess"
Cohesion: 0.24
Nodes (9): _build_ctx_tensor(), _build_tgt_tensors(), compute_viscosity_weights(), load_and_preprocess(), data.py ======= Training-time data loading: load_and_preprocess builds a fitted, Build a context tensor [1, N_points, 2+static_dim] from sample indices., Build query tensors for target samples., Compute per-point loss weights based on viscosity magnitude.      Points above ` (+1 more)

### Community 48 - "_drop_blank_rows"
Cohesion: 0.11
Nodes (18): DataFrame, test_coverage.py ================= Task 8 tests for visqai.analysis.coverage.  T, occupied + explicit-NaN bucket == total, for every binned axis this     module d, C-9's exact error class: REQUIRED_CLUSTERS_GENERALIZATION's native     unit_of_c, C-13's regression test: dropping poly-hIgG raises n_eff (11.94 ->     18.68) des, Retrospective check of C-10, evaluated against the population it     actually di, 500 rows, all the same non-placebo protein -- every class-A axis is     triviall, k clusters of size k each (n = k^2 rows) -- the specific construction     under (+10 more)

### Community 49 - "clinical_vs_preclinical_umap.py"
Cohesion: 0.48
Nodes (6): build_feature_matrix(), load(), main(), plot_clinical_vs_preclinical(), Visualizes how the 21 clinical mAbs (ID 161-502: Adalimumab, Bevacizumab, Pembro, run_umap()

### Community 50 - "_fix_zero_variance_scale"
Cohesion: 0.27
Nodes (9): _drop_blank_rows(), DataFrame, Pre-2.4 data-hygiene finding (issue1_query_conditioned_correction_     plan.md F, A row missing Protein_type but with a real Protein_conc is NOT the     blank/mal, The real-data case: a row with Protein_type AND Protein_conc both NaN,     but a, test_drop_blank_rows_keeps_row_with_only_protein_conc(), test_drop_blank_rows_leftover_stray_value_still_dropped(), test_drop_blank_rows_no_blanks_is_a_noop() (+1 more)

### Community 51 - "objective_cv"
Cohesion: 0.38
Nodes (7): ColumnTransformer, _fix_zero_variance_scale(), ndarray, Patch the fitted 'num' StandardScaler in place: any column that came     out zer, P0 fix: sklearn's StandardScaler sets scale_=1 for a zero-variance     training, test_no_zero_variance_columns_leaves_scaler_untouched(), test_zero_variance_column_gets_fallback_scale_not_degenerate_one()

### Community 56 - "objective_cv"
Cohesion: 0.40
Nodes (4): Few-shot held-out validation -- the metric that matches deployment.     For a he, validate_fewshot(), objective_cv(), tuning.py ========= Optuna hyperparameter search objective: group-held-out CV, s

## Knowledge Gaps
- **13 isolated node(s):** `run_bgg_ablation.sh script`, `PYTHONPATH`, `run_logo_before_after.sh script`, `PYTHONPATH`, `run_logo_multiseed.sh script` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_feature_frame()` connect `Categorical Feature Engineering` to `IBAL Learning-Curve Convergence Analysis`, `O-Net CNP Inference (ViscosityPredictorCNP)`, `ViscosityPredictorCNP`, `load_and_preprocess`, `test_charge_features.py`, `run_cnp_fold`, `calc_metrics`?**
  _High betweenness centrality (0.214) - this node is a cross-community bridge._
- **Why does `ViscosityPredictorCNP` connect `ViscosityPredictorCNP` to `O-Net CNP Inference (ViscosityPredictorCNP)`, `test_predictor_local_residual.py`, `test_predictor_kernel_corrector.py`, `plot_convergence`, `ViscosityPredictorCNP`, `configure_logging`, `calc_metrics`, `CrossSampleCNP`?**
  _High betweenness centrality (0.213) - this node is a cross-community bridge._
- **Why does `load_and_preprocess()` connect `load_and_preprocess` to `O-Net CNP Inference (ViscosityPredictorCNP)`, `Categorical Feature Engineering`, `_fix_zero_variance_scale`, `objective_cv`, `configure_logging`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `ViscosityPredictorCNP` (e.g. with `_init_clean_predictor()` and `main()`) actually correct?**
  _`ViscosityPredictorCNP` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `build_feature_frame()` (e.g. with `_importance_for_arm()` and `_melt_long()`) actually correct?**
  _`build_feature_frame()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `fit_fv_regime_params()` (e.g. with `criterion_3_lobo_gated_encoding()` and `main()`) actually correct?**
  _`fit_fv_regime_params()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **What connects `run_bgg_ablation.sh script`, `PYTHONPATH`, `run_logo_before_after.sh script` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._