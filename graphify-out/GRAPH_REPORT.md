# Graph Report - VisQ.AI  (2026-08-24)

## Corpus Check
- 78 files · ~60,768 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 792 nodes · 1404 edges · 46 communities (39 shown, 7 thin omitted)
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 280 edges (avg confidence: 0.79)
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
- SecurePredictorPackager
- test_predictor_local_residual.py
- constants.py
- visqai
- load_heldout_panel
- test_predictor_kernel_corrector.py
- _effective_n_repeats
- _drop_blank_rows
- __init__.py

## God Nodes (most connected - your core abstractions)
1. `ViscosityPredictorCNP` - 31 edges
2. `build_feature_frame()` - 21 edges
3. `build_gap_report()` - 19 edges
4. `CrossSampleCNP` - 18 edges
5. `_attach_scope()` - 16 edges
6. `_is_placebo()` - 15 edges
7. `featurize_chemical_categoricals()` - 15 edges
8. `_make_predictor_stub()` - 15 edges
9. `count_bin()` - 13 edges
10. `main()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `test_belatacept_sized_group_reaches_the_empirically_verified_30_repeats()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py
- `test_large_group_is_unchanged_not_reduced()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py
- `test_reference_sized_group_is_unchanged()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py
- `test_scaling_decreases_monotonically_with_group_size()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py
- `test_scaling_is_capped_at_max_multiplier()` --calls--> `_effective_n_repeats()`  [INFERRED]
  tests/unit/test_cnp_logo_repeats_scaling.py → src/visqai/eval/cnp_logo.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GitHub Actions CI Pipeline (CodeQL, Pylint, Python App)** — _github_workflows_codeql_workflow, _github_workflows_pylint_workflow, _github_workflows_python_app_workflow [INFERRED 0.85]

## Communities (46 total, 7 thin omitted)

### Community 0 - "IBAL Learning-Curve Convergence Analysis"
Cohesion: 0.12
Nodes (23): describe_property_space(), featurize_chemical_categoricals(), _keys(), _lookup(), _normalize_category(), DataFrame, categorical_features.py ======================= Rung-1 representation upgrade:, Lowercase, strip, and map empty/nan-likes to 'none'. (+15 more)

### Community 1 - "O-Net CNP Inference (ViscosityPredictorCNP)"
Cohesion: 0.08
Nodes (36): _assert_context_gate(), _check_fold_feature_range(), _effective_n_repeats(), cnp_logo.py =========== CNP side of the Phase 0 leave-one-GROUP-out harness: t, Scales `n_repeats` up for held-out groups smaller than     REFERENCE_N_HELD_FOR, Hard guardrail: context (few-shot) must never score worse than     zero-shot by, Guard (b)/(a) from the P0 fix: compare every held-out row's engineered     nume, _train_fold_model() (+28 more)

### Community 5 - "ML Architecture Docs & Dependencies"
Cohesion: 0.50
Nodes (4): Pylint Workflow, Python Application Workflow, ml/requirements.txt, Root requirements.txt

### Community 8 - "Charge Feature Engineering"
Cohesion: 0.06
Nodes (41): ColumnTransformer, _build_ctx_tensor(), _build_tgt_tensors(), compute_viscosity_weights(), _drop_blank_rows(), _fix_zero_variance_scale(), load_and_preprocess(), DataFrame (+33 more)

### Community 9 - "Categorical Feature Engineering"
Cohesion: 0.10
Nodes (31): calculate_cci(), calculate_regime(), calculate_row_priors(), priors.py ========= Physics-prior lookup tables and the charge-coupling-index, Charge-coupling index: peaks (i.e. -> C_Class) when the formulation pH     sits, Map a CCI value to a Near-pI/Mixed/Far regime, with per-protein-class     thres, Per-row prior/concentration-split features., _all_charge_columns() (+23 more)

### Community 10 - "ViscosityPredictorCNP"
Cohesion: 0.08
Nodes (25): DataFrame, ndarray, predictor.py ============ ViscosityPredictorCNP: loads a trained checkpoint + fi, Query-conditioned delta-corrector prediction (Task 1.1):         prior(x) + offs, The back half of predict(): prior + delta-corrector terms +         results-fram, Estimates the model's predictive uncertainty via MC Dropout., Inverse-scales a decoder output tensor to log10 viscosity values., `predictor.memory_vector = None` is the established "reset to         zero-shot" (+17 more)

### Community 19 - "plot_convergence"
Cohesion: 0.09
Nodes (40): main(), parse_args(), parity_eval.py =============== Combined Ibalizumab CNP context experiment: selec, plot_convergence(), plot_log_convergence(), plot_mape(), plot_sample_profile(), plot_shape_convergence() (+32 more)

### Community 20 - "test_charge_features.py"
Cohesion: 0.25
Nodes (14): featurize_charge(), normalize_charge_columns(), DataFrame, charge.py ========= Net protein charge feature: exactly one raw physical measure, Rename whichever raw whole-charge header is present to the internal     name `fe, Append `whole_charge`. 0.0 where absent (no protein present, or an     older CSV, _featurize(), DataFrame (+6 more)

### Community 21 - "run_convergence_replay"
Cohesion: 0.09
Nodes (29): _init_clean_predictor(), _load_order_ids(), _log_summary(), main(), parse_args(), learning_curve.py ================== Replays the optimal (and a random-baselin, Load the sample-addition order from `order_csv`. Accepts either:      - a `Sam, _run_and_plot() (+21 more)

### Community 22 - "run_cnp_fold"
Cohesion: 0.09
Nodes (40): Pipeline, _apply_quick_preset(), main(), parse_args(), logo_eval.py ============ Phase 0 scoreboard: leave-one-GROUP-out evaluation acr, fit_baseline(), _make_pipeline(), _melt_long() (+32 more)

### Community 23 - "ViscosityPredictorCNP"
Cohesion: 0.06
Nodes (69): _apply_quick_preset(), main(), parse_args(), condition_shift_eval.py ======================== Task 0.1 scoreboard (issue1_que, axis_rollup(), buffer_splits(), concentration_split(), ingredient_splits() (+61 more)

### Community 24 - "configure_logging"
Cohesion: 0.12
Nodes (23): main(), parse_args(), train.py ======== CLI training entrypoint: Optuna hyperparameter search (group, LogRecord, _drop_targets(), _load_zero_shot_df(), _log10_safe(), main() (+15 more)

### Community 25 - "calc_metrics"
Cohesion: 0.07
Nodes (34): Average calc_metrics (pooled across all shear columns) over     n_repeats rando, _shot_metrics(), prepare_df(), DataFrame, data_prep.py ============ prepare_df: int->float coercion + ID->str, with an opt, calc_metrics(), check_against_noise_band(), compute_metrics() (+26 more)

### Community 26 - "calculate_row_priors"
Cohesion: 0.17
Nodes (15): _aggregate_shape(), _classify_slopes(), compute_shape_metrics(), ndarray, shape_metrics.py ================ Shape-fidelity metrics for viscosity shear-rat, Aggregate per-sample shape metrics over a set of profiles.      shape_rmse_log10, Per-segment direction: -1 thinning, 0 flat, +1 thickening., Shape-fidelity metrics for one profile (actual vs predicted, linear cP).      Re (+7 more)

### Community 27 - "compute_shape_metrics"
Cohesion: 0.06
Nodes (78): _attach_scope(), bin_series(), BinCount, build_gap_report(), check_score_non_degeneracy(), _class_a_bin_column(), class_a_categorical_coverage(), class_a_collinearity_groups() (+70 more)

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

### Community 38 - "test_predictor_local_residual.py"
Cohesion: 0.18
Nodes (21): _make_ctx(), _make_predictor_stub(), test_predictor_local_residual.py ================================= Task 1.1 (iss, Reproduces T-R3.2's old `_shrink_offset` formula by hand and confirms     Task 1, The generalized whole-model LOO transfer check     (_transfer_check_passes) must, Rule 2 (never regress the safe path): offset_hat=conc_hat=slope_hat=0     (the m, Even a degenerate [0, 0] support range (the reset/no-context default)     must n, One formulation per entry of `conc_values`, each with a real point at     every (+13 more)

### Community 42 - "load_heldout_panel"
Cohesion: 0.16
Nodes (18): RuntimeError, class_a_deficit(), Retired (C-9): class-A ranking is degenerate at partition-cell scope     by cons, _calling_module(), HeldoutPanel, HeldoutPanelAccessError, load_heldout_panel(), _log_access() (+10 more)

### Community 43 - "test_predictor_kernel_corrector.py"
Cohesion: 0.20
Nodes (16): _make_predictor_stub(), test_predictor_kernel_corrector.py =================================== Task 1.2, When residuals are pure noise (no real cluster structure), the     kernel-weight, Rule 2: corrector_mode='kernel' with no fitted kernel state (gate     never fire, 8 formulations in two well-separated clusters (in a single kernel     feature di, test_bandwidth_selection_prefers_smallest_passing_over_lowest_mae(), test_clean_two_cluster_signal_passes_gate_and_picks_small_bandwidth(), test_fewer_than_two_formulations_gate_fails() (+8 more)

### Community 45 - "_effective_n_repeats"
Cohesion: 0.23
Nodes (20): generate_all(), _hatch_thin_cells(), _label_points_with_collision_avoidance(), plot_p1_logo_support_matrix(), plot_p2_descriptor_occupancy(), plot_p2b_whitened_void_space(), plot_p3_concentration_grid(), plot_p4_response_coverage() (+12 more)

### Community 48 - "_drop_blank_rows"
Cohesion: 0.11
Nodes (18): DataFrame, test_coverage.py ================= Task 8 tests for visqai.analysis.coverage.  T, occupied + explicit-NaN bucket == total, for every binned axis this     module d, C-9's exact error class: REQUIRED_CLUSTERS_GENERALIZATION's native     unit_of_c, C-13's regression test: dropping poly-hIgG raises n_eff (11.94 ->     18.68) des, Retrospective check of C-10, evaluated against the population it     actually di, 500 rows, all the same non-placebo protein -- every class-A axis is     triviall, k clusters of size k each (n = k^2 rows) -- the specific construction     under (+10 more)

## Knowledge Gaps
- **7 isolated node(s):** `visqai`, `HeldoutPanel`, `graphify`, `CodeQL Advanced Workflow`, `Pylint Workflow` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ViscosityPredictorCNP` connect `ViscosityPredictorCNP` to `test_predictor_local_residual.py`, `test_predictor_kernel_corrector.py`, `plot_convergence`, `run_convergence_replay`, `run_cnp_fold`, `ViscosityPredictorCNP`, `configure_logging`, `CrossSampleCNP`?**
  _High betweenness centrality (0.284) - this node is a cross-community bridge._
- **Why does `load_and_preprocess()` connect `Charge Feature Engineering` to `configure_logging`, `O-Net CNP Inference (ViscosityPredictorCNP)`, `Categorical Feature Engineering`?**
  _High betweenness centrality (0.248) - this node is a cross-community bridge._
- **Why does `_drop_blank_rows()` connect `Charge Feature Engineering` to `compute_shape_metrics`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `ViscosityPredictorCNP` (e.g. with `_init_clean_predictor()` and `main()`) actually correct?**
  _`ViscosityPredictorCNP` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `build_feature_frame()` (e.g. with `_melt_long()` and `_check_fold_feature_range()`) actually correct?**
  _`build_feature_frame()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `CrossSampleCNP` (e.g. with `_RemainderColsList` and `ViscosityPredictorCNP`) actually correct?**
  _`CrossSampleCNP` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `visqai`, `HeldoutPanel`, `graphify` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._