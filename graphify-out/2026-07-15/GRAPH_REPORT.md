# Graph Report - .  (2026-07-15)

## Corpus Check
- Corpus is ~28,285 words - fits in a single context window. You may not need a graph.

## Summary
- 257 nodes · 391 edges · 19 communities (17 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.84)
- Token cost: 55,222 input · 0 output

## Community Hubs (Navigation)
- IBAL Learning-Curve Convergence Analysis
- O-Net CNP Inference (ViscosityPredictorCNP)
- Model Packaging & Signing
- IBAL Parity Testing
- O-Net Training (Rung 1)
- ML Architecture Docs & Dependencies
- Output Mapping & Dimensionality Reduction
- Training Pipeline & Ensemble Tuning
- Charge Feature Engineering
- Categorical Feature Engineering
- Physics-Informed Design Rationale
- CodeQL Security Workflow
- Graphify Project Config (CLAUDE.md)

## God Nodes (most connected - your core abstractions)
1. `run_convergence_replay()` - 11 edges
2. `plot_convergence()` - 11 edges
3. `ViscosityPredictorCNP` - 10 edges
4. `plot_mape()` - 10 edges
5. `plot_log_convergence()` - 10 edges
6. `main()` - 9 edges
7. `main()` - 8 edges
8. `ModuleSigner` - 8 edges
9. `featurize_charge()` - 7 edges
10. `_score_set()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Python Application Workflow` --references--> `VisQAI: Physics-Informed Viscosity Prediction Library`  [INFERRED]
  .github/workflows/python-app.yml → ml/README.md
- `Root requirements.txt` --shares_data_with--> `ml/requirements.txt`  [INFERRED]
  requirements.txt → ml/requirements.txt
- `load_and_preprocess()` --calls--> `featurize_chemical_categoricals()`  [INFERRED]
  ml/cnp_mk2/train_o_net_v4_rung1.py → ml/cnp_mk2/categorical_features.py
- `_load_predictor_class()` --indirect_call--> `ViscosityPredictorCNP`  [INFERRED]
  ml/cnp_mk2/ibal_parity_test.py → ml/cnp_mk2/inference_o_net.py
- `Pylint Workflow` --semantically_similar_to--> `Python Application Workflow`  [INFERRED] [semantically similar]
  .github/workflows/pylint.yml → .github/workflows/python-app.yml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GitHub Actions CI Pipeline (CodeQL, Pylint, Python App)** — _github_workflows_codeql_workflow, _github_workflows_pylint_workflow, _github_workflows_python_app_workflow [INFERRED 0.85]
- **Physics-Informed ML Design Pattern (Priors + Loss Constraints)** — ml_readme_physics_informed_loss, ml_readme_learnable_physics_priors, ml_readme_learnablephysicsprior_layer, ml_readme_shear_thinning_constraint, ml_readme_input_gradient_constraints [INFERRED 0.85]

## Communities (19 total, 2 thin omitted)

### Community 0 - "IBAL Learning-Curve Convergence Analysis"
Cohesion: 0.09
Nodes (48): _aggregate_shape(), _annotate_best(), _annotate_convergence(), apply_base_style(), _classify_slopes(), compute_metrics(), compute_shape_metrics(), _encode_context() (+40 more)

### Community 1 - "O-Net CNP Inference (ViscosityPredictorCNP)"
Cohesion: 0.11
Nodes (16): AttentionPool, CrossSampleCNP, DataFrame, ndarray, Predicts using the cached memory (calibrated state)., Estimates the model's predictive uncertainty via MC Dropout.          The memo, Inverse-scales a decoder output tensor to log10 viscosity values., Append physicochemical property columns for the chemical categoricals.      Re (+8 more)

### Community 2 - "Model Packaging & Signing"
Cohesion: 0.12
Nodes (17): Any, Path, get_latest_checkpoints(), main(), ModuleSigner, Create the secure zip package., RSA-based signing for secure package verification., Create comprehensive metadata dictionary. (+9 more)

### Community 3 - "IBAL Parity Testing"
Cohesion: 0.13
Nodes (26): _apply_style(), build_long(), calc_metrics(), _ctx_indices(), greedy_select(), _held_out_errors(), _load_predictor_class(), main() (+18 more)

### Community 4 - "O-Net Training (Rung 1)"
Cohesion: 0.11
Nodes (19): AttentionPool, _build_ctx_tensor(), _build_tgt_tensors(), compute_viscosity_weights(), CrossSampleCNP, log_flatness(), log_latent_variance(), objective_cv() (+11 more)

### Community 5 - "ML Architecture Docs & Dependencies"
Cohesion: 0.10
Nodes (21): Pylint Workflow, Python Application Workflow, Categorical Embeddings (Protein Type, Buffer Type), Charge-Charge Interaction (CCI) score, Complex Protein-Excipient Interactions (design challenge), Concentration Splitting (E_low / E_high), src.data DataProcessor Module, Deep Residual Network (Residual Blocks) (+13 more)

### Community 6 - "Output Mapping & Dimensionality Reduction"
Cohesion: 0.10
Nodes (20): get_optimal_clusters(), get_predictions(), perform_clustering(), plot_2d_pca(), plot_2d_tsne(), plot_3d_pca(), plot_3d_tsne(), plot_parallel_coordinates_custom() (+12 more)

### Community 7 - "Training Pipeline & Ensemble Tuning"
Cohesion: 0.18
Nodes (14): DataProcessor, EnsembleModel, Module, check_model_health(), objective_cv(), ndarray, Primary Training and Tuning Script for VisQAI. Handles Optuna hyperparameter op, Run hyperparameter tuning. (+6 more)

### Community 8 - "Charge Feature Engineering"
Cohesion: 0.23
Nodes (14): audit(), charge_coupling_index(), featurize_charge(), normalize_charge_columns(), _protein_present(), DataFrame, charge_features.py ================== Rung-2 representation upgrade: turn the, Rename the raw CSV headers ('Charge', 'ProtPi PI', stray 'Unnamed: *')     to t (+6 more)

### Community 9 - "Categorical Feature Engineering"
Cohesion: 0.24
Nodes (11): describe_property_space(), featurize_chemical_categoricals(), _keys(), _lookup(), _normalize_category(), DataFrame, categorical_features.py ======================= Rung-1 representation upgrade:, Lowercase, strip, and map empty/nan-likes to 'none'. (+3 more)

### Community 10 - "Physics-Informed Design Rationale"
Cohesion: 0.67
Nodes (3): Data Scarcity in High-Viscosity Regions (design challenge), Learnable Physics Priors, Physics-Informed Loss

## Knowledge Gaps
- **8 isolated node(s):** `CodeQL Advanced Workflow`, `Pylint Workflow`, `Graphify Knowledge Graph Workflow Rules`, `ml/requirements.txt`, `Non-Linear Concentration Effects (design challenge)` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ViscosityPredictorCNP` connect `O-Net CNP Inference (ViscosityPredictorCNP)` to `IBAL Parity Testing`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `_load_predictor_class()` connect `IBAL Parity Testing` to `O-Net CNP Inference (ViscosityPredictorCNP)`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **What connects `CodeQL Advanced Workflow`, `Pylint Workflow`, `Graphify Knowledge Graph Workflow Rules` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `IBAL Learning-Curve Convergence Analysis` be split into smaller, more focused modules?**
  _Cohesion score 0.09013605442176871 - nodes in this community are weakly interconnected._
- **Should `O-Net CNP Inference (ViscosityPredictorCNP)` be split into smaller, more focused modules?**
  _Cohesion score 0.10967741935483871 - nodes in this community are weakly interconnected._
- **Should `Model Packaging & Signing` be split into smaller, more focused modules?**
  _Cohesion score 0.1168091168091168 - nodes in this community are weakly interconnected._
- **Should `IBAL Parity Testing` be split into smaller, more focused modules?**
  _Cohesion score 0.13105413105413105 - nodes in this community are weakly interconnected._