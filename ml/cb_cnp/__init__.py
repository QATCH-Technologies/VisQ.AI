"""
cbm_cnp — Concept Bottleneck Conditional Neural Process
========================================================

Package structure
-----------------
constants       Protein class map, concept definitions, and derived constants.
models          AttentionPool, CrossSampleCNP, ConceptBottleneckCNP, helpers.
data_pipeline   Feature engineering, preprocessing, sample construction.
batch_utils     Context/target tensor builders used by the training loop.
trainer         train_epoch, validate.
diagnostics     Latent variance, concept analysis, parity, feature importance.
tuning          Optuna cross-validation objective.
train           Main execution script (run directly with ``python train.py``).
"""

__version__ = "3.0.0"
__release__ = "2026-03-27"
