# train_pinn.py

from sklearn.model_selection import train_test_split
import os
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

from pinn_domain import DataLoader
from pinn_net import MLPHyperModel
from pinn_constraints import (
    MonotonicIncreasingConstraint,
    MonotonicDecreasingConstraint,
    FlatSlopeConstraint,
    ShearThinningConstraint,
    ArrheniusConstraint,
    GaussianBellAroundPIConstraint,
    EinsteinDiluteLimitConstraint,
    ExcludedVolumeDivergenceConstraint,
)
from pinn_loss import composite_loss

csv_path = os.path.join('content', 'formulation_data_05152025.csv')
loader = DataLoader(csv_path)
loader.load()
loader.build_preprocessor()
X, y = loader.split(preprocess=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

feature_names = loader.NUMERIC_FEATURES
"""
- MonotonicIncreasingConstraint(feature_names, ["Protein_concentration", "Sugar_concentration"]):
    Both protein and sugar raise solution viscosity as their concentrations increase.

- MonotonicDecreasingConstraint(feature_names, ["Temperature", "Surfactant_concentration"]):
    Higher temperature reduces viscosity via thermal agitation, and surfactants lower interfacial drag.
    
- FlatSlopeConstraint(feature_names, "Buffer_pH"):
    Outside of charge-based aggregation effects, buffer pH has negligible net impact-enforce zero slope globally.

---- MORE INTERESTING IDEAS ----
- ShearThinningConstraint():
    Many fluids shear-hin: viscosity decreases with increasing shear rate.

- ArrheniusConstraint(feature_names, "Temperature"):
    Viscosity typically follows an Arrhenius law: log(viscosity) inverse proportional 1/T.

- GaussianBellAroundPIConstraint(feature_names, "Buffer_pH", "PI_mean"):
    Protein solutions exhibit a bell-shaped viscosity peak at the isoelectric point (pH ~ pI).

- EinsteinDiluteLimitConstraint(feature_names, "Protein_concentration", threshold=0.05):
    In the dilute limit (phi -> 0), relative viscosity → 1 + 2.5·phi (Einstein's relation).

- ExcludedVolumeDivergenceConstraint(feature_names, "Protein_concentration"):
    At high concentrations, crowding causes a rapid, convex-up rise in viscosity as phi→phi_max.
"""
constraints = [
    MonotonicIncreasingConstraint(
        feature_names, ["Protein_concentration", "Sugar_concentration"], weight=1e-2),
    MonotonicDecreasingConstraint(
        feature_names, ["Temperature", "Surfactant_concentration"], weight=1e-2),
    FlatSlopeConstraint(feature_names, "Buffer_pH", weight=1e-2),
    ShearThinningConstraint(weight=1e-2),
    ArrheniusConstraint(feature_names, "Temperature", weight=1e-2),
    GaussianBellAroundPIConstraint(
        feature_names, "Buffer_pH", "PI_mean", weight=1e-2),
    EinsteinDiluteLimitConstraint(
        feature_names, "Protein_concentration", threshold=0.05, weight=1e-2),
    ExcludedVolumeDivergenceConstraint(
        feature_names, "Protein_concentration", weight=1e-2),
]
X_train = X_train.astype("float32")
y_train = y_train.astype("float32")
X_val = X_val.astype("float32")
y_val = y_val.astype("float32")
INPUT_DIM = X_train.shape[1]
OUTPUT_DIM = y_train.shape[1]
hypermodel = MLPHyperModel(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory="tuner_logs",
    project_name="pinn_viscosity",
    overwrite=True,
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)


def run_trial_with_physics(model, X_train, y_train, X_val, y_val, constraints, epochs=100, batch_size=16):
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .batch(batch_size)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x, y):
        y = tf.cast(y, tf.float32)
        with tf.GradientTape() as tape:
            total_loss, data_loss, phys_loss = composite_loss(
                model, x, y, constraints, data_weight=1.0
            )
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, data_loss, phys_loss

    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in train_dataset:
            train_step(x_batch, y_batch)
        val_total, val_data, val_phys = composite_loss(
            model, X_val, y_val, constraints
        )
        print(
            f"Epoch {epoch:03d} | val_data={val_data:.4e} | val_phys={val_phys:.4e}")
        if val_data < best_val:
            best_val = val_data
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                print("Early stopping at epoch", epoch)
                break


tuner.search(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=1,
)

for trial in tuner.oracle.get_best_trials(num_trials=3):
    print(f"\n--- Tuning trial: {trial.trial_id} ---")
    hp = trial.hyperparameters
    model = tuner.hypermodel.build(hp)
    run_trial_with_physics(model, X_train, y_train, X_val, y_val, constraints)


best_model = model
best_model.save("best_pinn_model.h5")
