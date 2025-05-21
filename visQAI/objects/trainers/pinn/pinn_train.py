import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import EarlyStopping
from pinn_domain import DataLoader
from pinn_net import MLPHyperModel
from pinn_loss import composite_loss
from keras_tuner import RandomSearch, BayesianOptimization
from tensorflow.keras.optimizers.schedules import CosineDecay

# 1) Load and preprocess data
csv_path = os.path.join('content', 'formulation_data_05152025.csv')
loader = DataLoader(csv_path)
loader.load()
loader.build_preprocessor()
X, y = loader.split(preprocess=True)

y_mean = np.mean(y, axis=1)
y_bins = pd.qcut(y_mean, q=10, labels=False, duplicates='drop')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X, y_bins))
X_train = X[train_idx].astype('float32')
X_val = X[val_idx].astype('float32')
y_train = y[train_idx].astype('float32')
y_val = y[val_idx].astype('float32')

feature_names = loader.NUMERIC_FEATURES
INPUT_DIM = X_train.shape[1]
OUTPUT_DIM = y_train.shape[1]
hypermodel = MLPHyperModel(INPUT_DIM, OUTPUT_DIM, feature_names)
tuner = BayesianOptimization(
    hypermodel,
    objective='val_loss',
    max_trials=20,
    num_initial_points=5,         # how many random trials before Bayesian kicks in
    alpha=1e-4,                   # GP exploration parameter
    beta=2.6,                     # GP exploitation parameter
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='pinn_viscosity_bayes',
    overwrite=True,
)

global_optimizer = tf.keras.optimizers.Adam()
dummy_hp = tuner.oracle.get_space()
dummy_model = hypermodel.build(dummy_hp)
zero_grads = [tf.zeros_like(v) for v in dummy_model.trainable_variables]
global_optimizer.apply_gradients(
    zip(zero_grads, dummy_model.trainable_variables))


@tf.function
def train_step(model, x_batch, y_batch, constraints):
    y_batch = tf.cast(y_batch, tf.float32)
    with tf.GradientTape() as tape:
        total_loss, data_loss, phys_loss = composite_loss(
            model, x_batch, y_batch, constraints=constraints, data_weight=1.0
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    global_optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, data_loss, phys_loss


tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16
)


def run_trial_with_physics(
    model,
    X_train, y_train,
    X_val,   y_val,
    extreme_fraction: float = 0.2,
    epochs:           int = 100,
    batch_size:       int = 16,
):
    # pull physics constraints off the model
    constraints = getattr(model, "_physics_constraints", [])

    # 2) Identify “extreme” indices by how far y deviates from its mean
    deviations = np.linalg.norm(y_train - y_train.mean(axis=0), axis=1)
    n_extreme = int(extreme_fraction * len(deviations))
    extreme_idx = np.argsort(deviations)[-n_extreme:]
    normal_idx = np.setdiff1d(np.arange(len(y_train)), extreme_idx)

    # 3) Learning‐rate schedule: Cosine decay from initial LR down to zero
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = steps_per_epoch * epochs
    initial_lr = float(global_optimizer.learning_rate)
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=0.0
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # prime local optimizer slots as well
    zero_grads_local = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads_local, model.trainable_variables))

    best_val, wait = float("inf"), 0

    # 5) Main training loop with stratified batches
    for epoch in range(1, epochs + 1):
        np.random.shuffle(extreme_idx)
        np.random.shuffle(normal_idx)

        ptr_ext, ptr_norm = 0, 0
        n_ext_batch = max(1, int(extreme_fraction * batch_size))
        n_norm_batch = batch_size - n_ext_batch

        while ptr_norm < len(normal_idx):
            if ptr_ext + n_ext_batch > len(extreme_idx):
                np.random.shuffle(extreme_idx)
                ptr_ext = 0
            batch_ext_idx = extreme_idx[ptr_ext:ptr_ext + n_ext_batch]
            batch_norm_idx = normal_idx[ptr_norm:ptr_norm + n_norm_batch]

            ptr_ext += n_ext_batch
            ptr_norm += n_norm_batch

            idx_batch = np.concatenate([batch_ext_idx, batch_norm_idx])
            np.random.shuffle(idx_batch)

            x_batch = tf.convert_to_tensor(
                X_train[idx_batch], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(
                y_train[idx_batch], dtype=tf.float32)

            train_step(model, x_batch, y_batch, constraints)

        # validation loss
        val_total, val_data, val_phys = composite_loss(
            model, X_val, y_val, constraints=constraints
        )
        print(f"Epoch {epoch:03d}"
              f" | val_data={val_data:.4e}"
              f" | val_phys={val_phys:.4e}"
              f" | lr={optimizer._decayed_lr(tf.float32):.3e}")

        if val_data < best_val:
            best_val, wait = val_data, 0
        else:
            wait += 1
            if wait >= 10:
                print("Early stopping at epoch", epoch)
                break

    return model


# 6) Run best trials
best_models = []
for trial in tuner.oracle.get_best_trials(num_trials=3):
    hp = trial.hyperparameters
    model = hypermodel.build(hp)
    # prime optimizer slots for this new model
    zero_grads_temp = [tf.zeros_like(v) for v in model.trainable_variables]
    global_optimizer.apply_gradients(
        zip(zero_grads_temp, model.trainable_variables))
    model = run_trial_with_physics(model, X_train, y_train, X_val, y_val)
    best_models.append(model)

# 7) Save the final model
best_models[-1].save('best_pinn_model.h5')
