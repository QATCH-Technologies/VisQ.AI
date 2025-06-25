from tensorflow.keras import layers, models, optimizers, losses, metrics, backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from generic_trainer import GenericTrainer
from keras import layers, models, optimizers, losses, metrics
from custom_layers import ReverseCumsum
# ─── 1) Define your architectures ────────────────────────────────────────────


def monotonic_penalty(y_true, y_pred):
    diff = y_pred[:, :-1] - y_pred[:, 1:]
    violations = tf.nn.relu(-diff)
    return tf.reduce_mean(violations)


def combined_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    mono_loss = monotonic_penalty(y_true, y_pred)
    return mse_loss + 0.1 * mono_loss


def build_cnn(input_dim: int,
              output_dim: int,
              filters: int,
              kernel_size: int,
              dense_units: int,
              learning_rate: float,
              dropout_rate: float):
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.Reshape((input_dim, 1), name="reshape")(inp)

    # two Conv1D blocks (no dropout here)
    x = layers.Conv1D(filters, kernel_size,
                      activation="relu",
                      padding="same",
                      name="conv1")(x)
    x = layers.Conv1D(filters, kernel_size,
                      activation="relu",
                      padding="same",
                      name="conv2")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units,
                     activation="relu",
                     name="dense_hidden")(x)

    # light dropout only on the dense representation
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)

    # raw‐step outputs
    raw_steps = layers.Dense(output_dim,
                             activation=None,
                             name="raw_steps")(x)
    pos_steps = layers.Activation("relu",
                                  name="positive_steps")(raw_steps)

    # enforce monotonicity
    monotonic_out = ReverseCumsum(name="monotonic_output")(pos_steps)

    return models.Model(inputs=inp,
                        outputs=monotonic_out,
                        name="cnn_monotonic_light_dropout")


# expand your hyperparameter space to include dropout_rate
cnn_hp_space = {
    "filters":       {"type": "Choice", "values": [16, 32, 64, 128],        "default": 32},
    "kernel_size":   {"type": "Int",    "min": 1, "max": 3, "step": 1,      "default": 2},
    "dense_units":   {"type": "Choice", "values": [16, 32, 64, 128],        "default": 64},
    "dropout_rate":  {"type": "Float",  "min": 0.001, "max": 0.1, "sampling": "linear", "default": 0.5},
    "learning_rate": {"type": "Float",  "min": 1e-4, "max": 1e-2, "sampling": "log",   "default": 1e-3},
}


def cnn_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.MeanSquaredError(name="mse")],
    }


def build_mlp(input_dim: int,
              output_dim: int,
              num_layers: int,
              units: int,
              learning_rate: float,
              activation: str = "relu",
              dropout_rate: float = 0.0,
              use_batch_norm: bool = True,
              use_residual: bool = False,
              ) -> tf.keras.Model:
    """
    Builds a robust MLP whose output vector is guaranteed non-increasing:
      y_1 >= y_2 >= ... >= y_output_dim
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    prev = None

    # hidden body
    for i in range(num_layers):
        x = layers.Dense(units,
                         activation=None,
                         kernel_initializer="he_normal",
                         name=f"dense_{i}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation(activation, name=f"act_{i}")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
        if use_residual and i % 2 == 1 and prev is not None:
            if prev.shape[-1] == x.shape[-1]:
                x = layers.Add(name=f"residual_{i}")([prev, x])
        if i % 2 == 0:
            prev = x

    # predict raw steps
    raw_steps = layers.Dense(output_dim,
                             activation=None,
                             name="raw_steps")(x)
    # enforce non-negativity
    positive_steps = layers.Activation("relu",
                                       name="positive_steps")(raw_steps)

    monotonic_output = ReverseCumsum(name="monotonic_output")(positive_steps)

    model = models.Model(
        inputs=inputs, outputs=monotonic_output, name="mlp_monotonic")
    return model


mlp_hp_space = {
    "num_layers":    {"type": "Int",    "min": 1,  "max": 10,   "step": 1,      "default": 2},
    "units":         {"type": "Choice", "values": [16, 32, 64, 128, 256, 512],  "default": 64},
    "activation":    {"type": "Choice", "values": ["relu", "elu", "tanh"],       "default": "relu"},
    "dropout_rate":  {"type": "Float",  "min": 0, "max": 0.6,   "step": 0.05,   "default": 0.0},
    "use_batch_norm": {"type": "Boolean",                                           "default": True},
    "use_residual":  {"type": "Boolean",                                           "default": False},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def mlp_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.MeanSquaredError(name="mse")],
    }


# ─── architectures dict ───────────────────────────────────────────────────────
architectures = {
    "mlp": {"builder": build_mlp, "hp": mlp_hp_space, "compile_fn": mlp_compile},
    "cnn": {"builder": build_cnn, "hp": cnn_hp_space, "compile_fn": cnn_compile},
}

# ─── 2) Load data once ─────────────────────────────────────────────────────────
df = pd.read_csv("content/train_features.csv")

# ─── 3) Prepare to collect performance ─────────────────────────────────────────
performance = {}

for name, cfg in architectures.items():
    for member in range(5):
        import random
        state = random.randint(1, 100)
        print(f"\n>>> Training {name} with state {state} <<<")
        out_dir = os.path.join("visQAI", "objects",
                               "architectures", f"{name}", f"member_{member}")
        os.makedirs(out_dir, exist_ok=True)

        trainer = GenericTrainer(
            df,
            builder=cfg["builder"],
            hyperparam_space=cfg["hp"],
            compile_args=cfg["compile_fn"],
            cv_splits=4,
            random_state=state,
        )
        # a) Hyperparameter search
        trainer.tune(
            max_trials=50,
            executions_per_trial=1,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            directory=out_dir,
            project_name=f"{name}_tuner",
            objective_name="val_mse",
        )

        # b) Cross-validate & collect
        mses = trainer.cross_validate(epochs=50, batch_size=16)
        print(f"[{name}] CV MSEs:", [f"{e:.4f}" for e in mses])
        performance[name] = mses

        # c) Save artifacts
        trainer.save(out_dir)
        print(f"[{name}] done; saved to {out_dir}")

# ─── 4) Build comparison report ────────────────────────────────────────────────

# a) assemble DataFrame
rows = []
for model_name, mses in performance.items():
    rows.append({
        "model":     model_name,
        "mean_mse": np.mean(mses),
        "std_mse":  np.std(mses),
    })
df_report = pd.DataFrame(rows).sort_values("mean_mse")

print("\n=== Model Comparison Report ===")
print(df_report.to_markdown(index=False, floatfmt=".4f"))


plt.figure()
plt.bar(df_report["model"], df_report["mean_mse"])
plt.ylabel("Mean CV MSE")
plt.title("Model Comparison")
plt.tight_layout()
plt.show()
