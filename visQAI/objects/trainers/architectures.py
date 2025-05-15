from tensorflow.keras import layers, models, optimizers, losses, metrics, backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from generic_trainer import GenericTrainer
from keras import layers, models, optimizers, losses, metrics

# ─── 1) Define your architectures ────────────────────────────────────────────


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class ReverseCumsum(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        y_rev = inputs[:, ::-1]
        csum_rev = K.cumsum(y_rev, axis=1)
        return csum_rev[:, ::-1]

    def get_config(self):
        # no extra args, so just delegate
        return super().get_config()


def invert(x):
    # Keras can now serialize/deserialize this by name
    return tf.subtract(1.0, x)


# def build_mlp(input_dim: int,
#               output_dim: int,
#               num_layers: int,
#               units: int,
#               learning_rate: float,
#               activation: str = "relu",
#               dropout_rate: float = 0.0,
#               use_batch_norm: bool = True,
#               use_residual: bool = False,

#               ) -> tf.keras.Model:
#     """
#     Builds a more robust MLP with optional batch-norm, dropout, and residual connections.

#     Args:
#       input_dim: Dimension of input feature vector.
#       output_dim: Dimension of output vector.
#       num_layers: Total number of Dense layers in the body.
#       units: Number of units in each Dense layer.
#       activation: Which activation to use in hidden layers.
#       dropout_rate: Dropout rate (0.0 = no dropout).
#       use_batch_norm: Whether to insert BatchNormalization after each Dense.
#       use_residual: Whether to add a residual skip every two layers.

#     Returns:
#       A compiled tf.keras.Model (without optimizer & loss).
#     """
#     inputs = layers.Input(shape=(input_dim,), name="features")
#     x = inputs
#     prev = None

#     for i in range(num_layers):
#         x = layers.Dense(
#             units,
#             activation=None,
#             kernel_initializer="he_normal",
#             name=f"dense_{i}"
#         )(x)
#         if use_batch_norm:
#             x = layers.BatchNormalization(name=f"bn_{i}")(x)
#         x = layers.Activation(activation, name=f"act_{i}")(x)
#         if dropout_rate > 0:
#             x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
#         if use_residual and i % 2 == 1:
#             if prev is not None and prev.shape[-1] == x.shape[-1]:
#                 x = layers.Add(name=f"residual_{i}")([prev, x])
#         prev = x if i % 2 == 0 else prev

#     outputs = layers.Dense(
#         output_dim,
#         activation="linear",
#         name="outputs"
#     )(x)

#     return models.Model(inputs, outputs, name="mlp_advanced")


# def mlp_compile(hp):
#     return {
#         "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
#         "loss":      losses.MeanSquaredError(),
#         "metrics":   [metrics.MeanSquaredError(name="mse")],
#     }


def build_cnn(input_dim: int, output_dim: int,
              filters: int, kernel_size: int, dense_units: int, learning_rate: float):
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.Reshape((input_dim, 1), name="reshape")(inp)

    # two Conv1D blocks
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

    # instead of direct y, predict raw steps
    raw_steps = layers.Dense(output_dim,
                             activation=None,
                             name="raw_steps")(x)
    pos_steps = layers.Activation("relu",
                                  name="positive_steps")(raw_steps)

    # enforce monotonicity
    monotonic_out = ReverseCumsum(name="monotonic_output")(pos_steps)

    model = models.Model(inputs=inp,
                         outputs=monotonic_out,
                         name="cnn_monotonic")
    return model


cnn_hp_space = {
    "filters":       {"type": "Choice", "values": [16, 32, 64, 128],        "default": 32},
    "kernel_size":   {"type": "Int",    "min": 1, "max": 3, "step": 1,      "default": 2},
    "dense_units":   {"type": "Choice", "values": [16, 32, 64, 128],        "default": 64},
    "learning_rate": {"type": "Float",  "min": 1e-4, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def cnn_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.MeanSquaredError(name="mse")],
    }


def build_transformer(
    input_dim: int,
    output_dim: int,
    emb_dim: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int,
    dropout_rate: float,
    learning_rate: float
) -> tf.keras.Model:
    """
    A simple Transformer encoder for tabular regression:
      • Projects each feature into an emb_dim‐dim token
      • Applies num_layers of (Attention → Add+Norm → FFN → Add+Norm)
      • Pools and linearly projects to output_dim
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    # expand into sequence of length=input_dim, tokens of size 1
    x = layers.Reshape((input_dim, 1))(inputs)
    # project to embedding dimension
    x = layers.Dense(emb_dim, name="embed")(x)

    # Transformer encoder blocks
    for i in range(num_layers):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=emb_dim // num_heads,
            dropout=dropout_rate,
            name=f"mha_{i}"
        )(x, x)
        x = layers.Add(name=f"res_attn_{i}")([x, attn_output])
        x = layers.LayerNormalization(name=f"norm1_{i}")(x)

        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation="relu", name=f"ffn1_{i}")(x)
        ffn = layers.Dropout(dropout_rate, name=f"drop_ffn_{i}")(ffn)
        ffn = layers.Dense(emb_dim, name=f"ffn2_{i}")(ffn)
        x = layers.Add(name=f"res_ffn_{i}")([x, ffn])
        x = layers.LayerNormalization(name=f"norm2_{i}")(x)

    # global pool & output
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)

    model = models.Model(inputs, outputs, name="transformer_regressor")
    return model


# ─── Transformer Hyperparameter Space ─────────────────────────────────────────
transformer_hp_space = {
    "emb_dim":       {"type": "Choice", "values": [32, 64, 128],           "default": 64},
    "num_heads":     {"type": "Choice", "values": [2, 4, 8],               "default": 4},
    "ff_dim":        {"type": "Choice", "values": [64, 128, 256],          "default": 128},
    "num_layers":    {"type": "Int",    "min": 1, "max": 4, "step": 1,     "default": 2},
    "dropout_rate":  {"type": "Float",  "min": 0.0, "max": 0.5, "step": 0.1, "default": 0.1},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}

# ─── Transformer Compile Function ─────────────────────────────────────────────


def transformer_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.MeanSquaredError(name="mse")],
    }


def build_autoencoder(
    input_dim: int,
    output_dim: int,
    latent_dim: int,
    dropout_rate: float,
    learning_rate: float
) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="features")

    # Encoder
    x = layers.Dense(latent_dim * 2, activation="relu",
                     name="enc_dense1")(inputs)
    x = layers.Dropout(dropout_rate, name="enc_dropout")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(latent_dim * 2, activation="relu",
                     name="dec_dense1")(latent)
    x = layers.Dropout(dropout_rate, name="dec_dropout")(x)
    outputs = layers.Dense(
        output_dim, activation="linear", name="reconstruction")(x)

    return models.Model(inputs, outputs, name="autoencoder")


# ─── Autoencoder Hyperparameter Space ─────────────────────────────────────────
autoencoder_hp_space = {
    "latent_dim":    {"type": "Choice", "values": [8, 16, 32, 64],           "default": 32},
    "dropout_rate":  {"type": "Float",  "min": 0.0,  "max": 0.5, "step": 0.1, "default": 0.1},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def autoencoder_compile(hp):
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
    "dropout_rate":  {"type": "Float",  "min": 0.0, "max": 0.5,   "step": 0.05,   "default": 0.0},
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
    # "transfomer": {"builder":    build_transformer, "hp": transformer_hp_space, "compile_fn": transformer_compile, },
    "autoencoder": {
        "builder":    build_autoencoder,
        "hp":         autoencoder_hp_space,
        "compile_fn": autoencoder_compile,
    },
}

# ─── 2) Load data once ─────────────────────────────────────────────────────────
df = pd.read_csv("content/formulation_data_05072025.csv")

# ─── 3) Prepare to collect performance ─────────────────────────────────────────
performance = {}

for name, cfg in architectures.items():
    print(f"\n▶▶▶ Training {name} ◀◀◀")
    out_dir = os.path.join("visQAI", "objects", "architectures", name)
    os.makedirs(out_dir, exist_ok=True)

    trainer = GenericTrainer(
        df,
        builder=cfg["builder"],
        hyperparam_space=cfg["hp"],
        compile_args=cfg["compile_fn"],
        cv_splits=4,
        random_state=42,
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
