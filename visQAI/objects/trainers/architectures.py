import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from generic_trainer import GenericTrainer
from keras import layers, models, optimizers, losses, metrics

# ─── 1) Define your architectures ────────────────────────────────────────────


def invert(x):
    # Keras can now serialize/deserialize this by name
    return tf.subtract(1.0, x)


def build_mlp(input_dim: int, output_dim: int,
              num_layers: int, units: int, learning_rate: float):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inputs, outputs)


mlp_hp_space = {
    "num_layers":    {"type": "Int",    "min": 1, "max": 5,   "step": 1,                      "default": 2},
    "units":         {"type": "Choice", "values": [16, 32, 64, 128],         "default": 64},
    "learning_rate": {"type": "Float",  "min": 1e-4, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def mlp_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }

# e.g. you might have a CNN builder too:


def build_cnn(input_dim: int, output_dim: int,
              filters: int, kernel_size: int, dense_units: int, learning_rate: float):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Conv1D(filters, kernel_size,
                      activation="relu", padding="same")(x)
    x = layers.Conv1D(filters, kernel_size,
                      activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inp, out)


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
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── 1) MLP with Dropout ───────────────────────────────────────────────────────
def build_mlp_dropout(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    units: int,
    dropout_rate: float,
    learning_rate: float
):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inputs, outputs)


mlp_dropout_hp_space = {
    "num_layers":    {"type": "Int",    "min": 1, "max": 5, "step": 1,                   "default": 2},
    "units":         {"type": "Choice", "values": [16, 32, 64, 128],                      "default": 64},
    "dropout_rate":  {"type": "Float",  "min": 0.1, "max": 0.5, "step": 0.1,               "default": 0.2},
    "learning_rate": {"type": "Float",  "min": 1e-4, "max": 1e-2, "sampling": "log",       "default": 1e-3},
}


def mlp_dropout_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── 2) Wide & Deep ────────────────────────────────────────────────────────────
def build_wide_and_deep(
    input_dim: int,
    output_dim: int,
    deep_layers: int,
    deep_units: int,
    learning_rate: float
):
    inputs = layers.Input(shape=(input_dim,), name="features")
    # wide path: a single linear layer
    wide = layers.Dense(output_dim, activation="linear")(inputs)

    # deep path:
    x = inputs
    for _ in range(deep_layers):
        x = layers.Dense(deep_units, activation="relu")(x)
    deep_out = layers.Dense(output_dim, activation="linear")(x)

    # combine
    combined = layers.Add()([wide, deep_out])
    return models.Model(inputs, combined, name="wide_and_deep")


wide_deep_hp_space = {
    "deep_layers":   {"type": "Int",    "min": 1, "max": 4, "step": 1,                 "default": 2},
    "deep_units":    {"type": "Choice", "values": [32, 64, 128],                      "default": 64},
    "learning_rate": {"type": "Float",  "min": 1e-4, "max": 1e-2, "sampling": "log",     "default": 1e-3},
}


def wide_deep_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }

# ─── New MLP Variant 1: BatchNorm + Dropout ─────────────────────────────────


def build_mlp_bn_dropout(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    units: int,
    dropout_rate: float,
    learning_rate: float
):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        x = layers.Dense(units, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inputs, outputs, name="mlp_bn_dropout")


mlp_bn_dropout_hp_space = {
    "num_layers":    {"type": "Int",    "min": 1, "max": 5,   "step": 1,     "default": 3},
    "units":         {"type": "Choice", "values": [32, 64, 128, 256],         "default": 128},
    "dropout_rate":  {"type": "Float",  "min": 0.1, "max": 0.5, "step": 0.1, "default": 0.3},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def mlp_bn_dropout_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── New MLP Variant 2: Residual MLP ─────────────────────────────────────────
def build_mlp_residual(
    input_dim: int,
    output_dim: int,
    num_blocks: int,
    units: int,
    learning_rate: float
):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(units, activation="relu")(inputs)
    for _ in range(num_blocks):
        shortcut = x
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dense(units, activation=None)(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inputs, outputs, name="mlp_residual")


mlp_residual_hp_space = {
    "num_blocks":    {"type": "Int",    "min": 1, "max": 4,   "step": 1,     "default": 2},
    "units":         {"type": "Choice", "values": [32, 64, 128, 256],         "default": 128},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def mlp_residual_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── 1) Highway MLP ───────────────────────────────────────────────────────────


def build_highway_mlp(
    input_dim: int,
    output_dim: int,
    num_blocks: int,
    units: int,
    learning_rate: float
) -> tf.keras.Model:
    """
    A highway network: each block learns H(x) and gate T(x), then
    y = T * H + (1−T) * x (with x projected to the same dim).
    """
    inputs = layers.Input(shape=(input_dim,), name="features")

    # Step 0: project to highway dimension
    x = layers.Dense(units, activation="relu", name="init_projection")(inputs)

    for i in range(num_blocks):
        H = layers.Dense(units, activation="relu", name=f"H_{i}")(x)
        T = layers.Dense(
            units,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-1),
            name=f"T_{i}"
        )(x)

        invT = layers.Lambda(invert, name=f"invT_{i}")(T)

        carry = layers.Multiply(name=f"carry_{i}")([invT, x])
        transform = layers.Multiply(name=f"transform_{i}")([T,    H])
        x = layers.Add(name=f"highway_out_{i}")([carry, transform])

    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    return models.Model(inputs, outputs, name="highway_mlp")


highway_hp_space = {
    "num_blocks":    {"type": "Int",    "min": 1, "max": 4, "step": 1,               "default": 2},
    "units":         {"type": "Choice", "values": [32, 64, 128, 256],               "default": 128},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def highway_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── 2) Dense‐Concatenation MLP ─────────────────────────────────────────────────
def build_dense_concat_mlp(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    units: int,
    learning_rate: float
) -> tf.keras.Model:
    """
    Each layer’s output is concatenated with its input, forming a dense‐block
    style MLP that grows width at each step.
    """
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for _ in range(num_layers):
        y = layers.Dense(units, activation="relu")(x)
        x = layers.Concatenate()([x, y])
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inputs, outputs, name="dense_concat_mlp")


dense_concat_hp_space = {
    "num_layers":    {"type": "Int",    "min": 1, "max": 5, "step": 1,               "default": 3},
    "units":         {"type": "Choice", "values": [32, 64, 128],                   "default": 64},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def dense_concat_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── 3) SIREN‐Style MLP (sin activations) ─────────────────────────────────────
def build_siren_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: int,
    hidden_units: int,
    w0: float,
    learning_rate: float
) -> tf.keras.Model:
    """
    A SIREN-style MLP that uses sin(w0 * x) activations to capture high-frequency
    variations.
    """
    def sine(x): return tf.sin(w0 * x)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_units)(inputs)
    x = layers.Lambda(sine)(x)
    for _ in range(hidden_layers - 1):
        x = layers.Dense(hidden_units)(x)
        x = layers.Lambda(sine)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    return models.Model(inputs, outputs, name="siren_mlp")


siren_hp_space = {
    "hidden_layers": {"type": "Int",    "min": 1, "max": 4, "step": 1,               "default": 2},
    "hidden_units":  {"type": "Choice", "values": [32, 64, 128, 256],               "default": 64},
    "w0":            {"type": "Float",  "min": 1.0, "max": 30.0, "sampling": "log", "default": 30.0},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-4},
}


def siren_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


def build_ann(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    units: int,
    activation: str,
    learning_rate: float
) -> models.Model:
    """
    A simple feed-forward ANN: num_layers of Dense(units, activation), then
    a linear output layer.
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(units, activation=activation, name=f"dense_{i}")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    return models.Model(inputs, outputs, name="ann")


# ─── 2) Hyperparameter space ──────────────────────────────────────────────────
ann_hp_space = {
    "num_layers":    {"type": "Int",    "min": 1, "max": 5,   "step": 1,               "default": 2},
    "units":         {"type": "Choice", "values": [16, 32, 64, 128, 256],             "default": 64},
    "activation":    {"type": "Choice", "values": ["relu", "tanh", "sigmoid"],      "default": "relu"},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}

# ─── 3) Compile function ───────────────────────────────────────────────────────


def ann_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
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
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── New Variant: Highway MLP + LayerNorm + Dropout ──────────────────────────


def build_highway_bn_dropout(
    input_dim: int,
    output_dim: int,
    num_blocks: int,
    units: int,
    dropout_rate: float,
    learning_rate: float
) -> models.Model:
    """
    Highway blocks with LayerNorm, ReLU and Dropout:
      x0 = Dense(units) → LayerNorm → ReLU
      for each block:
        H = Dense(units) → LayerNorm → ReLU → Dropout
        T = Dense(units, sigmoid, bias=-1)
        x = T * H + (1 - T) * x
        x = LayerNorm(x)
      output = Dense(output_dim)
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    # initial projection
    x = layers.Dense(units, activation=None, name="init_proj")(inputs)
    x = layers.LayerNormalization(name="init_ln")(x)
    x = layers.Activation("relu", name="init_act")(x)

    for i in range(num_blocks):
        # transform branch
        H = layers.Dense(units, activation=None, name=f"H_{i}")(x)
        H = layers.LayerNormalization(name=f"H_ln_{i}")(H)
        H = layers.Activation("relu", name=f"H_act_{i}")(H)
        H = layers.Dropout(dropout_rate, name=f"H_drop_{i}")(H)
        # gate
        T = layers.Dense(
            units,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-1),
            name=f"T_{i}"
        )(x)
        # carry = (1 - T) * x
        invT = layers.Lambda(invert, name=f"invT_{i}")(T)
        carry = layers.Multiply(name=f"carry_{i}")([invT, x])
        transform = layers.Multiply(name=f"transform_{i}")([T, H])
        # combine + norm
        x = layers.Add(name=f"highway_out_{i}")([carry, transform])
        x = layers.LayerNormalization(name=f"out_ln_{i}")(x)

    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    return models.Model(inputs, outputs, name="highway_bn_dropout")


highway_bn_dropout_hp_space = {
    "num_blocks":    {"type": "Int",    "min": 1, "max": 4,   "step": 1,     "default": 2},
    "units":         {"type": "Choice", "values": [64, 128, 256],               "default": 128},
    "dropout_rate":  {"type": "Float",  "min": 0.1, "max": 0.5, "step": 0.1, "default": 0.3},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def highway_bn_dropout_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── New Architecture: Mixture-of-Experts MLP ─────────────────────────────────

def expand_dims_last(x):
    return tf.expand_dims(x, -1)


def sum_over_experts(x):
    # Keras will record "sum_over_experts" in the config and be able to reload it
    return tf.reduce_sum(x, axis=1)


def build_moe_mlp(
    input_dim: int,
    output_dim: int,
    num_experts: int,
    num_layers: int,
    expert_units: int,
    gating_units: int,
    learning_rate: float
) -> tf.keras.Model:
    """
    A Mixture-of-Experts for regression:
      • A small gating MLP predicts a softmax over experts
      • Each expert is its own MLP stack
      • Final output is weighted sum of expert outputs by gate scores
    """
    inputs = layers.Input(shape=(input_dim,), name="features")

    # Gating network
    g = layers.Dense(gating_units, activation="relu", name="gate_hid")(inputs)
    gates = layers.Dense(num_experts, activation="softmax", name="gate_out")(g)
    # expand gates for broadcast
    gates_exp = layers.Lambda(
        expand_dims_last, name="gates_exp"
    )(gates)
    # Expert networks
    expert_outputs = []
    for i in range(num_experts):
        x = inputs
        for j in range(num_layers):
            x = layers.Dense(expert_units, activation="relu",
                             name=f"expert{i}_hid{j}")(x)
        out = layers.Dense(output_dim, activation="linear",
                           name=f"expert{i}_out")(x)
        expert_outputs.append(out)

    reshaped_experts = [
        layers.Reshape((1, output_dim), name=f"exp{i}_expanddims")(out)
        for i, out in enumerate(expert_outputs)
    ]
    stacked = layers.Concatenate(
        axis=1, name="stack_experts")(reshaped_experts)

    # Weighted sum: apply gates and sum over experts axis
    weighted = layers.Multiply(name="weighted_experts")([stacked, gates_exp])
    outputs = layers.Lambda(sum_over_experts, name="moe_output")(weighted)

    model = models.Model(inputs, outputs, name="moe_mlp")
    return model


moe_hp_space = {
    "num_experts":   {"type": "Choice", "values": [2, 3, 4],                   "default": 2},
    "num_layers":    {"type": "Int",    "min": 1, "max": 3, "step": 1,          "default": 2},
    "expert_units":  {"type": "Choice", "values": [32, 64, 128],                "default": 64},
    "gating_units":  {"type": "Choice", "values": [16, 32, 64],                 "default": 32},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def moe_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── architectures dict ───────────────────────────────────────────────────────
architectures = {
    "mlp":         {"builder": build_mlp,           "hp": mlp_hp_space,         "compile_fn": mlp_compile},
    "cnn":         {"builder": build_cnn,           "hp": cnn_hp_space,         "compile_fn": cnn_compile},
    "mlp_dropout": {"builder": build_mlp_dropout,   "hp": mlp_dropout_hp_space, "compile_fn": mlp_dropout_compile},
    "wide_deep":   {"builder": build_wide_and_deep,  "hp": wide_deep_hp_space,    "compile_fn": wide_deep_compile},
    "mlp_bn_dropout": {"builder": build_mlp_bn_dropout, "hp": mlp_bn_dropout_hp_space, "compile_fn": mlp_bn_dropout_compile},
    "mlp_residual": {"builder": build_mlp_residual, "hp": mlp_residual_hp_space, "compile_fn": mlp_residual_compile},
    "highway_mlp":       {"builder": build_highway_mlp,     "hp": highway_hp_space,      "compile_fn": highway_compile},
    "dense_concat_mlp":  {"builder": build_dense_concat_mlp, "hp": dense_concat_hp_space, "compile_fn": dense_concat_compile},
    "siren_mlp":         {"builder": build_siren_mlp,       "hp": siren_hp_space,        "compile_fn": siren_compile},
    "transfomer": {"builder":    build_transformer, "hp":         transformer_hp_space, "compile_fn": transformer_compile, },
    "ann": {"builder": build_ann, "hp": ann_hp_space, "compile_fn": ann_compile},
    "highway_bn_droupout": {"builder":    build_highway_bn_dropout, "hp":         highway_bn_dropout_hp_space, "compile_fn": highway_bn_dropout_compile, },
    "moe_mlp": {"builder":    build_moe_mlp, "hp":         moe_hp_space, "compile_fn": moe_compile, }
}

# ─── 1) Residual Mixture-of-Experts ────────────────────────────────────────────


def build_residual_moe(
    input_dim: int,
    output_dim: int,
    num_experts: int,
    expert_units: int,
    num_layers: int,
    residual_units: int,
    learning_rate: float
) -> tf.keras.Model:
    """
    A Mixture-of-Experts where each expert is a small residual block stack,
    and a gating network dynamically mixes their outputs.
    """
    inputs = layers.Input(shape=(input_dim,), name="features")

    # Gating network
    g = layers.Dense(residual_units, activation="relu",
                     name="gate_hid")(inputs)
    gates = layers.Dense(num_experts, activation="softmax", name="gate_out")(g)
    gates_exp = layers.Reshape((num_experts, 1), name="gates_exp")(gates)

    # Build each expert as a Residual MLP
    expert_outputs = []
    for i in range(num_experts):
        x = layers.Dense(expert_units, activation="relu",
                         name=f"exp{i}_in")(inputs)
        for j in range(num_layers):
            skip = x
            x = layers.Dense(expert_units, activation="relu",
                             name=f"exp{i}_l{j}_1")(x)
            x = layers.Dense(expert_units, activation=None,
                             name=f"exp{i}_l{j}_2")(x)
            x = layers.Add(name=f"exp{i}_res{j}")([x, skip])
            x = layers.Activation("relu", name=f"exp{i}_act{j}")(x)
        out = layers.Dense(output_dim, activation="linear",
                           name=f"exp{i}_out")(x)
        expert_outputs.append(out)

    reshaped_experts = [
        layers.Reshape((1, output_dim), name=f"exp{i}_expand")(out)
        for i, out in enumerate(expert_outputs)
    ]

    stacked = layers.Concatenate(
        axis=1, name="stack_experts")(reshaped_experts)

    weighted = layers.Multiply(name="weight_experts")([stacked, gates_exp])

    outputs = layers.Lambda(
        sum_over_experts, name="moe_residual_out")(weighted)
    return models.Model(inputs, outputs, name="residual_moe")


residual_moe_hp_space = {
    "num_experts":     {"type": "Choice", "values": [2, 3, 4],               "default": 2},
    "expert_units":    {"type": "Choice", "values": [64, 128, 256],           "default": 128},
    "num_layers":      {"type": "Int",    "min": 1, "max": 3, "step": 1,      "default": 2},
    "residual_units":  {"type": "Choice", "values": [32, 64],                "default": 32},
    "learning_rate":   {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log", "default": 1e-3},
}


def residual_moe_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }

# ─── 2) Squeeze-and-Excitation MLP ─────────────────────────────────────────────


def build_se_mlp(
    input_dim: int,
    output_dim: int,
    num_blocks: int,
    units: int,
    se_ratio: float,
    learning_rate: float
) -> tf.keras.Model:
    """
    A classic MLP stack where each block ends with a Squeeze-and-Excitation
    channel gating to reweight features.
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = layers.Dense(units, activation="relu", name="in_dense")(inputs)

    for i in range(num_blocks):
        x = layers.Dense(units, activation="relu", name=f"block{i}_dense")(x)
        # Squeeze: global pooling over features
        se = layers.Reshape((units, 1), name=f"block{i}_squeeze")(x)
        se = layers.GlobalAveragePooling1D(name=f"block{i}_gap")(se)

        # 2) Excitation
        se = layers.Dense(
            max(1, int(units * se_ratio)),
            activation="relu",
            name=f"block{i}_exc1"
        )(se)
        se = layers.Dense(units, activation="sigmoid",
                          name=f"block{i}_exc2")(se)
        # expand back to (batch, units, 1)
        se = layers.Reshape((units, 1), name=f"block{i}_scale")(se)

        # 3) Apply gating: expand x too
        x_exp = layers.Reshape((units, 1), name=f"block{i}_expand_x")(x)
        x = layers.Multiply(name=f"block{i}_scale_mul")([x_exp, se])

        # 4) Collapse back to (batch, units) for next block
        x = layers.Reshape((units,), name=f"block{i}_reshape")(x)

    outputs = layers.Dense(output_dim, activation="linear", name="output")(x)
    return models.Model(inputs, outputs, name="se_mlp")


se_mlp_hp_space = {
    "num_blocks":    {"type": "Int",    "min": 1, "max": 4,   "step": 1,     "default": 2},
    "units":         {"type": "Choice", "values": [64, 128, 256],             "default": 128},
    "se_ratio":      {"type": "Float",  "min": 0.05, "max": 0.5, "sampling": "linear", "default": 0.25},
    "learning_rate": {"type": "Float",  "min": 1e-5, "max": 1e-2, "sampling": "log",   "default": 1e-3},
}


def se_mlp_compile(hp):
    return {
        "optimizer": optimizers.Adam(learning_rate=hp["learning_rate"]),
        "loss":      losses.MeanSquaredError(),
        "metrics":   [metrics.RootMeanSquaredError(name="rmse")],
    }


# ─── Inject into architectures dict ────────────────────────────────────────────
architectures.update({
    "residual_moe": {"builder": build_residual_moe, "hp": residual_moe_hp_space, "compile_fn": residual_moe_compile},
    "se_mlp":       {"builder": build_se_mlp,       "hp": se_mlp_hp_space,       "compile_fn": se_mlp_compile},
})

# ─── 2) Load data once ─────────────────────────────────────────────────────────
df = pd.read_csv("content/formulation_data_05062025.csv")

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
        max_trials=30,
        executions_per_trial=1,
        epochs=15,
        batch_size=16,
        validation_split=0.2,
        directory=out_dir,
        project_name=f"{name}_tuner",
        objective_name="val_rmse",
    )

    # b) Cross-validate & collect
    rmses = trainer.cross_validate(epochs=50, batch_size=16)
    print(f"[{name}] CV RMSEs:", [f"{r:.4f}" for r in rmses])
    performance[name] = rmses

    # c) Save artifacts
    trainer.save(out_dir)
    print(f"[{name}] done; saved to {out_dir}")

# ─── 4) Build comparison report ────────────────────────────────────────────────

# a) assemble DataFrame
rows = []
for model_name, rmses in performance.items():
    rows.append({
        "model":     model_name,
        "mean_rmse": np.mean(rmses),
        "std_rmse":  np.std(rmses),
    })
df_report = pd.DataFrame(rows).sort_values("mean_rmse")

# b) print table
print("\n=== Model Comparison Report ===")
print(df_report.to_markdown(index=False, floatfmt=".4f"))

# c) bar chart of mean RMSE
plt.figure()
plt.bar(df_report["model"], df_report["mean_rmse"])
plt.ylabel("Mean CV RMSE")
plt.title("Model Comparison")
plt.tight_layout()
plt.show()
