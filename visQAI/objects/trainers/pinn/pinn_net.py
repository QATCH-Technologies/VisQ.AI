# tunable_model.py

from typing import Sequence
from keras import Model, layers, regularizers, optimizers
from keras_tuner import HyperModel, HyperParameters

import tensorflow as tf

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


class MLPHyperModel(HyperModel):
    def __init__(self, input_dim, output_dim, feature_names):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_names = feature_names

    def build(self, hp: HyperParameters) -> Model:
        # Physics constraint weights
        w_mon_inc = hp.Float("w_mon_inc", 1e-4, 1e-1,
                             sampling="log", default=1e-2)
        w_mon_dec = hp.Float("w_mon_dec", 1e-4, 1e-1,
                             sampling="log", default=1e-2)
        w_flat = hp.Float("w_flat", 1e-4, 1e-1, sampling="log", default=1e-2)
        w_shear = hp.Float("w_shear_thinning", 1e-4, 1e-1,
                           sampling="log", default=1e-2)
        w_arr = hp.Float("w_arhenius", 1e-4, 1e-1,
                         sampling="log", default=1e-2)
        w_bell = hp.Float("w_pi_bell", 1e-4, 1e-1,
                          sampling="log", default=1e-2)
        w_ein_thresh = hp.Float("w_einstein_threshold",
                                1e-4, 1e-1, sampling="log", default=1e-2)
        w_ein_dil = hp.Float("w_einstein_dilute_limit",
                             1e-4, 1e-1, sampling="log", default=1e-2)
        w_excl = hp.Float("w_exclude_volume_divergence", 1e-4,
                          1e-1, sampling="log", default=1e-2)

        # Architecture hyperparameters
        num_layers = hp.Int("num_layers", 1, 10)
        activation = hp.Choice("activation", ["relu", "tanh", "elu", "sine"])
        l2_reg = hp.Float("l2_reg", 1e-6, 1e-2, sampling="log")
        dropout_rate = hp.Float("dropout_rate", 0.0, 0.5, step=0.1)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

        units_list = [
            hp.Int(f"units_{i}", min_value=16, max_value=256, step=16)
            for i in range(num_layers)
        ]

        # Build constraint objects
        constraints = [
            MonotonicIncreasingConstraint(self.feature_names,
                                          ["Protein_concentration",
                                              "Sugar_concentration"],
                                          weight=w_mon_inc),
            MonotonicDecreasingConstraint(self.feature_names,
                                          ["Temperature", "Surfactant_concentration"],
                                          weight=w_mon_dec),
            FlatSlopeConstraint(self.feature_names,
                                "Buffer_pH", weight=w_flat),
            ShearThinningConstraint(weight=w_shear),
            ArrheniusConstraint(self.feature_names,
                                "Temperature", weight=w_arr),
            GaussianBellAroundPIConstraint(
                self.feature_names, "Buffer_pH", "PI_mean", weight=w_bell),
            EinsteinDiluteLimitConstraint(self.feature_names, "Protein_concentration", threshold=w_ein_thresh,
                                          weight=w_ein_dil),
            ExcludedVolumeDivergenceConstraint(
                self.feature_names, "Protein_concentration", weight=w_excl),
        ]

        # Input
        inp = layers.Input(shape=(self.input_dim,))
        x = inp

        # Residual MLP blocks with projection
        for i, units in enumerate(units_list):
            prev = x
            x = layers.Dense(
                units, kernel_regularizer=regularizers.l2(l2_reg))(x)
            if activation == "sine":
                x = tf.sin(x)
            else:
                x = layers.Activation(activation)(x)
            x = layers.Dropout(dropout_rate)(x)

            # Every 2 layers, project prev and add
            if i % 2 == 1:
                proj = layers.Dense(units, use_bias=False)(prev)
                x = layers.Add()([x, proj])

        # Final output layer: linear, then enforce non-negativity via softplus
        out = layers.Dense(self.output_dim)(x)
        out = tf.nn.softplus(out)

        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        # Attach physics constraints
        model._physics_constraints = constraints
        return model
