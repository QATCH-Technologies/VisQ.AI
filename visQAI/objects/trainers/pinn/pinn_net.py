# tunable_model.py

import keras_tuner as kt
from typing import Sequence
from keras import Model, layers, regularizers, optimizers


class MLPHyperModel(kt.HyperModel):
    """
    A HyperModel for a small MLP regressor, so you can tune:
      - number of hidden layers (1-3)
      - units per layer (16-128)
      - activation (relu/tanh)
      - L2 strength
      - dropout rate
      - learning rate
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, hp: kt.HyperParameters) -> Model:
        num_layers = hp.Int("num_layers", 1, 10)
        activation = hp.Choice("activation", ["relu", "tanh", "elu"])
        l2_reg = hp.Float("l2_reg", 1e-6, 1e-2, sampling="log")
        dropout_rate = hp.Float("dropout_rate", 0.0, 1.0, step=0.1)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

        units_list = [
            hp.Int(f"units_{i}", min_value=16, max_value=128, step=16)
            for i in range(num_layers)
        ]

        inp = layers.Input(shape=(self.input_dim,))
        x = inp
        for units in units_list:
            x = layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
            )(x)
            x = layers.Dropout(dropout_rate)(x)

        out = layers.Dense(self.output_dim, activation="linear")(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model
