import os
import numpy as np
from keras import layers
import tensorflow as tf
from keras.callbacks import EarlyStopping


class Trainer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int = None,
        learning_rate: float = 1e-3,
        layers_config: list[int] = [128, 64],
        patience: int = 5,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.layers_config = layers_config
        self.patience = patience
        self.model: tf.keras.Model | None = None

    def _build_model(self, dropout_rate=0.1) -> tf.keras.Model:
        model = tf.keras.Sequential()
        model.add(layers.InputLayer((self.input_dim,)))
        for units in self.layers_config:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.output_dim))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mse",
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray, save_dir: str):
        y_arr = np.array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        assert y_arr.shape[1] == 5, "Expected 5 targets but got %d" % y_arr.shape[1]
        self.output_dim = y_arr.shape[1]
        self.model = self._build_model()
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        self.model.fit(
            X,
            y_arr,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=2,
        )

        os.makedirs(save_dir, exist_ok=True)
        self.model.export(save_dir)
        print(f"Model exported to {save_dir}")
