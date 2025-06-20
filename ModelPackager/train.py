
import os
import numpy as np
from keras import layers, models
import tensorflow as tf
from keras.callbacks import EarlyStopping


class Trainer:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 1e-3, layers_config: list = [128, 64], patience: int = 5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.layers_config = layers_config
        self.patience = patience
        self.model = None

    def _build_model(self, dropout_rate=0.1):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer((self.input_dim,)))
        for units in self.layers_config:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.output_dim))  # final regression output
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self, X: np.ndarray, y: np.ndarray, save_dir: str):
        # 1) Build the model
        input(X.shape[1])
        self.model = self._build_model()

        # 2) Fit with EarlyStopping
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model.fit(
            X,
            y,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=2
        )

        # 3) Export SavedModel into save_dir
        #    (Keras 3+ API: use .export; if you’re on TF2.x/Keras2.x, use tf.saved_model.save)
        os.makedirs(save_dir, exist_ok=True)
        self.model.export(save_dir)
        print(f"Model exported to {save_dir}")
