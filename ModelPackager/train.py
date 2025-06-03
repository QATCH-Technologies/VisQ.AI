# train_regression_mlp.py

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from data_processor import DataProcessor
from transformer import ScalerPipeline


def build_regression_mlp(input_dim: int, output_dim: int) -> tf.keras.Model:
    """
    Returns a compiled Keras Sequential model for multi‐output regression.
    """
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(output_dim, activation="linear"),  # linear for regression
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


if __name__ == "__main__":
    import pickle
    import joblib
    from pathlib import Path

    # 1) Paths to your single CSV (features+targets)
    CSV_PATH = "all_data.csv"   # must contain columns: ID, features..., Viscosity_*

    # 2) Instantiate DataProcessor and split X,y
    dp = DataProcessor(config={
        # e.g. ["Protein_type"] if you want to one-hot encode separately
        "drop_columns": [],
        "feature_columns": None,   # None → automatically pick all non-Viscosity_* columns
        "target_prefix": "Viscosity_"
    })
    X_df, y_df = dp.process(CSV_PATH, None)

    # 3) Fit transformer on X_df
    transformer = ScalerPipeline()
    X_scaled = transformer.fit_transform(X_df, y_df)

    # 4) Build & compile the MLP with input_dim = #features, output_dim = #viscosity targets
    input_dim = X_scaled.shape[1]
    output_dim = y_df.shape[1]       # e.g. 5
    mlp = build_regression_mlp(input_dim=input_dim, output_dim=output_dim)

    # 5) Train with early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    mlp.fit(
        X_scaled.values,
        y_df.values,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    # 6) Save everything:
    #    - dataprocessor.pkl
    #    - preprocessor.pkl
    #    - model.h5
    #    - predictor.pkl  (see next section)
    out_dir = Path("pipeline_components")
    out_dir.mkdir(exist_ok=True)

    # 6a) Save the Keras model as model.h5
    model_path = out_dir / "model.h5"
    mlp.save(str(model_path))

    # 6b) Serialize DataProcessor
    with open(out_dir / "dataprocessor.pkl", "wb") as f_dp:
        pickle.dump(dp, f_dp)

    # 6c) Serialize Transformer
    joblib.dump(transformer, str(out_dir / "preprocessor.pkl"))

    print("Saved dataprocessor.pkl, preprocessor.pkl, model.h5 in", out_dir)

    # 6d) Create & save a PredictorWrapper that knows how to do regression
    from predictor import PredictorWrapper
    pw = PredictorWrapper()
    predictor_path = out_dir / "predictor.pkl"
    with open(predictor_path, "wb") as f_pw:
        pickle.dump(pw, f_pw)

    print("Wrote predictor.pkl in", out_dir)
