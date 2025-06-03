# build_and_package.py

import os
import zipfile
import pickle
import joblib

import pandas as pd
from keras.callbacks import EarlyStopping

# 1) Imports from your modules (make sure these filenames match your actual files)
from data_processor import DataProcessor
from transformer import ScalerPipeline
from train import build_regression_mlp
from predictor import PredictorWrapper


def train_and_serialize(
    train_csv_path: str,
    output_dir: str = "pipeline_components"
) -> str:
    """
    1) Read a single CSV that contains features + Viscosity_* targets.
    2) Split out (X_df, y_df) via DataProcessor.process(...)
    3) Scale X_df via ScalerPipeline.fit_transform(...)
    4) Build & train a multi-output MLP (one neuron per Viscosity_*).
    5) Serialize:
         - dataprocessor.pkl
         - preprocessor.pkl
         - model.h5
         - predictor.pkl
    Returns the path to `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- STEP A: Use DataProcessor to split features vs. viscosity targets
    dp = DataProcessor(config={
        "drop_columns": [],        # if you have extra columns to drop, list them here
        "feature_columns": None,   # None => let DataProcessor infer all non-Viscosity_* columns
        "target_prefix": "Viscosity_"
    })
    X_df, y_df = dp.process(train_csv_path, None)
    # X_df: DataFrame of shape (n_samples, n_features)
    # y_df: DataFrame of shape (n_samples, n_viscosity_targets)

    # --- STEP B: Fit the ScalerPipeline on X_df
    transformer = ScalerPipeline()
    X_scaled = transformer.fit_transform(X_df, y_df)
    # X_scaled: DataFrame of same shape as X_df but scaled

    # --- STEP C: Build & Train the multi-output regression MLP
    input_dim = X_scaled.shape[1]
    output_dim = y_df.shape[1]  # number of Visosity_* columns

    mlp = build_regression_mlp(input_dim=input_dim, output_dim=output_dim)

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

    # --- STEP D: Save the trained model as model.h5
    model_path = os.path.join(output_dir, "model.h5")
    mlp.save(model_path)

    # --- STEP E: Serialize DataProcessor & Transformer
    dp_path = os.path.join(output_dir, "dataprocessor.pkl")
    with open(dp_path, "wb") as f_dp:
        pickle.dump(dp, f_dp)

    transformer_path = os.path.join(output_dir, "preprocessor.pkl")
    joblib.dump(transformer, transformer_path)

    # --- STEP F: Create & Serialize PredictorWrapper
    pw = PredictorWrapper()
    predictor_path = os.path.join(output_dir, "predictor.pkl")
    with open(predictor_path, "wb") as f_pw:
        pickle.dump(pw, f_pw)

    print(f"Serialized files in '{output_dir}':")
    for fname in sorted(os.listdir(output_dir)):
        print("  ", fname)

    return output_dir


def package_components(
    components_dir: str,
    zip_name: str = "pipeline_bundle.zip"
) -> None:
    """
    Zip up everything under components_dir into zip_name.
    """
    with zipfile.ZipFile(zip_name, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(components_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # We want the files to be at the root of the ZIP, so use arcname=file
                zf.write(full_path, arcname=file)
    print(f"Packaged '{components_dir}' into '{zip_name}'")


if __name__ == "__main__":
    # If you have a single CSV called "all_data.csv" with headers:
    # ID,Protein_type,MW,PI_mean,PI_range,Protein_conc,Temperature,...,Viscosity_100,...,Viscosity_15000000
    TRAIN_CSV = r"ModelPackager\train_features.csv"

    # 1) Train everything & serialize into "pipeline_components/" folder
    components_dir = train_and_serialize(
        train_csv_path=TRAIN_CSV,
        output_dir="pipeline_components"
    )

    # 2) Package those four files into pipeline_bundle.zip
    package_components(components_dir, zip_name="pipeline_bundle.zip")
