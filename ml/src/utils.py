"""
Utilities module for the Viscosity library.

This module provides helper functions for data preprocessing, transformation,
and validation. It includes routines for:
    - Cleaning pandas DataFrames (imputation, string standardization).
    - Logarithmic transformations for target variables.
    - Converting NumPy arrays to PyTorch tensors.
    - Calculating sample weights to handle class imbalance in regression tasks.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

HIGH_VIS_THRESHOLD = 20.0
HIGH_VIS_MULTIPLIER = 5.0


def clean(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean and preprocess the input dataframe.

    This function performs several standard cleaning operations:
    1.  Standardizes string columns (lowercase, strip whitespace).
    2.  Converts "bad" string representations (e.g., "nan", "null") to actual NaNs.
    3.  Separates numeric columns into 'concentrations' and 'physical properties'
        based on column naming conventions.
    4.  Imputes missing values:
        - Concentrations default to 0.0.
        - Physical properties default to the column median.
    5.  Fills missing categorical values with the string "none".
    6.  Optionally filters out rows with invalid or zero values in target columns.

    Args:
        df (pd.DataFrame): The raw input dataframe.
        numeric_cols (List[str]): List of column names to be treated as numeric features.
        categorical_cols (List[str]): List of column names to be treated as categorical features.
        target_cols (Optional[List[str]]): List of target column names. If provided,
            rows with missing or non-positive targets will be dropped.

    Returns:
        pd.DataFrame: A copy of the processed and cleaned dataframe.
    """
    df_clean = df.copy()
    bad_str = {"nan", "none", "null", "", "na", "n/a"}

    # Standardize string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = (
                df_clean[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace(list(bad_str), np.nan)
            )

    # Separate concentration columns from physical property columns
    conc_cols = [c for c in numeric_cols if "conc" in c.lower()]
    phys_cols = [c for c in numeric_cols if c not in conc_cols]

    # Convert to numeric
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Impute missing values
    # Concentrations default to 0.0
    df_clean[conc_cols] = df_clean[conc_cols].fillna(0.0)

    # Physical properties default to median
    for col in phys_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_clean[col] = df_clean[col].fillna(median_val)

    # Handle Categoricals
    for col in categorical_cols:
        df_clean[col] = df_clean[col].replace({np.nan: "none"})

    # Filter Targets
    if target_cols:
        for col in target_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean.dropna(subset=target_cols)
        for col in target_cols:
            df_clean = df_clean[df_clean[col] > 0]

    return df_clean


def log_transform_targets(y: np.ndarray) -> np.ndarray:
    """
    Apply Log10 transformation with epsilon shift.

    Used to normalize targets that span several orders of magnitude (e.g., viscosity).
    The epsilon shift (1e-8) prevents errors when transforming zero values.

    Args:
        y (np.ndarray): The raw target values.

    Returns:
        np.ndarray: The log-transformed values.
    """
    return np.log10(y + 1e-8)


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """
    Apply inverse Log10 transformation to recover original scale.

    Args:
        y_log (np.ndarray): The log-transformed predictions or targets.

    Returns:
        np.ndarray: The values in their original scale.
    """
    return 10**y_log - 1e-8


def to_tensors(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Convert numpy arrays to PyTorch tensors.

    Args:
        X_num (np.ndarray): Numeric features array.
        X_cat (np.ndarray): Categorical features array.
        y (Optional[np.ndarray]): Target array. Defaults to None.
        weights (Optional[np.ndarray]): Sample weights array. Defaults to None.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the converted tensors.
        The order is always (X_num, X_cat), followed by y and weights if they
        were provided.
    """
    tensors = [
        torch.tensor(X_num, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
    ]
    if y is not None:
        tensors.append(torch.tensor(y, dtype=torch.float32))
    if weights is not None:
        tensors.append(torch.tensor(weights, dtype=torch.float32))
    return tuple(tensors)


def validate_data(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.ndarray,
    name: str = "data",
) -> bool:
    """
    Validate data for NaN and Inf values.

    Ensures that the numerical inputs and targets are clean before being passed
    into the model to prevent training instability.

    Args:
        X_num (np.ndarray): Numeric features.
        X_cat (np.ndarray): Categorical features.
        y (np.ndarray): Target values.
        name (str, optional): Name of the dataset for error messages. Defaults to "data".

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If data contains NaN or Inf values.
    """
    if np.any(np.isnan(X_num)):
        raise ValueError(f"NaN in {name} numeric features")
    if np.any(np.isinf(X_num)):
        raise ValueError(f"Inf in {name} numeric features")
    if np.any(np.isnan(y)):
        raise ValueError(f"NaN in {name} targets")
    return True


def calculate_sample_weights(y_raw: np.ndarray) -> np.ndarray:
    """
    Calculate sample weights based on viscosity values.

    This function generates weights to address the imbalance in scientific data
    where high-viscosity samples are rare but critical. It uses a logarithmic
    base weight and applies a multiplier for samples exceeding a threshold.

    Logic:
        1. Base weight = 1.0 + log1p(viscosity)
        2. High viscosity (> 20.0) multiplier = 5.0

    Args:
        y_raw (np.ndarray): Raw target values of shape (n_samples, n_targets).
            Assumes the primary viscosity target is at index 0.

    Returns:
        np.ndarray: Sample weights of shape (n_samples,).
    """
    # Viscosity_100 is usually the first target column (index 0)
    v100 = y_raw[:, 0]

    # Base weight: Logarithmic
    weights = 1.0 + np.log1p(v100)

    # Boost High Viscosity Samples
    # Samples > 20 cP are rare
    high_vis_mask = v100 > HIGH_VIS_THRESHOLD
    weights[high_vis_mask] *= HIGH_VIS_MULTIPLIER

    return weights.astype(np.float32)
