import argparse
import copy
import os
from collections import defaultdict

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# 1. Model Architecture
# ==========================================
class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()

        # --- ARCHITECTURE CHANGE ---
        # Encoder now takes ONLY 2 inputs: (Shear_Rate, Viscosity)
        # We removed 'static_dim' from here.
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # <--- Changed from (2 + static_dim)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder remains the same (Integrates Static + Context)
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        # --- INPUT SLICING ---
        # context_tensor is [Batch, Points, (Shear, Visc, Static...)]
        # We slice it to keep ONLY the first 2 dimensions (Shear, Visc) for the encoder
        context_physics = context_tensor[:, :, :2]

        # 1. Encode Physics Only
        encoded = self.encoder(context_physics)
        r = torch.mean(encoded, dim=1)

        # 2. Decode using Physics Context + Static Identity
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)

        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    # Helper for inference (add this if missing)
    def encode_memory(self, context_tensor):
        context_physics = context_tensor[:, :, :2]
        encoded = self.encoder(context_physics)
        return torch.mean(encoded, dim=1)

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_queries = query_shear.size(1)
        r_expanded = memory_vector.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)


# ==========================================
# 2. Data Pipeline
# ==========================================
CONC_THRESHOLDS = {
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    "tween-20": 0.01,
    "tween-80": 0.01,
    "stabilizer": 0.2,
    "trehalose": 0.2,
}
PRIOR_TABLE = {
    "mab_igg1": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Far": {
            "arginine": 0,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
    },
    "mab_igg4": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Far": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
    },
    "fc-fusion": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -2,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -2,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "bispecific": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "adc": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "bsa": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "polyclonal": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "default": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
}


def load_and_preprocess(csv_path, save_dir=None):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Feature Config
    cat_cols = [
        "Protein_type",
        "Protein_class_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    num_cols = [
        "kP",
        "MW",
        "PI_mean",
        "PI_range",
        "Protein_conc",
        "Temperature",
        "Buffer_pH",
        "Buffer_conc",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
        "Excipient_conc",
        "C_Class",
        "HCI",
    ]

    # 1. Fill defaults for numeric columns
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0

    # 2. Normalize categorical columns to lowercase
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()
        else:
            df[c] = "unknown"

    # =========================================================
    # NEW: Physics-Informed Feature Engineering
    # =========================================================

    # 2a. Define Priors and Split Features names
    # Note: These map to the keys in the PRIOR_TABLE's inner dictionaries
    new_prior_cols = [
        "prior_arginine",
        "prior_lysine",
        "prior_proline",
        "prior_nacl",
        "prior_stabilizer",
        "prior_tween-20",
        "prior_tween-80",
    ]

    new_conc_cols = []
    for k in CONC_THRESHOLDS.keys():
        new_conc_cols.append(f"{k}_low")
        new_conc_cols.append(f"{k}_high")

    def process_row_features(row):
        # --- A. Determine Regime ---
        c_class = row.get("C_Class", 1.0)
        delta_ph = abs(row.get("Buffer_pH", 7.0) - row.get("PI_mean", 7.0))
        tau = 1.5
        cci = c_class * np.exp(-delta_ph / tau)

        p_type = str(row.get("Protein_class_type", "default")).lower()
        regime = "Far"

        # Simplified boolean logic for Regime detection
        if "mab_igg1" in p_type:
            if cci >= 0.90:
                regime = "Near-pI"
            elif cci >= 0.50:
                regime = "Mixed"
        elif "mab_igg4" in p_type:
            if cci >= 0.80:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"
        elif any(x in p_type for x in ["fc-fusion", "trispecific"]):
            if cci >= 0.70:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"
        elif any(x in p_type for x in ["bispecific", "adc"]):
            if cci >= 0.80:
                regime = "Near-pI"
            elif cci >= 0.45:
                regime = "Mixed"
        elif any(x in p_type for x in ["bsa", "polyclonal"]):
            if cci >= 0.70:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"
        else:
            if cci >= 0.70:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"

        # --- B. Get Prior Table ---
        lookup_key = "default"
        for key in PRIOR_TABLE.keys():
            if key != "default" and key in p_type:
                lookup_key = key
                break

        table = PRIOR_TABLE[lookup_key]
        regime_dict = table.get(regime, table["Far"])

        # --- C. Calculate Priors & Split Concentrations ---
        # Initialize with 0.0
        priors = {k: 0.0 for k in new_prior_cols}
        concs = {k: 0.0 for k in new_conc_cols}

        # Pairs of (Type_Col, Conc_Col) to scan
        scan_cols = [
            ("Salt_type", "Salt_conc"),
            ("Stabilizer_type", "Stabilizer_conc"),
            ("Excipient_type", "Excipient_conc"),
            ("Surfactant_type", "Surfactant_conc"),
        ]

        for type_col, conc_col in scan_cols:
            ing_name = str(row.get(type_col, "none")).lower()
            ing_conc = float(row.get(conc_col, 0.0))

            if ing_name in ["none", "unknown", "nan"] or ing_conc <= 0:
                continue

            # --- Logic for PRIORS ---
            if "arginine" in ing_name or "arg" in ing_name:
                priors["prior_arginine"] = regime_dict.get("arginine", 0)
            elif "lysine" in ing_name or "lys" in ing_name:
                priors["prior_lysine"] = regime_dict.get("lysine", 0)
            elif "proline" in ing_name:
                priors["prior_proline"] = regime_dict.get("proline", 0)
            elif "nacl" in ing_name:
                priors["prior_nacl"] = regime_dict.get("nacl", 0)

            # Stabilizer check
            elif type_col == "Stabilizer_type":
                # This catches sucrose, trehalose, etc as general stabilizers
                priors["prior_stabilizer"] = regime_dict.get("stabilizer", 0)

            # Tween check
            elif "tween" in ing_name or "polysorbate" in ing_name:
                # Specific map for tween-20 vs 80
                t_key = "tween-20" if "20" in ing_name else "tween-80"
                priors[f"prior_{t_key}"] = regime_dict.get(t_key, 0)

            # --- Logic for CONCENTRATION SPLITS ---
            for target_ing, threshold in CONC_THRESHOLDS.items():
                # Check if current ingredient matches target
                match = False
                if target_ing in ing_name:
                    match = True
                elif target_ing == "arginine" and "arg" in ing_name:
                    match = True

                if match:
                    # Calculate Splits
                    e_low = min(ing_conc, threshold)
                    e_high = max(ing_conc - threshold, 0)

                    concs[f"{target_ing}_low"] = e_low
                    concs[f"{target_ing}_high"] = e_high

        # Return concatenated dict
        return {**priors, **concs}

    # Apply Logic
    print("Calculating Physics Priors and Concentration Splits...")
    # Apply function and expand result dictionary into columns
    features_df = df.apply(process_row_features, axis=1, result_type="expand")

    # Concatenate new features to original dataframe
    df = pd.concat([df, features_df], axis=1)

    # Add all new columns to num_cols so they get normalized
    num_cols.extend(new_prior_cols)
    num_cols.extend(new_conc_cols)

    # =========================================================
    # Pipeline & Processing
    # =========================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    X_matrix = preprocessor.fit_transform(df)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))

    # Flatten Data
    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }

    samples = []
    for i in range(len(df)):
        pts = []
        for col, shear_val in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                pts.append([np.log10(shear_val), np.log10(v)])

        if pts:
            samples.append(
                {
                    "static": torch.tensor(X_matrix[i], dtype=torch.float32),
                    "points": torch.tensor(pts, dtype=torch.float32),
                    "group": df.iloc[i]["Protein_type"],
                    "id": df.iloc[i]["ID"],
                }
            )

    return samples, X_matrix.shape[1]


# ==========================================
# 3. Training/Validation Helpers
# ==========================================
def train_epoch(model, samples, optimizer, device, iterations=100):
    model.train()
    total_loss = 0

    # Group for task sampling
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)
    protein_list = list(groups.keys())

    for _ in range(iterations):
        prot = np.random.choice(protein_list)
        task_samples = groups[prot]

        # Need at least 4 samples to create disjoint context sets
        if len(task_samples) < 4:
            continue

        # -----------------------------------------------------------
        # FIX 1: Context Splitting for Latent Consistency
        # Split data into Context A, Context B, and Target
        # -----------------------------------------------------------
        indices = np.random.permutation(len(task_samples))
        n_total = len(indices)

        # Randomly size the splits (ensure at least 1 sample each)
        n_ctx_a = np.random.randint(1, min(5, n_total - 2))
        n_ctx_b = np.random.randint(1, min(5, n_total - n_ctx_a - 1))
        n_tgt = min(5, n_total - n_ctx_a - n_ctx_b)

        idx_a = indices[:n_ctx_a]
        idx_b = indices[n_ctx_a : n_ctx_a + n_ctx_b]
        idx_tgt = indices[n_ctx_a + n_ctx_b : n_ctx_a + n_ctx_b + n_tgt]

        # Helper to build batch tensors
        def build_batch(sample_indices):
            shear_list, y_list, stat_list, combined_list = [], [], [], []
            for i in sample_indices:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)

                shear_list.append(s["points"][:, [0]])
                y_list.append(s["points"][:, [1]])
                stat_list.append(stat)
                combined_list.append(torch.cat([s["points"], stat], dim=1))

            # Cat and move to device
            if not combined_list:
                return None, None, None, None
            return (
                torch.cat(combined_list, dim=0).unsqueeze(0).to(device),
                torch.cat(shear_list, dim=0).unsqueeze(0).to(device),
                torch.cat(stat_list, dim=0).unsqueeze(0).to(device),
                torch.cat(y_list, dim=0).unsqueeze(0).to(device),
            )

        ctx_a_tensor, _, _, _ = build_batch(idx_a)
        ctx_b_tensor, _, _, _ = build_batch(idx_b)
        _, tgt_x, tgt_stat, tgt_y = build_batch(idx_tgt)

        if tgt_x is None:
            continue

        # -----------------------------------------------------------
        # FIX 2: Static Dropout ("Blindfolding")
        # 50% of the time, zero out static features in the DECODER.
        # This forces the model to look at Context A to identify the protein.
        # -----------------------------------------------------------
        if np.random.random() < 0.5:
            tgt_stat_input = torch.zeros_like(tgt_stat)
        else:
            tgt_stat_input = tgt_stat

        # Forward Pass
        pred_y = model(ctx_a_tensor, tgt_x, tgt_stat_input)
        mse_loss = F.mse_loss(pred_y, tgt_y)

        # -----------------------------------------------------------
        # FIX 3: Consistency Loss
        # Enforce that Context A and Context B produce the same latent 'r'
        # -----------------------------------------------------------
        encoded_a = model.encoder(ctx_a_tensor[:, :, :2])
        r_a = torch.mean(encoded_a, dim=1)

        encoded_b = model.encoder(ctx_b_tensor[:, :, :2])
        r_b = torch.mean(encoded_b, dim=1)

        # 2. Consistency Loss (Keep contexts similar)
        consistency_loss = F.mse_loss(r_a, r_b)

        # 3. NEW: Latent Regularization (Keep r small/sparse)
        # This aligns the "Learned" r with the "Zero-Shot" r
        latent_reg = torch.mean(r_a**2) + torch.mean(r_b**2)

        # Combined Loss
        # mse_loss: fit the data
        # 0.1 * consistency: be consistent across samples
        # 0.01 * latent_reg: stay close to zero (prior)
        loss = mse_loss + (0.1 * consistency_loss) + (0.01 * latent_reg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / iterations


def validate(model, samples, device):
    model.eval()
    total_error = 0
    count = 0
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue

            # 50/50 Split for robust validation
            mid = len(task_samples) // 2
            ctx_idx = range(mid)
            tgt_idx = range(mid, len(task_samples))

            # Context
            ctx_list = []
            for i in ctx_idx:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_list.append(torch.cat([s["points"], stat], dim=1))
            ctx_tensor = torch.cat(ctx_list, dim=0).unsqueeze(0).to(device)

            # Targinineet
            tgt_shear, tgt_y, tgt_stat = [], [], []
            for i in tgt_idx:
                s = task_samples[i]
                tgt_shear.append(s["points"][:, [0]])
                tgt_y.append(s["points"][:, [1]])
                tgt_stat.append(
                    s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                )

            if not tgt_shear:
                continue
            q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
            q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
            true_y = torch.cat(tgt_y, dim=0).unsqueeze(0).to(device)

            loss = F.mse_loss(model(ctx_tensor, q_x, q_stat), true_y)
            total_error += loss.item()
            count += 1

    return total_error / max(1, count)


# ==========================================
# 4. Optuna Objective
# ==========================================
def objective(trial, train_samples, val_samples, static_dim, device):
    # Hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=32)
    latent_dim = trial.suggest_int("latent_dim", 16, 64, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model = CrossSampleCNP(static_dim, hidden_dim, latent_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fast epoch count for tuning (use more for final train)
    epochs = 100

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_samples, optimizer, device)

        # Report intermediate result for pruning
        val_loss = validate(model, val_samples, device)

        trial.report(val_loss, epoch)

        # Pruning (Early Stopping for bad trials)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":

    data = "data/processed/formulation_data_augmented_no_trast.csv"
    out = "./models/experiments/o_net_no_trast"
    trials = 25
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, static_dim = load_and_preprocess(data, save_dir=out)

    # Validating on unseen proteins (GroupShuffleSplit)
    groups = [s["group"] for s in samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(samples, groups=groups))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    print(f"Training on {len(set([s['group'] for s in train_samples]))} proteins.")
    print(
        f"Validating on {len(set([s['group'] for s in val_samples]))} unseen proteins."
    )

    # Run Optuna
    print("\n--- Starting Optuna Optimization ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, train_samples, val_samples, static_dim, device),
        n_trials=trials,
    )

    print("\n--- Tuning Complete ---")
    print("Best params:", study.best_params)
    print("Best validation loss:", study.best_value)

    # Retrain final model on best params
    print("\nRetraining final model on full dataset with best params...")
    best_params = study.best_params

    final_model = CrossSampleCNP(
        static_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])

    # Train longer for final model
    for ep in range(300):
        loss = train_epoch(
            final_model, samples, optimizer, device, iterations=100
        )  # Use ALL samples
        if ep % 50 == 0:
            print(f"Final Train Epoch {ep}: {loss:.4f}")

    # Save
    save_path = os.path.join(out, "best_model.pth")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": best_params,
            "static_dim": static_dim,
        },
        save_path,
    )

    print(f"\nSUCCESS. Model saved to {save_path}")
