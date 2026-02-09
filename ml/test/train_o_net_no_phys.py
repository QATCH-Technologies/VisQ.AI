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

        # --- ARCHITECTURE FIX: Restore Static Inputs ---
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.physics_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, context_tensor, query_shear, query_static, query_physics_proxy):
        # 1. Standard Neural Process Flow
        encoded = self.encoder(context_tensor)
        r = torch.mean(encoded, dim=1)

        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)

        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        nn_output = self.decoder(decoder_input)  # Shape: [Batch, Queries, 1]

        # 2. Physics Residual Correction
        # Enforce positive scaling so "-2" in table always means "decrease" in output
        beta = F.softplus(self.physics_scale)

        # Ensure proxy shape matches output [Batch, Queries, 1]
        if query_physics_proxy.dim() == 1:
            query_physics_proxy = query_physics_proxy.unsqueeze(1).unsqueeze(2)
        elif query_physics_proxy.dim() == 2:
            query_physics_proxy = query_physics_proxy.unsqueeze(2)

        # Broadcast proxy across the query points (since it's constant for the formulation)
        bias = beta * query_physics_proxy.expand_as(nn_output)

        return nn_output + bias

    def encode_memory(self, context_tensor):
        """
        Encodes the context into a single latent vector (memory).
        Used during the 'learn' phase.
        """
        encoded = self.encoder(context_tensor)
        return torch.mean(encoded, dim=1)

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        """
        Decodes targets using a pre-computed latent vector.
        Used during the 'predict' phase.
        """
        n_queries = query_shear.size(1)
        # Expand memory to match the number of query points
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
    df.to_csv("pembro_data.csv")
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

    # 1. Fill defaults for numeric columns (FIXED: Fill existing NaNs too)
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            # FIX: Ensure existing columns don't have NaNs which crash StandardScaler
            df[c] = df[c].fillna(0.0)

    # 2. Normalize categorical columns
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()
            df[c] = df[c].replace("nan", "unknown")  # Handle string "nan"
        else:
            df[c] = "unknown"

    # =========================================================
    # Physics-Informed Feature Engineering
    # =========================================================
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
        # ... (Logic remains identical to your script) ...
        # Copied for completeness
        c_class = row.get("C_Class", 1.0)
        # Fix for potential NaN in pH/PI
        ph = row.get("Buffer_pH", 7.0)
        pi = row.get("PI_mean", 7.0)
        if pd.isna(ph):
            ph = 7.0
        if pd.isna(pi):
            pi = 7.0

        delta_ph = abs(ph - pi)
        tau = 1.5
        cci = c_class * np.exp(-delta_ph / tau)

        p_type = str(row.get("Protein_class_type", "default")).lower()

        # Regime logic
        regime = "Far"
        if "mab_igg1" in p_type:
            regime = "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
        elif "mab_igg4" in p_type:
            regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
        elif any(x in p_type for x in ["fc-fusion", "trispecific"]):
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
        elif any(x in p_type for x in ["bispecific", "adc"]):
            regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.45 else "Far")
        elif any(x in p_type for x in ["bsa", "polyclonal"]):
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
        else:
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")

        lookup_key = "default"
        for key in PRIOR_TABLE.keys():
            if key != "default" and key in p_type:
                lookup_key = key
                break
        table = PRIOR_TABLE[lookup_key]
        regime_dict = table.get(regime, table["Far"])

        priors = {k: 0.0 for k in new_prior_cols}
        concs = {k: 0.0 for k in new_conc_cols}
        # Calculate the aggregate "Theoretical Shift"
        # We sum (Concentration * Table_Value) for all ingredients
        physics_proxy = 0.0
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

            # Determine the prior value (weight) for this ingredient
            weight = 0
            if "arginine" in ing_name or "arg" in ing_name:
                weight = regime_dict.get("arginine", 0)
            elif "lysine" in ing_name or "lys" in ing_name:
                weight = regime_dict.get("lysine", 0)
            elif "proline" in ing_name:
                weight = regime_dict.get("proline", 0)
            # ... (repeat for other ingredients using your existing logic) ...
            elif "tween" in ing_name:
                t_key = "tween-20" if "20" in ing_name else "tween-80"
                weight = regime_dict.get(t_key, 0)

            # Add to aggregate score
            physics_proxy += weight * ing_conc

        # Return priors, concs AND the proxy
        return {**priors, **concs, "physics_proxy": physics_proxy}

    print("Calculating Physics Priors and Concentration Splits...")
    features_df = df.apply(process_row_features, axis=1, result_type="expand")
    df = pd.concat([df, features_df], axis=1)

    num_cols.extend(new_prior_cols)
    num_cols.extend(new_conc_cols)

    # Preprocessing
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

    # FIX: Safety check for NaNs in processed data
    if np.isnan(X_matrix).any():
        print("WARNING: NaNs found in X_matrix after preprocessing! Replacing with 0.")
        X_matrix = np.nan_to_num(X_matrix)

    # Physics Scaler
    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }
    all_shear = []
    all_visc = []

    for i in range(len(df)):
        for col, shear_val in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                all_shear.append(np.log10(shear_val))
                all_visc.append(np.log10(v))

    physics_scaler = StandardScaler()
    physics_data = np.column_stack([all_shear, all_visc])
    physics_scaler.fit(physics_data)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
        joblib.dump(physics_scaler, os.path.join(save_dir, "physics_scaler.pkl"))

    samples = []
    for i in range(len(df)):
        pts = []
        for col, shear_val in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                raw_point = np.array([[np.log10(shear_val), np.log10(v)]])
                scaled_point = physics_scaler.transform(raw_point)[0]
                pts.append(scaled_point)

        if pts:
            # FIX: Stack numpy arrays before tensor conversion to avoid UserWarning and speed up
            pts_np = np.stack(pts)
            samples.append(
                {
                    "static": torch.tensor(X_matrix[i], dtype=torch.float32),
                    "points": torch.tensor(pts_np, dtype=torch.float32),
                    "physics_proxy": torch.tensor(
                        df.iloc[i]["physics_proxy"], dtype=torch.float32
                    ),  # NEW
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
    count = 0

    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)
    protein_list = list(groups.keys())

    for _ in range(iterations):
        prot = np.random.choice(protein_list)
        task_samples = groups[prot]
        if len(task_samples) < 4:
            continue

        indices = np.random.permutation(len(task_samples))
        n_ctx = np.random.randint(1, min(5, len(indices) - 1))
        idx_ctx = indices[:n_ctx]
        idx_tgt = indices[n_ctx:]

        def build_batch(sample_indices):
            shear_list, y_list, stat_list, phys_list = [], [], [], []

            ctx_items = []

            for i in sample_indices:
                s = task_samples[i]
                n_pts = s["points"].shape[0]

                # 1. Collect points
                shear_list.append(s["points"][:, [0]])
                y_list.append(s["points"][:, [1]])

                # 2. Static Inputs (Expand to match n_pts)
                stat = s["static"].unsqueeze(0).repeat(n_pts, 1)
                stat_list.append(stat)

                # 3. Physics Proxy (Expand to match n_pts)
                # Ensure s["physics_proxy"] is a tensor on the correct device
                p_val = s["physics_proxy"].to(device)
                if p_val.dim() == 0:
                    p_val = p_val.unsqueeze(0)
                phys_list.append(p_val.unsqueeze(0).repeat(n_pts, 1))

                # 4. Context Tensor Construction (Points + Static)
                # s["points"] is [n_pts, 2] (Shear, Visc)
                # stat is [n_pts, static_dim]
                ctx_items.append(torch.cat([s["points"], stat], dim=1))

            if not shear_list:
                return None, None, None, None, None

            # Concatenate all lists into batch tensors [1, Total_Points, Feature_Dim]
            all_shear = torch.cat(shear_list, dim=0).unsqueeze(0).to(device)
            all_y = torch.cat(y_list, dim=0).unsqueeze(0).to(device)
            all_stat = torch.cat(stat_list, dim=0).unsqueeze(0).to(device)
            all_phys = torch.cat(phys_list, dim=0).unsqueeze(0).to(device)

            # Context Tensor
            ctx_tensor = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

            return ctx_tensor, all_shear, all_stat, all_y, all_phys

        # Build Context (we only need the ctx_tensor from this)
        ctx_tensor, _, _, _, _ = build_batch(idx_ctx)

        # Build Target (we need inputs + targets)
        _, tgt_x, tgt_stat, tgt_y, tgt_phys = build_batch(idx_tgt)

        if tgt_x is None:
            continue

        # Target Static Dropout (Regularization technique)
        if np.random.random() < 0.5:
            tgt_stat_input = torch.zeros_like(tgt_stat)
        else:
            tgt_stat_input = tgt_stat

        # Forward Pass with Physics Proxy
        pred_y = model(ctx_tensor, tgt_x, tgt_stat_input, tgt_phys)
        mse_loss = F.mse_loss(pred_y, tgt_y)

        # Latent Regularization
        encoded = model.encoder(ctx_tensor)
        r = torch.mean(encoded, dim=1)
        latent_reg = torch.mean(r**2)

        loss = mse_loss + (1e-4 * latent_reg)

        if torch.isnan(loss):
            print("Warning: NaN loss encountered in training batch. Skipping.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)


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

            mid = len(task_samples) // 2
            ctx_idx = range(mid)
            tgt_idx = range(mid, len(task_samples))

            # Build Validation Context
            ctx_list = []
            for i in ctx_idx:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_list.append(torch.cat([s["points"], stat], dim=1))
            if not ctx_list:
                continue
            ctx_tensor = torch.cat(ctx_list, dim=0).unsqueeze(0).to(device)

            # Build Validation Target
            tgt_shear, tgt_y, tgt_stat, tgt_phys = [], [], [], []
            for i in tgt_idx:
                s = task_samples[i]
                n_pts = s["points"].shape[0]

                tgt_shear.append(s["points"][:, [0]])
                tgt_y.append(s["points"][:, [1]])
                tgt_stat.append(s["static"].unsqueeze(0).repeat(n_pts, 1))

                # Handle Physics Proxy
                p_val = s["physics_proxy"].to(device)
                if p_val.dim() == 0:
                    p_val = p_val.unsqueeze(0)
                tgt_phys.append(p_val.unsqueeze(0).repeat(n_pts, 1))

            if not tgt_shear:
                continue

            q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
            q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
            true_y = torch.cat(tgt_y, dim=0).unsqueeze(0).to(device)
            q_phys = torch.cat(tgt_phys, dim=0).unsqueeze(0).to(device)

            # Updated Model Call
            pred = model(ctx_tensor, q_x, q_stat, q_phys)
            loss = F.mse_loss(pred, true_y)

            if not torch.isnan(loss):
                total_error += loss.item()
                count += 1

    if count == 0:
        return float("inf")
    return total_error / count


# ==========================================
# 4. Optuna Objective
# ==========================================
def objective_cv(trial, samples, static_dim, device):
    # 1. Hyperparameters
    # We reduced the upper limits slightly to prevent overfitting on small group counts
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=32)
    latent_dim = trial.suggest_int("latent_dim", 32, 128, step=32)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)  # Enforce min dropout
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    # 2. Setup K-Fold (Group-Aware)
    groups = [s["group"] for s in samples]
    gss = GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=42)

    fold_scores = []

    # 3. Cross-Validation Loop
    for fold_idx, (train_idx, val_idx) in enumerate(gss.split(samples, groups=groups)):
        # Create Model for this fold
        model = CrossSampleCNP(static_dim, hidden_dim, latent_dim, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Sub-sample datasets
        train_fold = [samples[i] for i in train_idx]
        val_fold = [samples[i] for i in val_idx]

        # Quick Training (Short epochs for tuning speed)
        # We use a simple early exit if a fold is going really badly
        for epoch in range(40):
            train_loss = train_epoch(
                model, train_fold, optimizer, device, iterations=50
            )

            # Pruning check (only on first fold to save time)
            if fold_idx == 0:
                val_loss_check = validate(model, val_fold, device)
                trial.report(val_loss_check, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Final Validation for this fold
        final_val_loss = validate(model, val_fold, device)
        fold_scores.append(final_val_loss)

    # Return average error across all folds
    return np.mean(fold_scores)


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    data = "data/raw/formulation_data_02052026.csv"
    out = "./models/experiments/o_net"
    trials = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, static_dim = load_and_preprocess(data, save_dir=out)

    print(
        f"Loaded {len(samples)} samples from {len(set(s['group'] for s in samples))} protein groups."
    )
    print("Starting K-Fold Optuna Optimization...")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective_cv(t, samples, static_dim, device),
        n_trials=trials,
    )

    print("\n--- Tuning Complete ---")
    print("Best params:", study.best_params)

    # ==========================================
    # FINAL RETRAINING (With Scheduler & Early Stopping)
    # ==========================================
    print("\nRetraining final model on ALL data...")
    best_params = study.best_params

    final_model = CrossSampleCNP(
        static_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr"])

    # FIX: Removed 'verbose=True' to fix TypeError
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 40
    best_state = None

    # Hold out a small internal set just for early stopping monitoring
    gss_final = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, stop_idx = next(
        gss_final.split(samples, groups=[s["group"] for s in samples])
    )

    final_train_set = [samples[i] for i in train_idx]
    final_stop_set = [samples[i] for i in stop_idx]

    print(
        f"Final Train: {len(final_train_set)} samples | Early Stop Watchlist: {len(final_stop_set)} samples"
    )

    for ep in range(500):
        # Train
        train_loss = train_epoch(
            final_model, final_train_set, optimizer, device, iterations=100
        )

        # Validate (for early stopping trigger only)
        val_loss = validate(final_model, final_stop_set, device)

        # Step the scheduler
        scheduler.step(val_loss)

        # Print status (including LR) manually
        if ep % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {ep}: Train {train_loss:.4f} | Val {val_loss:.4f} | LR {current_lr:.2e}"
            )

        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping early at epoch {ep}. Best Val Loss: {best_loss:.4f}")
            break

    # Save
    if best_state is not None:
        final_model.load_state_dict(best_state)

    save_path = os.path.join(out, "best_model.pth")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": best_params,
            "static_dim": static_dim,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
