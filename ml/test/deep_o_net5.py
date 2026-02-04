import copy
import io
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. Configuration & Data
# ==========================================
TRAIN_FILE = "data/processed/formulation_data_augmented_no_trast.csv"

NEW_DATA_CSV = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F304,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,147,25,Acetate,5,20,none,0,none,0,none,0,none,0,1,1,5.83,4.79,2.84,2.8,1.06
F305,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,157,25,Histidine,6,15,none,0,none,0,none,0,none,0,1,1,3.75,3.75,3.67,3.35,1.9
F306,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,100,25,Histidine,6,15,none,0,Sucrose,0.3,tween-80,0.05,none,0,1,1,2.72,2.72,2.64,2.64,2.34
F307,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,50,25,Histidine,6,15,none,0,none,0,tween-80,0.1,none,0,1,1,1.52,1.52,1.52,1.36,1.2
F308,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,75,25,Histidine,6,15,none,0,Sucrose,0.5,none,0,none,0,1,1,1.56,1.56,1.56,1.56,0.8
F309,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,90,25,Acetate,5,20,none,0,Sucrose,0.3,tween-80,0.1,none,0,1,1,3.19,2.72,2.48,2.72,1.85
F310,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,70,25,Acetate,5,20,none,0,Sucrose,0.5,none,0,none,0,1,1,3.12,2.64,2.64,2.16,1.8
F311,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,100,25,Acetate,5,20,none,0,none,0,tween-80,0.05,none,0,1,1,2.8,2.08,1.6,1.6,1.6
F312,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,60,25,Acetate,5,20,none,0,Sucrose,0.4,tween-80,0.05,none,0,1,1,2,2,1.68,1.44,1.25
F313,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,15,none,0,none,0,none,0,none,0,1,1,2.36,2.36,2.36,2.36,2.36
F314,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,none,0,none,0,none,0,1,1,3.19,2.52,1.92,1.68,1.68
F315,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,none,0,none,0,Arginine,60,1,1,2.88,2.88,2.52,2.52,2.5
F316,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,none,0,none,0,Arginine,160,1,1,3.35,2.72,2.08,1.73,1.73
F317,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,40,none,0,none,0,none,0,none,0,1,1,2.08,2.08,1.64,1.64,1.64
F319,Trastuzumab,mAb_IgG1,3,149,8.9,0.3,125,25,Histidine,6,25,none,0,Sucrose,0.2,none,0,Arginine,60,1,1,2.96,2.53,2.16,2,2"""


# ==========================================
# 2. Data Processing Logic
# ==========================================
def process_data(train_csv_path, new_csv_string):
    train_df = pd.read_csv(train_csv_path)
    new_df = pd.read_csv(io.StringIO(new_csv_string))

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

    # Handle missing columns
    all_df = pd.concat([train_df, new_df], axis=0, ignore_index=True)
    for c in num_cols:
        if c not in train_df:
            train_df[c] = 0.0
        if c not in new_df:
            new_df[c] = 0.0

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

    preprocessor.fit(train_df)
    X_train = preprocessor.transform(train_df)
    X_new = preprocessor.transform(new_df)

    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }

    def flatten_dataset(X, df):
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
                        "static": torch.tensor(X[i], dtype=torch.float32),
                        "points": torch.tensor(pts, dtype=torch.float32),
                        "group": df.iloc[i]["Protein_type"],
                        "id": df.iloc[i]["ID"],
                    }
                )
        return samples

    train_samples = flatten_dataset(X_train, train_df)
    new_samples = flatten_dataset(X_new, new_df)

    return train_samples, new_samples, X_train.shape[1]


# ==========================================
# 3. Model with Memory
# ==========================================
class MemoryAwareCNP(nn.Module):
    def __init__(self, static_dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        encoded = self.encoder(context_tensor)
        r = torch.mean(encoded, dim=1)
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    def fine_tune(
        self, context_tensor, target_shear, target_static, target_y, steps=20, lr=1e-4
    ):
        """
        Updates model weights (Long-term Memory Storage)
        """
        ft_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(steps):
            pred = self(context_tensor, target_shear, target_static)
            loss = F.mse_loss(pred, target_y)
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
        return loss.item()


# ==========================================
# 4. Generic Evaluation Function
# ==========================================
def evaluate_dataset(model, samples, sample_limit=50):
    """
    Generic evaluator.
    Can measure 'Learning' (if samples=new_data) or 'Forgetting' (if samples=old_data).
    """
    model.eval()
    errors = []

    # Use all samples if list is small, otherwise sample random subset
    if len(samples) <= sample_limit:
        indices = range(len(samples))
    else:
        indices = np.random.choice(len(samples), sample_limit, replace=False)

    with torch.no_grad():
        for i in indices:
            s = samples[i]
            n_pts = s["points"].shape[0]
            if n_pts < 2:
                continue

            # Split-Half Context/Target validation
            half = n_pts // 2

            # Context
            ctx_pts = s["points"][:half]
            stat_expanded = s["static"].unsqueeze(0).repeat(half, 1)
            ctx_tensor = torch.cat([ctx_pts, stat_expanded], dim=1).unsqueeze(0)

            # Target
            tgt_pts = s["points"][half:]
            tgt_stat = s["static"].unsqueeze(0).repeat(n_pts - half, 1).unsqueeze(0)
            tgt_shear = tgt_pts[:, [0]].unsqueeze(0)
            tgt_y = tgt_pts[:, [1]].unsqueeze(0)

            pred = model(ctx_tensor, tgt_shear, tgt_stat)
            mse = F.mse_loss(pred, tgt_y).item()
            errors.append(mse)

    return np.mean(errors)


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    train_samples, new_samples, static_dim = process_data(TRAIN_FILE, NEW_DATA_CSV)

    model = MemoryAwareCNP(static_dim=static_dim)

    # ------------------------------------------
    # Phase 1: Meta-Training
    # ------------------------------------------
    print("Phase 1: Meta-Training Base Model (Learning Physics)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    protein_groups = defaultdict(list)
    for s in train_samples:
        protein_groups[s["group"]].append(s)

    for epoch in range(601):
        model.train()
        prot = np.random.choice(list(protein_groups.keys()))
        samples = protein_groups[prot]
        if len(samples) < 2:
            continue

        indices = np.random.permutation(len(samples))
        k = np.random.randint(1, min(6, len(samples)))
        ctx_idx, tgt_idx = indices[:k], indices[k : k + 5]
        if len(tgt_idx) == 0:
            continue

        ctx_list = []
        for i in ctx_idx:
            s = samples[i]
            stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            ctx_list.append(torch.cat([s["points"], stat], dim=1))
        ctx_tensor = torch.cat(ctx_list, dim=0).unsqueeze(0)

        tgt_shears, tgt_stats, tgt_ys = [], [], []
        for i in tgt_idx:
            s = samples[i]
            tgt_shears.append(s["points"][:, [0]])
            tgt_ys.append(s["points"][:, [1]])
            tgt_stats.append(s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1))

        q_x = torch.cat(tgt_shears, dim=0).unsqueeze(0)
        q_stat = torch.cat(tgt_stats, dim=0).unsqueeze(0)
        true_y = torch.cat(tgt_ys, dim=0).unsqueeze(0)

        loss = F.mse_loss(model(ctx_tensor, q_x, q_stat), true_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: Loss {loss.item():.4f}")

    # ------------------------------------------
    # Phase 2: Plasticity-Stability Eval
    # ------------------------------------------
    print("\nPhase 2: Fine-Tuning & Dual Evaluation...")
    print("Measuring Learning (New MSE) vs Forgetting (Old MSE)\n")

    # Baselines
    baseline_old = evaluate_dataset(model, train_samples)
    baseline_new = evaluate_dataset(model, new_samples)

    print(
        f"Baseline (0-Shot) | Old MSE: {baseline_old:.4f} | New MSE: {baseline_new:.4f}"
    )
    print("-" * 75)
    print(
        f"{'Samples Learned':<15} | {'Old MSE (Forget)':<18} | {'New MSE (Learn)':<18} | {'Status'}"
    )
    print("-" * 75)

    learn_order_idx = np.random.permutation(len(new_samples))

    for k in range(1, 11):
        # 1. Fine-Tune on kth sample
        idx = learn_order_idx[k - 1]
        sample = new_samples[idx]

        stat_expanded = (
            sample["static"].unsqueeze(0).repeat(sample["points"].shape[0], 1)
        )
        ctx_tensor = torch.cat([sample["points"], stat_expanded], dim=1).unsqueeze(0)
        q_x = sample["points"][:, [0]].unsqueeze(0)
        q_stat = stat_expanded.unsqueeze(0)
        true_y = sample["points"][:, [1]].unsqueeze(0)

        # Update weights (Fine-Tuning)
        model.fine_tune(ctx_tensor, q_x, q_stat, true_y, steps=15, lr=5e-5)

        # 2. Evaluate Both Datasets
        current_old_mse = evaluate_dataset(model, train_samples)
        current_new_mse = evaluate_dataset(
            model, new_samples
        )  # Performance on ALL Trastuzumab

        drift = current_old_mse - baseline_old
        learning_gain = baseline_new - current_new_mse

        status = "OK"
        if drift > 0.05:
            status = "DRIFT"
        if drift > 0.2:
            status = "CATASTROPHIC"

        print(
            f"{k:<15} | {current_old_mse:.4f} ({drift:+.3f})   | {current_new_mse:.4f} ({learning_gain:+.3f})   | {status}"
        )
