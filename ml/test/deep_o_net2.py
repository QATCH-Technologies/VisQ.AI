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
# 1. Configuration & Data Strings
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

    # Define Feature Columns
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

    # Fill defaults
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
# 3. Model Architecture
# ==========================================
class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, latent_dim=128):
        super().__init__()
        # Encoder: Shear(1) + Visc(1) + Static(D) -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        # Decoder: Query_Shear(1) + Query_Static(D) + Latent(L) -> Visc
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        # 1. Encode context set
        encoded = self.encoder(context_tensor)
        # 2. Average to get protein representation
        r = torch.mean(encoded, dim=1)
        # 3. Predict for query set
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    train_samples, new_samples, static_dim = process_data(TRAIN_FILE, NEW_DATA_CSV)

    model = CrossSampleCNP(static_dim=static_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Prepare Meta-Training Data
    protein_groups = defaultdict(list)
    for s in train_samples:
        protein_groups[s["group"]].append(s)

    print("Meta-Training Model...")
    for epoch in range(1000):  # Short for demo; use 1000+ for production
        model.train()
        prot = np.random.choice(list(protein_groups.keys()))
        samples = protein_groups[prot]
        if len(samples) < 2:
            continue

        # Split: Context vs Target
        k_shot = np.random.randint(1, min(6, len(samples)))
        indices = np.random.permutation(len(samples))
        ctx_idx, tgt_idx = indices[:k_shot], indices[k_shot:]
        if len(tgt_idx) > 5:
            tgt_idx = tgt_idx[:5]
        if len(tgt_idx) == 0:
            continue

        # Create Tensors
        ctx_inputs = []
        for i in ctx_idx:
            s = samples[i]
            stat_expanded = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            ctx_inputs.append(torch.cat([s["points"], stat_expanded], dim=1))
        ctx_tensor = torch.cat(ctx_inputs, dim=0).unsqueeze(0)

        tgt_shears, tgt_statics, tgt_viscs = [], [], []
        for i in tgt_idx:
            s = samples[i]
            tgt_shears.append(s["points"][:, [0]])
            tgt_viscs.append(s["points"][:, [1]])
            tgt_statics.append(s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1))

        q_x = torch.cat(tgt_shears, dim=0).unsqueeze(0)
        q_static = torch.cat(tgt_statics, dim=0).unsqueeze(0)
        true_y = torch.cat(tgt_viscs, dim=0).unsqueeze(0)

        loss = F.mse_loss(model(ctx_tensor, q_x, q_static), true_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # ==========================================
    # 5. Incremental Learning Evaluation
    # ==========================================
    print("\n=== Detailed Incremental Learning Evaluation ===")
    np.random.seed(42)  # Ensure consistent 'random' learning order
    learn_order_idx = np.random.permutation(len(new_samples))
    learn_order_ids = [new_samples[i]["id"] for i in learn_order_idx]

    print(f"Learning Sequence: {learn_order_ids}")

    steps = [1, 3, 5, 10]  # Show results after learning 1, 3, and 5 samples

    for k in steps:
        print(f"\n--- After Learning {k} Sample(s) ---")
        print(f"Context (Known) Samples: {learn_order_ids[:k]}")
        print(
            f"{'ID':<6} | {'Status':<7} | {'Shear':<8} | {'Actual':<8} | {'Pred':<8} | {'Error %':<7}"
        )
        print("-" * 65)

        # Build Context Tensor from first k samples
        ctx_inputs = []
        for i in learn_order_idx[:k]:
            s = new_samples[i]
            stat_expanded = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            ctx_inputs.append(torch.cat([s["points"], stat_expanded], dim=1))
        ctx_tensor = torch.cat(ctx_inputs, dim=0).unsqueeze(0)

        avg_mapes = []

        # Predict Every Sample (Context + Test)
        for i in range(len(new_samples)):
            s = new_samples[i]
            sample_id = s["id"]
            status = "CTX" if sample_id in learn_order_ids[:k] else "TEST"

            # Query Tensor
            q_x = s["points"][:, [0]].unsqueeze(0)
            q_static = (
                s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1).unsqueeze(0)
            )

            model.eval()
            with torch.no_grad():
                pred_log = model(ctx_tensor, q_x, q_static)

            real_visc = 10 ** s["points"][:, 1]
            pred_visc = 10 ** pred_log[0, :, 0]
            shear_rates = 10 ** s["points"][:, 0]

            # Calculate Error
            mape = (
                torch.mean(torch.abs((real_visc - pred_visc) / real_visc)).item() * 100
            )
            avg_mapes.append(mape)

            # Print row (showing the lowest shear point)
            idx = 0
            print(
                f"{sample_id:<6} | {status:<7} | {shear_rates[idx]:<8.1f} | {real_visc[idx]:<8.4f} | {pred_visc[idx]:<8.4f} | {mape:<7.1f}"
            )

        print("-" * 65)
        print(f"Mean MAPE across all samples: {np.mean(avg_mapes):.2f}%")
