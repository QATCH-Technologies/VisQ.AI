import io

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 0. Data Strings & Config
# ==========================================
new_data_csv = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
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
# 1. Feature Engineering
# ==========================================
def process_full_features(train_df, new_df):
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

    all_df = pd.concat([train_df, new_df], axis=0, ignore_index=True)
    for col in num_cols:
        if col not in train_df.columns:
            train_df[col] = 0.0
        if col not in new_df.columns:
            new_df[col] = 0.0

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

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        X_new, dtype=torch.float32
    )


def extract_curves_v2(df):
    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }
    curves = []
    valid_indices = []
    for idx, row in df.iterrows():
        pts = []
        for col, shear in shear_map.items():
            if col in row and pd.notna(row[col]):
                visc = row[col]
                if visc <= 0:
                    visc = 1e-6
                pts.append([np.log10(shear), np.log10(visc)])
        if pts:
            curves.append(torch.tensor(pts, dtype=torch.float32))
            valid_indices.append(idx)
    return curves, valid_indices


# ==========================================
# 2. Model
# ==========================================
class FullFeatureCNP(nn.Module):
    def __init__(self, static_dim, encoder_hidden=128, decoder_hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.ReLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1 + encoder_hidden + static_dim, decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(self, context_x, context_y, query_x, static_features):
        pairs = torch.cat([context_x, context_y], dim=-1)
        encoded = self.encoder(pairs)
        latent = torch.mean(encoded, dim=1)
        n_q = query_x.size(1)
        latent_exp = latent.unsqueeze(1).repeat(1, n_q, 1)
        static_exp = static_features.unsqueeze(1).repeat(1, n_q, 1)
        in_vec = torch.cat([query_x, latent_exp, static_exp], dim=-1)
        return self.decoder(in_vec)


# ==========================================
# 3. Execution
# ==========================================
train_df = pd.read_csv("data/processed/formulation_data_augmented_no_trast.csv")
new_df = pd.read_csv(io.StringIO(new_data_csv))

train_curves, train_valid_idx = extract_curves_v2(train_df)
new_curves, new_valid_idx = extract_curves_v2(new_df)

train_df_valid = train_df.iloc[train_valid_idx].reset_index(drop=True)
new_df_valid = new_df.iloc[new_valid_idx].reset_index(drop=True)

X_train, X_new = process_full_features(train_df_valid, new_df_valid)

model = FullFeatureCNP(static_dim=X_train.shape[1])
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(300):  # Keep training iterations reasonable
    model.train()
    indices = np.random.choice(len(train_curves), 32)
    loss_accum = 0
    for i in indices:
        curve = train_curves[i]
        static = X_train[i].unsqueeze(0)
        if len(curve) < 2:
            continue
        n = len(curve)
        split = np.random.randint(1, n)
        perm = torch.randperm(n)
        ctx_idx = perm[:split]
        tgt_idx = perm
        ctx = curve[ctx_idx].unsqueeze(0)
        tgt = curve[tgt_idx].unsqueeze(0)
        pred = model(ctx[:, :, [0]], ctx[:, :, [1]], tgt[:, :, [0]], static)
        mse = F.mse_loss(pred, tgt[:, :, [1]])

        # Physics Loss
        tgt_x_grad = tgt[:, :, [0]].clone().detach().requires_grad_(True)
        pred_grad = model(ctx[:, :, [0]], ctx[:, :, [1]], tgt_x_grad, static)
        grads = torch.autograd.grad(
            pred_grad, tgt_x_grad, torch.ones_like(pred_grad), create_graph=True
        )[0]
        mono = torch.mean(torch.relu(grads))

        loss = mse + 0.1 * mono
        loss_accum += loss

    optim.zero_grad()
    (loss_accum / 32).backward()
    optim.step()

# Evaluation Loop
print("=== Predictions for All Trastuzumab Samples ===")
model.eval()
shear_rates = [100.0, 1000.0, 10000.0, 100000.0, 1.5e7]

# Iterate through all new samples
for sample_id in range(len(new_curves)):
    curve = new_curves[sample_id]
    sample_info = new_df_valid.iloc[sample_id]

    # Context: First 2 points (Shear 100, 1000)
    # We find indices where shear is 100 (log 2) or 1000 (log 3)
    ctx_indices = []
    for k in range(len(curve)):
        val = curve[k, 0].item()
        if np.isclose(val, 2.0) or np.isclose(val, 3.0):
            ctx_indices.append(k)

    if len(ctx_indices) == 0:
        continue

    ctx = curve[ctx_indices].unsqueeze(0)
    tgt = curve.unsqueeze(0)  # Predict full curve

    with torch.no_grad():
        pred = model(
            ctx[:, :, [0]],
            ctx[:, :, [1]],
            tgt[:, :, [0]],
            X_new[sample_id].unsqueeze(0),
        )

    print(f"\nSample ID: {sample_info['ID']}")
    print(f"{'Shear Rate':<15} | {'Actual':<10} | {'Predicted':<10} | {'Error %':<10}")
    print("-" * 55)

    # Sort by shear rate for display
    # Get values
    tgt_shear = 10 ** tgt[0, :, 0].numpy()
    tgt_visc = 10 ** tgt[0, :, 1].numpy()
    pred_visc = 10 ** pred[0, :, 0].numpy()

    # Sort
    sorted_idx = np.argsort(tgt_shear)

    for idx in sorted_idx:
        sh = tgt_shear[idx]
        act = tgt_visc[idx]
        prd = pred_visc[idx]
        err = abs(act - prd) / act * 100

        # Mark context points
        is_ctx = "(*)" if (np.isclose(sh, 100.0) or np.isclose(sh, 1000.0)) else ""

        print(f"{sh:<15.1f} | {act:<10.4f} | {prd:<10.4f} | {err:<9.1f} {is_ctx}")
    print("(*) = Context Point")
