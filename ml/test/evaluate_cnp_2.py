"""
evaluate_cnp.py  (v2 — updated for train_o_net_v2.py)
======================================================
Rigorous evaluation framework for the CrossSampleCNP in a limited-data regime.
Updated to be compatible with v2 model checkpoints (LayerNorm in AttentionPool)
and to evaluate the specific improvements made in v2 training.

Four complementary evaluation strategies:
  1. Leave-One-Group-Out (LOGO) CV    — generalization to unseen protein types
                                        (retrains with v2 contrastive losses)
  2. Within-Group Few-Shot Sweep      — context-size sensitivity (1..N shots)
  3. Bootstrap CI on aggregate metrics — uncertainty on the evaluation itself
  4. [NEW] Latent Variance Diagnostic — verifies context-collapse fix worked
  5. [NEW] Context Utility Diagnostic — measures real vs null context RMSE gain

Outputs:
  - logo_results.csv          : per-fold RMSE/MAE (log-space viscosity)
  - fewshot_sweep.csv         : RMSE vs context size across protein groups
  - bootstrap_ci.csv          : 95% CI on aggregate metrics
  - per_group_diagnostics.csv : detailed per-protein-type breakdown
  - latent_variance.csv       : [NEW] inter/intra group latent L2 distances
  - context_utility.csv       : [NEW] real vs null context RMSE per group
  - evaluation_report.txt     : human-readable summary of all above

Usage:
    python evaluate_cnp.py \
        --data  formulation_data_02162026.csv \
        --model ./models/experiments/o_net_v2/best_model.pth \
        --out   ./eval_results_v2
"""

import argparse
import copy
import os
from collections import defaultdict
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# 0. Re-import model + preprocessing from your training file
#    (kept self-contained so this script can run standalone)
# ============================================================


class AttentionPool(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        # v2: LayerNorm added after attention output to stabilize latent scale.
        # Required to load v2 checkpoints — omitting this causes a state_dict
        # key mismatch on pooler.norm.weight / pooler.norm.bias.
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        encoded = self.encoder(context_tensor)
        r = self.pooler(encoded)
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    def encode_memory(self, context_tensor):
        return self.pooler(self.encoder(context_tensor))

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_queries = query_shear.size(1)
        r_expanded = memory_vector.unsqueeze(1).repeat(1, n_queries, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_expanded], dim=-1))


# ============================================================
# 1. Preprocessing (mirrors train_o_net_no_phys.py exactly)
# ============================================================
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
    df = pd.read_csv(csv_path)

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

    for c in num_cols:
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().replace("nan", "unknown")
        else:
            df[c] = "unknown"

    new_prior_cols = [
        "prior_arginine",
        "prior_lysine",
        "prior_proline",
        "prior_nacl",
        "prior_stabilizer",
        "prior_tween-20",
        "prior_tween-80",
    ]
    new_conc_cols = [f"{k}_{s}" for k in CONC_THRESHOLDS for s in ("low", "high")]

    df["log_conc"] = np.log1p(df["Protein_conc"])
    df["conc_sq"] = df["Protein_conc"] ** 2
    df["conc_x_kP"] = df["Protein_conc"] * df["kP"]
    df["conc_x_HCI"] = df["Protein_conc"] * df["HCI"]

    def process_row(row):
        ph = float(row.get("Buffer_pH", 7.0) or 7.0)
        pi = float(row.get("PI_mean", 7.0) or 7.0)
        ccl = float(row.get("C_Class", 1.0) or 1.0)
        cci = ccl * np.exp(-abs(ph - pi) / 1.5)
        p = str(row.get("Protein_class_type", "default")).lower()
        if "mab_igg1" in p:
            regime = "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
        elif "mab_igg4" in p:
            regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
        elif any(x in p for x in ["fc-fusion", "trispecific"]):
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
        elif any(x in p for x in ["bispecific", "adc"]):
            regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.45 else "Far")
        elif any(x in p for x in ["bsa", "polyclonal"]):
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
        else:
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
        lk = next((k for k in PRIOR_TABLE if k != "default" and k in p), "default")
        rd = PRIOR_TABLE[lk].get(regime, PRIOR_TABLE[lk]["Far"])
        priors = {k: 0.0 for k in new_prior_cols}
        concs = {k: 0.0 for k in new_conc_cols}
        for tc, cc in [
            ("Salt_type", "Salt_conc"),
            ("Stabilizer_type", "Stabilizer_conc"),
            ("Excipient_type", "Excipient_conc"),
            ("Surfactant_type", "Surfactant_conc"),
        ]:
            ing = str(row.get(tc, "none")).lower()
            conc = float(row.get(cc, 0.0) or 0.0)
            if ing in ["none", "unknown", "nan"] or conc <= 0:
                continue
            if "arginine" in ing or "arg" in ing:
                priors["prior_arginine"] = rd.get("arginine", 0)
            elif "lysine" in ing or "lys" in ing:
                priors["prior_lysine"] = rd.get("lysine", 0)
            elif "proline" in ing:
                priors["prior_proline"] = rd.get("proline", 0)
            elif "nacl" in ing:
                priors["prior_nacl"] = rd.get("nacl", 0)
            elif tc == "Stabilizer_type":
                priors["prior_stabilizer"] = rd.get("stabilizer", 0)
            elif "tween" in ing or "polysorbate" in ing:
                tk = "tween-20" if "20" in ing else "tween-80"
                priors[f"prior_{tk}"] = rd.get(tk, 0)
            for ti, thr in CONC_THRESHOLDS.items():
                if ti in ing or (ti == "arginine" and "arg" in ing):
                    concs[f"{ti}_low"] = min(conc, thr)
                    concs[f"{ti}_high"] = max(conc - thr, 0)
        return {**priors, **concs}

    feats = df.apply(process_row, axis=1, result_type="expand")
    df = pd.concat([df, feats], axis=1)
    num_cols = num_cols + new_prior_cols + new_conc_cols

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    X = preprocessor.fit_transform(df)
    X = np.nan_to_num(X)

    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }

    all_pts = []
    for i in range(len(df)):
        for col, sv in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = max(df.iloc[i][col], 1e-6)
                all_pts.append([np.log10(sv), np.log10(v)])
    physics_scaler = StandardScaler().fit(np.array(all_pts))

    samples = []
    for i in range(len(df)):
        rx, ry = [], []
        for col, sv in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = max(df.iloc[i][col], 1e-6)
                rx.append(np.log10(sv))
                ry.append(np.log10(v))
        if len(rx) < 3:
            continue
        si = np.argsort(rx)
        xa, ya = np.array(rx)[si], np.array(ry)[si]
        interp = PchipInterpolator(xa, ya)
        dx = np.linspace(xa.min(), xa.max(), 50)
        dy = interp(dx)
        pts = physics_scaler.transform(np.stack([dx, dy], axis=1))
        samples.append(
            {
                "static": torch.tensor(X[i], dtype=torch.float32),
                "points": torch.tensor(pts, dtype=torch.float32),
                "group": df.iloc[i]["Protein_type"],
                "id": df.iloc[i]["ID"],
            }
        )

    return samples, X.shape[1], physics_scaler


# ============================================================
# 2. Core prediction helper
# ============================================================
def predict_from_context(model, ctx_samples, tgt_samples, device):
    """
    Given context samples and target samples (same protein group),
    predict the full viscosity curve for each target.

    Returns (pred_log_visc, true_log_visc) as numpy arrays [n_points].
    """
    model.eval()
    with torch.no_grad():
        # Build context tensor
        ctx_items = []
        for s in ctx_samples:
            stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            ctx_items.append(torch.cat([s["points"], stat], dim=1))
        ctx_tensor = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

        # Build target
        tgt_shear, tgt_stat, tgt_true = [], [], []
        for s in tgt_samples:
            n = s["points"].shape[0]
            tgt_shear.append(s["points"][:, [0]])
            tgt_true.append(s["points"][:, 1].numpy())
            tgt_stat.append(s["static"].unsqueeze(0).repeat(n, 1))

        q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
        q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
        true_y = np.concatenate(tgt_true)

        pred = model(ctx_tensor, q_x, q_stat).squeeze().cpu().numpy()

    return pred, true_y


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


# ============================================================
# 2b. Batch-building helpers (v2)
# ============================================================


def _build_ctx_tensor(task_samples, indices, device):
    """Build a context tensor [1, N_points, 2+static_dim] from sample indices."""
    ctx_items = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        ctx_items.append(torch.cat([s["points"], stat], dim=1))
    return torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)


def _build_tgt_tensors(task_samples, indices, device):
    """Build query tensors for target samples."""
    shear_list, y_list, stat_list = [], [], []
    for i in indices:
        s = task_samples[i]
        n = s["points"].shape[0]
        shear_list.append(s["points"][:, [0]])
        y_list.append(s["points"][:, [1]])
        stat_list.append(s["static"].unsqueeze(0).repeat(n, 1))
    if not shear_list:
        return None, None, None
    return (
        torch.cat(shear_list, dim=0).unsqueeze(0).to(device),
        torch.cat(stat_list, dim=0).unsqueeze(0).to(device),
        torch.cat(y_list, dim=0).unsqueeze(0).to(device),
    )


# ============================================================
# 3. STRATEGY 1 — Leave-One-Group-Out Cross-Validation
#    Trains a fresh model on all-but-one protein group,
#    evaluates on the held-out group.
#    This is the gold-standard for limited-data generalization.
# ============================================================
def logo_cv(all_samples, static_dim, model_cfg, device, n_epochs=200, iterations=80):
    """
    Full LOGO-CV.  For each unique protein group:
      - train on all other groups
      - evaluate on held-out group using 1-shot context (first sample)
        and predicting the rest
    Returns a DataFrame with per-fold results.
    """
    groups = defaultdict(list)
    for s in all_samples:
        groups[s["group"]].append(s)

    protein_groups = [g for g, samps in groups.items() if len(samps) >= 2]
    print(f"\n[LOGO-CV] {len(protein_groups)} groups with ≥2 samples")

    rows = []
    for fold_i, held_out_group in enumerate(protein_groups):
        train_samples = [s for s in all_samples if s["group"] != held_out_group]
        test_samples = groups[held_out_group]

        if len(train_samples) == 0 or len(test_samples) < 2:
            continue

        # Fresh model for this fold
        model = CrossSampleCNP(
            static_dim,
            hidden_dim=model_cfg["hidden_dim"],
            latent_dim=model_cfg["latent_dim"],
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_cfg.get("lr", 1e-3),
            weight_decay=model_cfg.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        # --- Train (v2: contrastive + consistency + utility losses) ---
        train_groups = defaultdict(list)
        for s in train_samples:
            train_groups[s["group"]].append(s)
        protein_list = [g for g, sl in train_groups.items() if len(sl) >= 4]

        model.train()
        for ep in range(n_epochs):
            for _ in range(iterations):
                if not protein_list:
                    continue

                # Sample two distinct groups for contrastive loss
                prot_A = np.random.choice(protein_list)
                other = [g for g in protein_list if g != prot_A]
                prot_B = np.random.choice(other) if other else prot_A

                task_A = train_groups[prot_A]
                task_B = train_groups[prot_B]

                idx_A = np.random.permutation(len(task_A))
                n_ctx = np.random.randint(1, min(8, len(idx_A) - 1))
                ctx_A = _build_ctx_tensor(task_A, idx_A[:n_ctx], device)
                qx, qstat, ty = _build_tgt_tensors(task_A, idx_A[n_ctx:], device)
                if qx is None:
                    continue

                pred = model(ctx_A, qx, qstat)
                mse_loss = F.mse_loss(pred, ty)

                # [v2] Context utility loss
                null_ctx = torch.zeros_like(ctx_A)
                with torch.no_grad():
                    pred_null = model(null_ctx, qx, qstat)
                mse_null = F.mse_loss(pred_null, ty).detach()
                utility_loss = torch.clamp(
                    F.mse_loss(pred, ty) - mse_null + 1e-3, min=0.0
                )

                # [v2] Contrastive loss: push r_A and r_B apart
                r_A = model.encode_memory(ctx_A)
                idx_B = np.random.permutation(len(task_B))
                ctx_B = _build_ctx_tensor(
                    task_B, idx_B[: max(1, len(idx_B) // 2)], device
                )
                r_B = model.encode_memory(ctx_B)
                cos_AB = F.cosine_similarity(r_A, r_B, dim=-1)
                contrastive_loss = torch.clamp(cos_AB + 0.3, min=0.0).mean()

                # [v2] Within-group consistency loss
                consistency_loss = torch.tensor(0.0, device=device)
                if len(task_A) >= 4:
                    split = len(idx_A) // 2
                    if split >= 1 and (len(idx_A) - split) >= 1:
                        r_A1 = model.encode_memory(
                            _build_ctx_tensor(task_A, idx_A[:split], device)
                        )
                        r_A2 = model.encode_memory(
                            _build_ctx_tensor(task_A, idx_A[split:], device)
                        )
                        consistency_loss = (
                            1.0 - F.cosine_similarity(r_A1, r_A2, dim=-1)
                        ).mean()

                loss = (
                    mse_loss
                    + 0.10 * utility_loss
                    + 0.05 * contrastive_loss
                    + 0.02 * consistency_loss
                )

                if torch.isnan(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()  # CosineAnnealingLR: no argument

        # --- Evaluate: randomized 1-shot context, averaged over repeats ---
        # Averaging over multiple random context choices removes ordering bias
        # from always using test_samples[0] as context.
        model.eval()
        if len(test_samples) < 2:
            continue
        fold_rmse_list, fold_mae_list = [], []
        n_eval_reps = min(10, len(test_samples))
        for _ in range(n_eval_reps):
            idx = np.random.permutation(len(test_samples))
            ctx_samps = [test_samples[idx[0]]]
            tgt_samps = [test_samples[i] for i in idx[1:]]
            pred, true = predict_from_context(model, ctx_samps, tgt_samps, device)
            fold_rmse_list.append(rmse(pred, true))
            fold_mae_list.append(mae(pred, true))
        rows.append(
            {
                "fold": fold_i,
                "held_out_group": held_out_group,
                "n_train": len(train_samples),
                "n_test": len(test_samples) - 1,
                "rmse_logspace": float(np.mean(fold_rmse_list)),
                "rmse_std": float(np.std(fold_rmse_list)),
                "mae_logspace": float(np.mean(fold_mae_list)),
            }
        )
        print(
            f"  Fold {fold_i:2d} | {held_out_group:30s} | RMSE={rows[-1]['rmse_logspace']:.4f} ± {rows[-1]['rmse_std']:.4f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. STRATEGY 2 — Within-Group Few-Shot Context Size Sweep
#    Using the PRE-TRAINED model (not retraining per fold).
#    Sweeps context size k=1..min(N-1,10) for each group,
#    averaging over multiple random context selections.
# ============================================================
def fewshot_sweep(model, all_samples, device, max_ctx=10, n_repeats=20):
    """
    For each protein group with ≥3 samples:
      For k in 1..min(group_size-1, max_ctx):
        Randomly draw k context samples n_repeats times,
        predict remaining samples, record RMSE.
    Returns a long-format DataFrame.
    """
    groups = defaultdict(list)
    for s in all_samples:
        groups[s["group"]].append(s)

    rows = []
    for group, samps in groups.items():
        if len(samps) < 3:
            continue
        k_max = min(len(samps) - 1, max_ctx)
        for k in range(1, k_max + 1):
            fold_rmse, fold_mae = [], []
            for _ in range(n_repeats):
                idx = np.random.permutation(len(samps))
                ctx = [samps[i] for i in idx[:k]]
                tgt = [samps[i] for i in idx[k:]]
                if not tgt:
                    continue
                pred, true = predict_from_context(model, ctx, tgt, device)
                fold_rmse.append(rmse(pred, true))
                fold_mae.append(mae(pred, true))
            if fold_rmse:
                rows.append(
                    {
                        "group": group,
                        "n_total": len(samps),
                        "context_size": k,
                        "rmse_mean": np.mean(fold_rmse),
                        "rmse_std": np.std(fold_rmse),
                        "mae_mean": np.mean(fold_mae),
                        "mae_std": np.std(fold_mae),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# 5. STRATEGY 3 — Bootstrap CI on Aggregate Metrics
#    Uses the within-group evaluation (half context, half target)
#    over B bootstrap resamples of the sample list.
# ============================================================
def bootstrap_ci(model, all_samples, device, B=1000, n_ctx_frac=0.5):
    """
    Bootstrap confidence intervals on aggregate RMSE and MAE.
    Each bootstrap resample draws groups WITH replacement,
    then evaluates using the first n_ctx_frac as context.
    """
    groups = defaultdict(list)
    for s in all_samples:
        groups[s["group"]].append(s)
    group_names = [g for g, sl in groups.items() if len(sl) >= 2]

    boot_rmse, boot_mae = [], []

    for b in range(B):
        sampled_groups = np.random.choice(
            group_names, size=len(group_names), replace=True
        )
        all_pred, all_true = [], []
        for g in sampled_groups:
            samps = groups[g]
            # Randomize which samples are context vs target each bootstrap rep
            idx = np.random.permutation(len(samps))
            split = max(1, len(idx) // 2)
            ctx = [samps[i] for i in idx[:split]]
            tgt = [samps[i] for i in idx[split:]]
            if not tgt:
                continue
            pred, true = predict_from_context(model, ctx, tgt, device)
            all_pred.append(pred)
            all_true.append(true)
        if not all_pred:
            continue
        p = np.concatenate(all_pred)
        t = np.concatenate(all_true)
        boot_rmse.append(rmse(p, t))
        boot_mae.append(mae(p, t))

    def ci(arr):
        return np.percentile(arr, 2.5), np.mean(arr), np.percentile(arr, 97.5)

    r_lo, r_mu, r_hi = ci(boot_rmse)
    m_lo, m_mu, m_hi = ci(boot_mae)

    return pd.DataFrame(
        [
            {
                "metric": "RMSE (log10 viscosity)",
                "ci_low_2.5": r_lo,
                "mean": r_mu,
                "ci_high_97.5": r_hi,
            },
            {
                "metric": "MAE (log10 viscosity)",
                "ci_low_2.5": m_lo,
                "mean": m_mu,
                "ci_high_97.5": m_hi,
            },
        ]
    )


# ============================================================
# 6. Per-Group Diagnostics
# ============================================================
def per_group_diagnostics(model, all_samples, device, n_repeats=30):
    """
    For each protein group, evaluate using random 50/50 splits
    and report mean/std RMSE, plus a shear-thinning flag
    (groups where viscosity range > 1 log decade = likely shear-thinning).
    """
    groups = defaultdict(list)
    for s in all_samples:
        groups[s["group"]].append(s)

    rows = []
    for group, samps in groups.items():
        if len(samps) < 2:
            continue

        # Shear-thinning detection: range of log-viscosity
        all_visc = torch.cat([s["points"][:, 1] for s in samps]).numpy()
        visc_range = float(all_visc.max() - all_visc.min())  # in scaled space
        is_shear_thinning = visc_range > 1.0  # heuristic in scaled units

        fold_rmse = []
        for _ in range(n_repeats):
            idx = np.random.permutation(len(samps))
            split = max(1, len(idx) // 2)
            ctx = [samps[i] for i in idx[:split]]
            tgt = [samps[i] for i in idx[split:]]
            if not tgt:
                continue
            pred, true = predict_from_context(model, ctx, tgt, device)
            fold_rmse.append(rmse(pred, true))

        rows.append(
            {
                "group": group,
                "n_samples": len(samps),
                "visc_range_scaled": round(visc_range, 3),
                "likely_shear_thinning": is_shear_thinning,
                "rmse_mean": np.mean(fold_rmse),
                "rmse_std": np.std(fold_rmse),
                "rmse_cv_pct": 100 * np.std(fold_rmse) / (np.mean(fold_rmse) + 1e-9),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse_mean", ascending=False)


# ============================================================
# 7. NEW — Latent Variance Diagnostic
#    Verifies the context-collapse fix actually worked.
#    Computes mean inter-group latent L2 distance.
#    Near-zero -> collapse still occurring.
#    Should be meaningfully > 0 for a well-trained v2 model.
# ============================================================
def latent_variance_diagnostic(model, all_samples, device, n_ctx=5, n_repeats=10):
    """
    For each protein group, encode a random subset of context samples to get r.
    Compute mean pairwise L2 distance between group latent vectors.
    Also returns per-group mean latent norm as a secondary indicator.
    A healthy v2 model should show inter-group distance >> intra-group spread.
    """
    model.eval()
    groups = defaultdict(list)
    for s in all_samples:
        groups[s["group"]].append(s)

    group_r_samples = defaultdict(list)
    with torch.no_grad():
        for prot, samps in groups.items():
            if len(samps) < 2:
                continue
            for _ in range(n_repeats):
                k = min(n_ctx, len(samps))
                idx = np.random.permutation(len(samps))[:k]
                ctx_items = []
                for i in idx:
                    s = samps[i]
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_items.append(torch.cat([s["points"], stat], dim=1))
                ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
                r = model.encode_memory(ctx_t).squeeze(0).cpu().numpy()
                group_r_samples[prot].append(r)

    # Per-group centroid and intra-group spread
    rows = []
    centroids = {}
    for prot, r_list in group_r_samples.items():
        arr = np.stack(r_list)  # [n_repeats, latent_dim]
        centroid = arr.mean(axis=0)
        centroids[prot] = centroid
        intra_spread = float(np.mean([np.linalg.norm(r - centroid) for r in arr]))
        rows.append(
            {
                "group": prot,
                "latent_norm": float(np.linalg.norm(centroid)),
                "intra_spread": round(intra_spread, 4),
            }
        )

    # Inter-group pairwise L2 distances between centroids
    group_names = list(centroids.keys())
    inter_dists = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            d = np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]])
            inter_dists.append(d)

    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    mean_intra = float(np.mean([r["intra_spread"] for r in rows]))

    df = pd.DataFrame(rows).sort_values("latent_norm", ascending=False)
    df["mean_inter_group_dist"] = mean_inter
    df["mean_intra_group_spread"] = mean_intra
    # Separation ratio: good models should have ratio >> 1
    df["separation_ratio"] = round(mean_inter / (mean_intra + 1e-9), 3)

    return df, mean_inter, mean_intra


# ============================================================
# 8. NEW — Context Utility Diagnostic
#    Measures how much context actually helps per group.
#    For each group: compare real-context RMSE vs null-context RMSE.
#    v2 training included a utility loss to enforce this;
#    this diagnostic verifies it worked in practice.
# ============================================================
def context_utility_diagnostic(model, all_samples, device, n_repeats=20):
    """
    For each group, evaluate:
      - real_rmse:  prediction using actual context samples
      - null_rmse:  prediction using a zero-filled context tensor (same shape)
      - utility_gain = (null_rmse - real_rmse) / null_rmse  (positive = context helps)
    A v1 model will show utility_gain ~ 0 everywhere (context ignored).
    A v2 model should show positive utility_gain, especially for hard groups.
    """
    model.eval()
    groups = defaultdict(list)
    for s in all_samples:
        groups[s["group"]].append(s)

    rows = []
    with torch.no_grad():
        for group, samps in groups.items():
            if len(samps) < 3:
                continue
            real_rmses, null_rmses = [], []
            for _ in range(n_repeats):
                idx = np.random.permutation(len(samps))
                split = max(1, len(idx) // 2)
                ctx_samps = [samps[i] for i in idx[:split]]
                tgt_samps = [samps[i] for i in idx[split:]]
                if not tgt_samps:
                    continue

                # Real context
                ctx_items = []
                for s in ctx_samps:
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_items.append(torch.cat([s["points"], stat], dim=1))
                ctx_real = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

                # Null context (same shape, zeroed)
                ctx_null = torch.zeros_like(ctx_real)

                # Targets
                tgt_shear, tgt_stat, tgt_true = [], [], []
                for s in tgt_samps:
                    n = s["points"].shape[0]
                    tgt_shear.append(s["points"][:, [0]])
                    tgt_true.append(s["points"][:, 1].numpy())
                    tgt_stat.append(s["static"].unsqueeze(0).repeat(n, 1))
                q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
                q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
                true_y = np.concatenate(tgt_true)

                pred_real = model(ctx_real, q_x, q_stat).squeeze().cpu().numpy()
                pred_null = model(ctx_null, q_x, q_stat).squeeze().cpu().numpy()

                real_rmses.append(rmse(pred_real, true_y))
                null_rmses.append(rmse(pred_null, true_y))

            if real_rmses:
                r_mean = float(np.mean(real_rmses))
                n_mean = float(np.mean(null_rmses))
                rows.append(
                    {
                        "group": group,
                        "n_samples": len(samps),
                        "real_rmse": round(r_mean, 4),
                        "null_rmse": round(n_mean, 4),
                        # Positive = context helps; negative = context hurts (collapse sign)
                        "utility_gain": round((n_mean - r_mean) / (n_mean + 1e-9), 4),
                    }
                )

    df = pd.DataFrame(rows).sort_values("utility_gain")
    return df


# ============================================================
# 9. Main
# ============================================================
def main():
    out = "./eval_results_v2"
    data = "data/raw/formulation_data_02162026.csv"
    model = "models/experiments/o_net_v2/best_model.pth"
    skip_logo = True
    logo_epochs = 150
    bootstrap_n = 1000

    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Preprocessing data...")
    samples, static_dim, physics_scaler = load_and_preprocess(data)
    print(f"  {len(samples)} samples | {len(set(s['group'] for s in samples))} groups")

    # --- Load trained model ---
    print(f"Loading model from {model} ...")
    ckpt = torch.load(model, map_location=device)
    cfg = ckpt["config"]
    model = CrossSampleCNP(
        static_dim=ckpt["static_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    lines = []  # for report

    # ----------------------------------------------------------------
    # Strategy 1: LOGO-CV
    # ----------------------------------------------------------------
    if not skip_logo:
        print("\n====== STRATEGY 1: Leave-One-Group-Out CV ======")
        logo_df = logo_cv(samples, static_dim, cfg, device, n_epochs=logo_epochs)
        logo_df.to_csv(os.path.join(out, "logo_results.csv"), index=False)

        lines += [
            "=" * 60,
            "LEAVE-ONE-GROUP-OUT CV SUMMARY",
            "=" * 60,
            f"  Groups evaluated: {len(logo_df)}",
            f"  Aggregate RMSE (log-space):  {logo_df['rmse_logspace'].mean():.4f} ± {logo_df['rmse_logspace'].std():.4f}",
            f"  Aggregate MAE  (log-space):  {logo_df['mae_logspace'].mean():.4f} ± {logo_df['mae_logspace'].std():.4f}",
            f"  Worst group:  {logo_df.loc[logo_df['rmse_logspace'].idxmax(), 'held_out_group']} "
            f"(RMSE={logo_df['rmse_logspace'].max():.4f})",
            f"  Best group:   {logo_df.loc[logo_df['rmse_logspace'].idxmin(), 'held_out_group']} "
            f"(RMSE={logo_df['rmse_logspace'].min():.4f})",
            "",
            "  NOTE: RMSE is in log10(viscosity) space scaled by physics_scaler.",
            "  Approx. 1 unit ~ 1 order of magnitude viscosity error.",
        ]
    else:
        print("\n  [Skipping LOGO-CV — pass without --skip_logo to enable]")

    # ----------------------------------------------------------------
    # Strategy 2: Few-Shot Sweep (uses pre-trained model, fast)
    # ----------------------------------------------------------------
    print("\n====== STRATEGY 2: Few-Shot Context Size Sweep ======")
    sweep_df = fewshot_sweep(model, samples, device, max_ctx=10, n_repeats=30)
    sweep_df.to_csv(os.path.join(out, "fewshot_sweep.csv"), index=False)

    # Aggregate: RMSE vs context size across all groups
    agg = (
        sweep_df.groupby("context_size")[["rmse_mean", "mae_mean"]].mean().reset_index()
    )
    lines += [
        "",
        "=" * 60,
        "FEW-SHOT CONTEXT SIZE SWEEP (pre-trained model)",
        "=" * 60,
        "  Context Size | Avg RMSE | Avg MAE",
        "  " + "-" * 40,
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"  k={int(row['context_size']):2d}           | {row['rmse_mean']:.4f}   | {row['mae_mean']:.4f}"
        )
    lines += [
        "",
        "  Interpretation: Look for the 'elbow' where adding more context",
        "  gives diminishing RMSE returns — that is your minimum viable",
        "  context size for deployment.",
    ]

    # ----------------------------------------------------------------
    # Strategy 3: Bootstrap CI
    # ----------------------------------------------------------------
    print("\n====== STRATEGY 3: Bootstrap CI ======")
    ci_df = bootstrap_ci(model, samples, device, B=bootstrap_n)
    ci_df.to_csv(os.path.join(out, "bootstrap_ci.csv"), index=False)

    lines += [
        "",
        "=" * 60,
        f"BOOTSTRAP CI (B={bootstrap_n}, group-level resampling)",
        "=" * 60,
    ]
    for _, row in ci_df.iterrows():
        lines.append(
            f"  {row['metric']:30s}: {row['mean']:.4f}  "
            f"[95% CI: {row['ci_low_2.5']:.4f} – {row['ci_high_97.5']:.4f}]"
        )
    lines += [
        "",
        "  NOTE: CI width reflects evaluation uncertainty due to limited data,",
        "  not just model variance. Wide CI = your evaluation itself is noisy.",
    ]

    # ----------------------------------------------------------------
    # Per-Group Diagnostics
    # ----------------------------------------------------------------
    print("\n====== Per-Group Diagnostics ======")
    diag_df = per_group_diagnostics(model, samples, device)
    diag_df.to_csv(os.path.join(out, "per_group_diagnostics.csv"), index=False)

    top5_worst = diag_df.head(5)
    lines += [
        "",
        "=" * 60,
        "PER-GROUP DIAGNOSTICS (sorted by RMSE desc)",
        "=" * 60,
        "  Top 5 hardest groups:",
    ]
    for _, row in top5_worst.iterrows():
        st = " [SHEAR-THINNING]" if row["likely_shear_thinning"] else ""
        lines.append(
            f"    {row['group']:30s}  n={int(row['n_samples']):3d}  "
            f"RMSE={row['rmse_mean']:.4f}±{row['rmse_std']:.4f}{st}"
        )

    # ----------------------------------------------------------------
    # NEW: Latent Variance Diagnostic (v2 context-collapse check)
    # ----------------------------------------------------------------
    print("\n====== Latent Variance Diagnostic ======")
    latent_df, mean_inter, mean_intra = latent_variance_diagnostic(
        model, samples, device
    )
    latent_df.to_csv(os.path.join(out, "latent_variance.csv"), index=False)
    sep_ratio = mean_inter / (mean_intra + 1e-9)

    lines += [
        "",
        "=" * 60,
        "LATENT VARIANCE DIAGNOSTIC (v2 context-collapse check)",
        "=" * 60,
        f"  Mean inter-group latent L2 distance : {mean_inter:.4f}",
        f"  Mean intra-group latent spread      : {mean_intra:.4f}",
        f"  Separation ratio (inter/intra)      : {sep_ratio:.2f}",
        "",
        "  Interpretation:",
        "  - Ratio >> 1  -> context is encoding group-discriminative info (GOOD)",
        "  - Ratio ~ 1   -> latents overlap; partial collapse still present",
        "  - Ratio < 1   -> context collapse: r is noise, not signal (BAD)",
        "",
    ]
    if sep_ratio < 1.5:
        lines.append(
            "  *** WARNING: Low separation ratio suggests context collapse "
            "may still be present. Consider increasing lambda_contrastive "
            "or checking that multiple protein groups reach train_epoch. ***"
        )
    else:
        lines.append(" Separation ratio indicates healthy latent structure.")

    # ----------------------------------------------------------------
    # NEW: Context Utility Diagnostic (v2 utility-loss verification)
    # ----------------------------------------------------------------
    print("\n====== Context Utility Diagnostic ======")
    utility_df = context_utility_diagnostic(model, samples, device)
    utility_df.to_csv(os.path.join(out, "context_utility.csv"), index=False)

    n_negative = (utility_df["utility_gain"] < 0).sum()
    mean_gain = utility_df["utility_gain"].mean()
    worst_util = utility_df.iloc[0]  # sorted ascending, so lowest gain first

    lines += [
        "",
        "=" * 60,
        "CONTEXT UTILITY DIAGNOSTIC (real vs null context RMSE)",
        "=" * 60,
        f"  Mean utility gain across groups  : {mean_gain:.4f}",
        f"  Groups where context HURTS (< 0) : {n_negative} / {len(utility_df)}",
        "",
        "  Groups with lowest context utility (most likely to still be ignoring context):",
    ]
    for _, row in utility_df.head(5).iterrows():
        flag = " <- CONTEXT IGNORED" if row["utility_gain"] < 0.01 else ""
        lines.append(
            f"    {row['group']:30s}  real={row['real_rmse']:.4f}  "
            f"null={row['null_rmse']:.4f}  gain={row['utility_gain']:+.4f}{flag}"
        )
    lines += [
        "",
        "  Interpretation:",
        "  - utility_gain > 0  -> context reduces error (model uses context)",
        "  - utility_gain ~ 0  -> context has no effect (collapse symptom)",
        "  - utility_gain < 0  -> context increases error (degenerate case)",
    ]

    # ----------------------------------------------------------------
    # Write report
    # ----------------------------------------------------------------
    report_path = os.path.join(out, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nAll results saved to: {out}")


if __name__ == "__main__":
    main()
