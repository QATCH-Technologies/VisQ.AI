import os

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

# Import model class from your training script
# Ensure train_o_net.py is in the same directory and has the updated CrossSampleCNP class
from train_o_net_no_phys import CrossSampleCNP, load_and_preprocess

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = "models/experiments/o_net/best_model.pth"
PREPROC_PATH = "models/experiments/o_net/preprocessor.pkl"
SCALER_PATH = "models/experiments/o_net/physics_scaler.pkl"
TRAIN_DATA_PATH = "data/raw/formulation_data_02052026.csv"
OUTPUT_DIR = "models/experiments/o_net/scenarios_calibrated"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target Shear Rates (1/s)
SHEAR_RATES = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]
TARGET_PROTEIN_NAME = "pembrolizumab"

# Base Constants for Pembrolizumab
CONSTANTS = {
    "Protein_type": "Pembrolizumab",
    "Protein_class_type": "mAb_IgG4",
    "kP": 3.5,
    "MW": 149.0,
    "PI_mean": 7.57,
    "PI_range": 0.3,
    "Temperature": 25.0,
    "Buffer_type": "Histidine",
    "Buffer_pH": 6.0,
    "Buffer_conc": 15.0,
    "C_Class": 1.3,
    "HCI": 1.1,
}

# Formulation Scenarios
SCENARIOS = [
    ("Control (Buffer Only)", {}),
    ("Arginine (25mM)", {"Excipient_type": "Arginine", "Excipient_conc": 25.0}),
    ("Proline (25mM)", {"Excipient_type": "Proline", "Excipient_conc": 25.0}),
    ("Lysine (25mM)", {"Excipient_type": "Lysine", "Excipient_conc": 25.0}),
    ("Sucrose (0.2M)", {"Stabilizer_type": "Sucrose", "Stabilizer_conc": 0.2}),
]

CONCENTRATIONS = [130.0, 200.0]

# Physics Config (Must match training)
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
    # ... (Include other keys if necessary, or just rely on 'default') ...
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


# ==========================================
# 2. Calibration Wrapper (UPDATED)
# ==========================================
class Calibrator:
    def __init__(self, model, preprocessor, physics_scaler):
        self.model = model
        self.preprocessor = preprocessor
        self.physics_scaler = physics_scaler
        self.memory_vector = None  # Stores the calibrated context

    def learn(self, samples, steps=50, lr=1e-3):
        """
        Fine-tunes the model on the provided samples.
        UPDATED: Now handles 'physics_proxy'.
        """
        print(f" > Calibrating on {len(samples)} samples for {steps} steps...")
        self.model.train()

        # Build Batch from Samples
        shear_list, visc_list, static_list, phys_list = [], [], [], []

        for s in samples:
            # s['points'] is [N, 2] -> shear, visc
            n_pts = s["points"].shape[0]
            shear_list.append(s["points"][:, [0]])
            visc_list.append(s["points"][:, [1]])

            # s['static'] is [StaticDim] -> Expand to [N, StaticDim]
            stat_expanded = s["static"].unsqueeze(0).repeat(n_pts, 1)
            static_list.append(stat_expanded)

            # s['physics_proxy'] -> Expand to [N, 1]
            # Ensure it's a tensor
            p_val = s["physics_proxy"]
            if not isinstance(p_val, torch.Tensor):
                p_val = torch.tensor(p_val, dtype=torch.float32)
            p_val = p_val.to(DEVICE)

            if p_val.dim() == 0:
                p_val = p_val.unsqueeze(0)
            phys_list.append(p_val.unsqueeze(0).repeat(n_pts, 1))

        shear_t = torch.cat(shear_list, dim=0).to(DEVICE)
        visc_t = torch.cat(visc_list, dim=0).to(DEVICE)
        static_t = torch.cat(static_list, dim=0).to(DEVICE)
        phys_t = torch.cat(phys_list, dim=0).to(DEVICE)

        # Add Batch Dim [1, TotalPoints, Dim]
        shear_t = shear_t.unsqueeze(0)
        visc_t = visc_t.unsqueeze(0)
        static_t = static_t.unsqueeze(0)
        phys_t = phys_t.unsqueeze(0)

        # Context [1, TotalPoints, Shear+Visc+Static]
        # Note: Physics proxy is NOT part of the encoder context in your architecture,
        # it is only used in the decoder (forward).
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)

        # Optimization Loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(steps):
            # Forward: Model predicts based on context (Self-Consistency)
            # UPDATED: Pass physics proxy to forward
            pred = self.model(context_t, shear_t, static_t, phys_t)

            loss = F.mse_loss(pred, visc_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f" > Calibration complete. Final MSE: {loss.item():.5f}")

        # Cache the memory for inference
        self.model.eval()
        with torch.no_grad():
            self.memory_vector = self.model.encode_memory(context_t)

    def predict(self, static_vec, shear_rates, physics_proxy_val):
        """
        Predicts using the cached memory (calibrated state).
        UPDATED: Manually applies physics bias to ensure compatibility.
        """
        # Prepare Query
        q_list = []
        for sr in shear_rates:
            raw = np.array([[np.log10(sr), 0.0]])
            scaled = self.physics_scaler.transform(raw)[0][0]
            q_list.append(scaled)

        q_shear = torch.tensor(q_list, dtype=torch.float32).view(1, -1, 1).to(DEVICE)
        q_static = (
            static_vec.unsqueeze(0)
            .unsqueeze(1)
            .repeat(1, len(shear_rates), 1)
            .to(DEVICE)
        )

        with torch.no_grad():
            # 1. Decode using the NN (latent path)
            # We assume decode_from_memory might NOT have been updated in train_o_net.py
            # so we manually implement the decoder + bias logic here to be safe.

            # Expand memory [1, n_queries, latent]
            n_queries = q_shear.size(1)
            r_expanded = self.memory_vector.unsqueeze(1).repeat(1, n_queries, 1)

            # Concat [Shear, Static, Memory]
            decoder_input = torch.cat([q_shear, q_static, r_expanded], dim=-1)

            # Raw NN prediction
            nn_out = self.model.decoder(decoder_input)

            # 2. Apply Physics Bias (Residual)
            # proxy * softplus(beta)
            beta = F.softplus(self.model.physics_scale)
            proxy_tensor = torch.tensor(physics_proxy_val, dtype=torch.float32).to(
                DEVICE
            )
            bias = beta * proxy_tensor

            y_scaled = nn_out + bias

        # Inverse Transform
        preds = []
        for j, val in enumerate(y_scaled.view(-1).cpu().numpy()):
            scaled_shear = q_list[j]
            inv_inp = np.array([[scaled_shear, val]])
            inv_out = self.physics_scaler.inverse_transform(inv_inp)[0]
            preds.append(10 ** inv_out[1])

        return preds


# ==========================================
# 3. Physics Logic (UPDATED)
# ==========================================
def process_row_physics(row):
    """
    Calculates Priors, Concentrations, AND the new Physics Proxy.
    """
    c_class = row.get("C_Class", 1.0)
    ph = row.get("Buffer_pH", 7.0)
    pi = row.get("PI_mean", 7.0)

    # Simple CCI estimate
    cci = c_class * np.exp(-abs(ph - pi) / 1.5)
    p_type = str(row.get("Protein_class_type", "default")).lower()

    # Regime Logic
    regime = "Far"
    if "mab_igg4" in p_type:
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
    else:
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")

    lookup_key = "default"
    for key in PRIOR_TABLE.keys():
        if key != "default" and key in p_type:
            lookup_key = key
            break
    table = PRIOR_TABLE[lookup_key]
    regime_dict = table.get(regime, table["Far"])

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
        new_conc_cols.extend([f"{k}_low", f"{k}_high"])

    priors = {k: 0.0 for k in new_prior_cols}
    concs = {k: 0.0 for k in new_conc_cols}

    # --- NEW: Physics Proxy Calculation ---
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
        if ing_name in ["none", "nan", "unknown"] or ing_conc <= 0:
            continue

        weight = 0
        # Determine weight and Priors
        if "arginine" in ing_name:
            weight = regime_dict.get("arginine", 0)
            priors["prior_arginine"] = weight
        elif "lysine" in ing_name:
            weight = regime_dict.get("lysine", 0)
            priors["prior_lysine"] = weight
        elif "proline" in ing_name:
            weight = regime_dict.get("proline", 0)
            priors["prior_proline"] = weight
        elif "sucrose" in ing_name or "trehalose" in ing_name:
            weight = regime_dict.get("stabilizer", 0)
            priors["prior_stabilizer"] = weight
        elif "nacl" in ing_name:
            weight = regime_dict.get("nacl", 0)
            priors["prior_nacl"] = weight
        elif "tween" in ing_name or "polysorbate" in ing_name:
            t_key = "tween-20" if "20" in ing_name else "tween-80"
            weight = regime_dict.get(t_key, 0)
            priors[f"prior_{t_key}"] = weight

        # Add to proxy
        physics_proxy += weight * ing_conc

        # Concentration splits
        for target_ing, threshold in CONC_THRESHOLDS.items():
            if target_ing in ing_name or (
                target_ing == "arginine" and "arg" in ing_name
            ):
                concs[f"{target_ing}_low"] = min(ing_conc, threshold)
                concs[f"{target_ing}_high"] = max(ing_conc - threshold, 0)

    return {**priors, **concs, "physics_proxy": physics_proxy}


# ==========================================
# 4. Main Execution
# ==========================================
def run():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Everything
    print("Loading model and data...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    preprocessor = joblib.load(PREPROC_PATH)
    physics_scaler = joblib.load(SCALER_PATH)

    # Load raw training data
    # Note: load_and_preprocess from updated train_o_net should now return samples WITH 'physics_proxy'
    full_samples, _ = load_and_preprocess(TRAIN_DATA_PATH)

    calibration_samples = [
        s
        for s in full_samples
        if TARGET_PROTEIN_NAME.lower() in str(s["group"]).lower()
    ]

    if not calibration_samples:
        print(
            f"ERROR: No samples found for protein '{TARGET_PROTEIN_NAME}' in training data."
        )
        return

    print(
        f"Found {len(calibration_samples)} historical samples for {TARGET_PROTEIN_NAME}."
    )

    # 2. Init Model & Calibrate
    model = CrossSampleCNP(
        checkpoint["static_dim"],
        checkpoint["config"]["hidden_dim"],
        checkpoint["config"]["latent_dim"],
        checkpoint["config"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    calibrator = Calibrator(model, preprocessor, physics_scaler)

    # RUN FINE-TUNING
    calibrator.learn(calibration_samples, steps=100, lr=5e-4)

    # 3. Build Scenarios
    print("Constructing scenarios...")
    rows = []
    for conc in CONCENTRATIONS:
        for name, changes in SCENARIOS:
            r = CONSTANTS.copy()
            r.update(
                {
                    "Salt_type": "none",
                    "Salt_conc": 0.0,
                    "Stabilizer_type": "none",
                    "Stabilizer_conc": 0.0,
                    "Surfactant_type": "none",
                    "Surfactant_conc": 0.0,
                    "Excipient_type": "none",
                    "Excipient_conc": 0.0,
                }
            )
            r["Protein_conc"] = conc
            r.update(changes)
            r["Scenario_Name"] = name
            rows.append(r)

    raw_df = pd.DataFrame(rows)
    # This now generates 'physics_proxy' column
    feat_df = raw_df.apply(process_row_physics, axis=1, result_type="expand")
    proc_df = pd.concat([raw_df, feat_df], axis=1)

    cat_cols = [
        "Protein_type",
        "Protein_class_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    for c in cat_cols:
        proc_df[c] = proc_df[c].astype(str).str.lower()

    # 4. Predict using Calibrated Model
    print("Predicting viscosity profiles...")
    X_mat = preprocessor.transform(proc_df)
    results = []

    for i in range(len(proc_df)):
        static_vec = torch.tensor(X_mat[i], dtype=torch.float32).to(DEVICE)

        # Extract the physics proxy computed in process_row_physics
        proxy_val = proc_df.iloc[i]["physics_proxy"]

        # Pass proxy to predict
        preds = calibrator.predict(static_vec, SHEAR_RATES, proxy_val)

        res_row = proc_df.iloc[i].to_dict()
        for k, sr in enumerate(SHEAR_RATES):
            res_row[f"Pred_Visc_{int(sr)}"] = preds[k]
        results.append(res_row)

    # 5. Save & Plot
    res_df = pd.DataFrame(results)
    res_df.to_csv(
        os.path.join(OUTPUT_DIR, "pembrolizumab_calibrated_predictions.csv"),
        index=False,
    )

    sns.set_context("talk")
    sns.set_style("ticks")

    for conc in CONCENTRATIONS:
        plt.figure(figsize=(10, 7))
        subset = res_df[res_df["Protein_conc"] == conc]
        colors = sns.color_palette("husl", len(subset))

        for idx, (_, row) in enumerate(subset.iterrows()):
            y = [row[f"Pred_Visc_{int(sr)}"] for sr in SHEAR_RATES]
            plt.plot(
                SHEAR_RATES,
                y,
                marker="o",
                markersize=8,
                linewidth=2.5,
                label=row["Scenario_Name"],
                color=colors[idx],
            )

        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        plt.xlabel("Shear Rate (1/s)", fontweight="bold")
        plt.ylabel("Viscosity (cP)", fontweight="bold")
        plt.title(
            f"Pembrolizumab (Calibrated) @ {int(conc)} mg/mL\nFormulation Screen (Physics-Informed)",
            fontsize=14,
            pad=15,
        )

        plt.grid(True, which="major", ls="-", alpha=0.5)
        plt.grid(True, which="minor", ls=":", alpha=0.2)

        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"calibrated_plot_{int(conc)}mg_ml.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    run()
