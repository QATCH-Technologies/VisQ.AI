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
from train_o_net import CrossSampleCNP, load_and_preprocess

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

# --- UPDATED SCENARIOS (Control + Arginine 1-100mM) ---
ARG_CONCS = [1.0, 25.0, 50.0, 75.0, 100.0]

# 1. Baseline Control
control_scenario = [("Control (Buffer Only)", {})]

# 2. Arginine Scenarios
arg_scenarios = [
    (
        f"Arginine ({int(c)}mM)",
        {"Excipient_type": "Arginine", "Excipient_conc": float(c)},
    )
    for c in ARG_CONCS
]

# Combine: Control first, then increasing amounts of Arginine
SCENARIOS = control_scenario + arg_scenarios

# --- CONCENTRATIONS (4 Levels) ---
CONCENTRATIONS = [120.0, 150.0, 180.0, 210.0]

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


# ==========================================
# 2. Calibration Wrapper
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
        """
        print(f" > Calibrating on {len(samples)} samples for {steps} steps...")
        self.model.train()

        # Build Batch from Samples
        # Stack all points from all samples into one big context
        shear_list, visc_list, static_list = [], [], []

        for s in samples:
            # s['points'] is [N, 2] -> shear, visc
            shear_list.append(s["points"][:, [0]])
            visc_list.append(s["points"][:, [1]])

            # s['static'] is [StaticDim] -> Expand to [N, StaticDim]
            stat_expanded = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            static_list.append(stat_expanded)

        shear_t = torch.cat(shear_list, dim=0).to(DEVICE)
        visc_t = torch.cat(visc_list, dim=0).to(DEVICE)
        static_t = torch.cat(static_list, dim=0).to(DEVICE)

        # Add Batch Dim [1, TotalPoints, Dim]
        shear_t = shear_t.unsqueeze(0)
        visc_t = visc_t.unsqueeze(0)
        static_t = static_t.unsqueeze(0)

        # Context [1, TotalPoints, Shear+Visc+Static]
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)

        # Optimization Loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(steps):
            # Forward: Model predicts based on context (Self-Consistency)
            # We predict the same points we are looking at to learn their physics
            pred = self.model(context_t, shear_t, static_t)

            loss = F.mse_loss(pred, visc_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f" > Calibration complete. Final MSE: {loss.item():.5f}")

        # Cache the memory for inference
        self.model.eval()
        with torch.no_grad():
            self.memory_vector = self.model.encode_memory(context_t)

    def predict(self, static_vec, shear_rates):
        """
        Predicts using the cached memory (calibrated state).
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
            # Use decode_from_memory with cached vector
            y_scaled = self.model.decode_from_memory(
                self.memory_vector, q_shear, q_static
            )

        # Inverse Transform
        preds = []
        for j, val in enumerate(y_scaled.view(-1).cpu().numpy()):
            scaled_shear = q_list[j]
            inv_inp = np.array([[scaled_shear, val]])
            inv_out = self.physics_scaler.inverse_transform(inv_inp)[0]
            preds.append(10 ** inv_out[1])

        return preds


# ==========================================
# 3. Main Logic
# ==========================================
def process_row_physics(row):
    """(Same physics logic as before)"""
    c_class = row.get("C_Class", 1.0)
    ph = row.get("Buffer_pH", 7.0)
    pi = row.get("PI_mean", 7.0)
    cci = c_class * np.exp(-abs(ph - pi) / 1.5)
    p_type = str(row.get("Protein_class_type", "default")).lower()

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

    scan_cols = [
        ("Salt_type", "Salt_conc"),
        ("Stabilizer_type", "Stabilizer_conc"),
        ("Excipient_type", "Excipient_conc"),
        ("Surfactant_type", "Surfactant_conc"),
    ]
    for type_col, conc_col in scan_cols:
        ing_name = str(row.get(type_col, "none")).lower()
        ing_conc = float(row.get(conc_col, 0.0))
        if ing_name in ["none", "nan"] or ing_conc <= 0:
            continue

        if "arginine" in ing_name:
            priors["prior_arginine"] = regime_dict.get("arginine", 0)
        elif "lysine" in ing_name:
            priors["prior_lysine"] = regime_dict.get("lysine", 0)
        elif "proline" in ing_name:
            priors["prior_proline"] = regime_dict.get("proline", 0)
        elif "sucrose" in ing_name:
            priors["prior_stabilizer"] = regime_dict.get("stabilizer", 0)

        for target_ing, threshold in CONC_THRESHOLDS.items():
            if target_ing in ing_name:
                concs[f"{target_ing}_low"] = min(ing_conc, threshold)
                concs[f"{target_ing}_high"] = max(ing_conc - threshold, 0)
    return {**priors, **concs}


def run():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Everything
    print("Loading model and data...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    preprocessor = joblib.load(PREPROC_PATH)
    physics_scaler = joblib.load(SCALER_PATH)

    # Load raw training data to find calibration samples
    full_samples, _ = load_and_preprocess(TRAIN_DATA_PATH)
    # Filter for Target Protein
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
    print("Constructing Arginine scan scenarios (with Baseline)...")
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
        preds = calibrator.predict(static_vec, SHEAR_RATES)

        res_row = proc_df.iloc[i].to_dict()
        for k, sr in enumerate(SHEAR_RATES):
            res_row[f"Pred_Visc_{int(sr)}"] = preds[k]
        results.append(res_row)

    # 5. Save & Plot
    res_df = pd.DataFrame(results)
    res_df.to_csv(
        os.path.join(OUTPUT_DIR, "pembrolizumab_arginine_scan.csv"),
        index=False,
    )

    sns.set_context("talk")
    sns.set_style("ticks")

    print("Generating plots with annotations...")
    for conc in CONCENTRATIONS:
        plt.figure(figsize=(10, 7))
        subset = res_df[res_df["Protein_conc"] == conc]
        # Use viridis for sequential formulation changes
        # Control will be mapped to the first color (dark), High Arginine to last (light)
        colors = sns.color_palette("viridis", len(subset))

        for idx, (_, row) in enumerate(subset.iterrows()):
            y = [row[f"Pred_Visc_{int(sr)}"] for sr in SHEAR_RATES]

            # Plot line
            plt.plot(
                SHEAR_RATES,
                y,
                marker="o",
                markersize=6,
                linewidth=2.5,
                label=row["Scenario_Name"],
                color=colors[idx],
            )

            # Annotate the final point (High Shear Viscosity)
            final_x = SHEAR_RATES[-1]
            final_y = y[-1]
            plt.annotate(
                f"{final_y:.1f} cP",
                xy=(final_x, final_y),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color=colors[idx],
                fontweight="bold",
            )

        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Pretty Ticks
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        locmin = ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        plt.xlabel("Shear Rate (1/s)", fontweight="bold")
        plt.ylabel("Viscosity (cP)", fontweight="bold")
        plt.title(
            f"Pembrolizumab (Calibrated) @ {int(conc)} mg/mL\nArginine Scan (0-100 mM)",
            fontsize=14,
            pad=15,
        )

        plt.grid(True, which="major", ls="-", alpha=0.5)
        plt.grid(True, which="minor", ls=":", alpha=0.2)

        plt.legend(bbox_to_anchor=(1.10, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f"arginine_scan_{int(conc)}mg_ml.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    run()
