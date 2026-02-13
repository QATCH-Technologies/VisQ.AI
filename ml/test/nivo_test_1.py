import os

import pandas as pd
import torch

# Import the predictor class from your provided script
from inference_o_net import ViscosityPredictorCNP

# ==========================================
# 1. Configuration
# ==========================================
# Update this path to where your model files are located
MODEL_DIR = "models/experiments/o_net_no_nivo"
DATA_FILE = "data/raw/formulation_data_02122026.csv"
OUTPUT_FILE = "nivolumab_predictions.xlsx"

# Nivolumab Constants (Extracted from formulation_data_02122026.csv)
# Based on rows F488-F504
NIVO_CONSTANTS = {
    "Protein_type": "Nivolumab",
    "Protein_class_type": "mAb_IgG4",
    "kP": 3.5,
    "MW": 146.0,
    "PI_mean": 8.8,
    "PI_range": 0.3,
    "C_Class": 1.3,
    "HCI": 1.1,
    "Temperature": 25.0,
}

# Scenario Definition
# Histidine (15mM), NaCl 70mM, Tween-80 0.05%, Sucrose 0.2M
TARGET_CONCS = [120.0, 180.0, 240.0, 300.0]


def create_scenario_df(concs):
    rows = []
    for c in concs:
        row = NIVO_CONSTANTS.copy()
        row.update(
            {
                # Formulation Scenarios
                "Buffer_type": "Histidine",
                "Buffer_pH": 6.0,
                "Buffer_conc": 15.0,
                "Salt_type": "NaCl",
                "Salt_conc": 70.0,
                "Surfactant_type": "tween-80",
                "Surfactant_conc": 0.05,
                "Stabilizer_type": "Sucrose",
                "Stabilizer_conc": 0.2,  # 0.2 Molar
                "Excipient_type": "Lysine",
                "Excipient_conc": 25.0,
                "Protein_conc": c,
                "ID": f"Nivo_{int(c)}mg",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    print("--- Initializing Nivolumab Scenario Test ---")

    # 1. Setup Data
    scenario_df = create_scenario_df(TARGET_CONCS)
    print(f"Scenario Created: {len(scenario_df)} samples (120-300 mg/mL)")

    # 2. Initialize Predictor
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory '{MODEL_DIR}' not found.")
        return

    predictor = ViscosityPredictorCNP(MODEL_DIR)

    # 3. Mode A: Zero-Shot Prediction (No prior knowledge of Nivolumab)
    print("\n[Mode 1] Running Zero-Shot Prediction...")
    # Ensure memory is clear/default for zero-shot
    predictor.memory_vector = None

    zs_results = predictor.predict(scenario_df)
    zs_results.insert(0, "Mode", "Zero-Shot")

    # 4. Mode B: In-Context Learning (ICL)
    print("\n[Mode 2] performing In-Context Learning (ICL)...")

    if os.path.exists(DATA_FILE):
        full_df = pd.read_csv(DATA_FILE)
        # Filter for existing Nivolumab data to learn from
        nivo_train_df = full_df[full_df["Protein_type"] == "Nivolumab"].copy()

        if not nivo_train_df.empty:
            print(
                f" > Found {len(nivo_train_df)} historical Nivolumab samples. Learning..."
            )
            input(nivo_train_df)
            predictor.learn(nivo_train_df, steps=50, lr=1e-3)

            print(" > Running ICL Prediction...")
            icl_results = predictor.predict(scenario_df)
            icl_results.insert(0, "Mode", "ICL (Calibrated)")
        else:
            print("WARNING: No Nivolumab data found in CSV for ICL.")
            icl_results = pd.DataFrame()
    else:
        print(f"WARNING: Data file '{DATA_FILE}' not found.")
        icl_results = pd.DataFrame()

    # 5. Combine and Output
    final_cols = [
        "Mode",
        "ID",
        "Protein_conc",
        "Buffer_type",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
        "Excipient_conc",
    ]
    pred_cols = [c for c in zs_results.columns if "Pred_" in c]

    # Select clean columns for output
    final_zs = zs_results[final_cols + pred_cols]
    if not icl_results.empty:
        final_icl = icl_results[final_cols + pred_cols]
        combined_df = pd.concat([final_zs, final_icl], ignore_index=True)
    else:
        combined_df = final_zs

    print(f"\nSaving results to {OUTPUT_FILE}...")
    combined_df.to_excel(OUTPUT_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    run()
