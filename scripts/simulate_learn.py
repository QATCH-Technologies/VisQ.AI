import os
from io import StringIO

import numpy as np
import pandas as pd
from src.inference import ViscosityPredictor


def run_adaptive_cycle():
    # ==========================================
    # 1. Setup Data & Targets
    # ==========================================
    csv_data = """ID,Protein_type,Buffer_type,Stabilizer_type,Surfactant_type,Salt_type,Excipient_type,Protein_conc,Temperature,Buffer_conc,Stabilizer_conc,Surfactant_conc,Salt_conc,Excipient_conc,Protein_class_type,kP,MW,PI_mean,PI_range,C_Class,HCI,Buffer_pH
0,UNKOWN_PROTEIN,Histidine,Sucrose,Tween-20,NaCl,Arginine,122.60439514440365,30.791177985586863,6.627802787526979,0.06052639418812722,0.3811645304224701,160.97510593452978,14.331617880578822,polyclonal,3.0,150.0,7.6,1.0,0.9,0.9,6.0"""

    df = pd.read_csv(StringIO(csv_data))

    # Define dummy targets (Ground Truth) for the learning step
    target_cols = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]
    ground_truth_values = [15.5, 12.2, 8.4, 5.1, 2.3]  # Example linear viscosity values

    for col, val in zip(target_cols, ground_truth_values):
        df[col] = val

    # Split into features and targets
    y_new = df[target_cols].values
    df_features = df.drop(columns=target_cols)

    # ==========================================
    # 2. Initialize & Adapt (Learn)
    # ==========================================
    checkpoint_path = "models/experiments/20260120_152300/model_0.pt"
    save_path = "models/experiments/20260120_152300/model_0_adapted.pt"

    print(f"--- 1. Initializing from {checkpoint_path} ---")
    try:
        # Create predictor instance
        vp = ViscosityPredictor(checkpoint_path, device="cpu")
        vp.hydrate()

        print(f"\n--- 2. Learning from new sample (Epochs=100) ---")
        # Running fewer epochs for demonstration speed
        vp.learn(df_features, y_new, epochs=100, lr=1e-2)
        print("Adaptation complete.")

        # ==========================================
        # 3. Predict with In-Memory Adapted Model
        # ==========================================
        print(f"\n--- 3. Prediction Check (In-Memory) ---")
        preds_memory = vp.predict(df_features)

        print(f"{'Target':<20} | {'Truth':<10} | {'Pred':<10} | {'Diff':<10}")
        print("-" * 60)
        for i, col in enumerate(target_cols):
            diff = abs(preds_memory[0][i] - ground_truth_values[i])
            print(
                f"{col:<20} | {ground_truth_values[i]:<10.4f} | {preds_memory[0][i]:<10.4f} | {diff:<10.4f}"
            )

        # ==========================================
        # 4. Save the Adapted Model
        # ==========================================
        print(f"\n--- 4. Saving Checkpoint to {save_path} ---")
        vp.save_checkpoint(save_path)

        # ==========================================
        # 5. Reload & Re-Predict
        # ==========================================
        print(f"\n--- 5. Reloading and Re-Predicting ---")

        # Initialize NEW instance from the SAVED file
        vp_reloaded = ViscosityPredictor(save_path, device="cpu")
        vp_reloaded.hydrate()

        preds_reloaded = vp_reloaded.predict(df_features)

        # Compare results
        if np.allclose(preds_memory, preds_reloaded, atol=1e-5):
            print(
                "\n[SUCCESS] Reloaded predictions match in-memory predictions exactly."
            )
            print("The model state (including adaptation) was successfully persisted.")
        else:
            print("\n[WARNING] Reloaded predictions differ from in-memory predictions.")
            print(
                "Analysis: The 'hydrate()' method in 'inference.py' loads the base model"
            )
            print(
                "but may not automatically restore the 'adapter' or the 'forward' monkey-patch"
            )
            print("unless 'load_model_checkpoint' handles it implicitly.")

            print(f"\nFirst Target Comparison:")
            print(f"In-Memory: {preds_memory[0][0]:.4f}")
            print(f"Reloaded:  {preds_reloaded[0][0]:.4f}")

    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_adaptive_cycle()
