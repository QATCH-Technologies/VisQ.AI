import io
import sys

import numpy as np
import pandas as pd
import torch

# Handle imports for different directory structures
try:
    from inference import ViscosityPredictor
except ImportError:
    try:
        from visq_ml.inference import ViscosityPredictor
    except ImportError:
        print(
            "Error: Could not import ViscosityPredictor. Ensure you are in the project root."
        )
        sys.exit(1)

# --- 1. Load the Vudalimab Data ---
# Note: Using 'Viscosity_100' as the target column for the adaptation test
csv_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F466,Vudalimab,bispecific,5,152,8.1,0.3,145,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,3.6,3.4,3.4,3.4,3.4
F467,Vudalimab,bispecific,5,152,8.1,0.3,80,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,3.3,2.3,2,1.8,1.3
F468,Vudalimab,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,2.7,2.6,2.5,2.4,2.3
F469,Vudalimab,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,NaCl,140,none,0,none,0,none,0,1.5,1.3,1.2,1.2,1.2,1.2,1.2
F470,Vudalimab,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1.5,1.3,2.3,1.9,1.5,1.2,1
F471,Vudalimab,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,none,0,tween-80,0.05,none,0,1.5,1.3,2.1,2.1,2,1.9,1.9
F472,Vudalimab,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,NaCl,140,Sucrose,0.4,none,0,none,0,1.5,1.3,2.5,1.9,1.7,1.6,1.3
F473,Vudalimab,bispecific,5,152,8.1,0.3,145,25,Histidine,6,15,none,0,none,0,none,0,Arginine,100,1.5,1.3,6.4,4.3,3.9,3.5,2.7
F474,Vudalimab,bispecific,5,152,8.1,0.3,145,25,Histidine,6,15,none,0,none,0,none,0,Arginine,200,1.5,1.3,5,4.2,4,3.9,3.7
F475,Vudalimab,bispecific,5,152,8.1,0.3,276,25,Acetate,5,20,none,0,none,0,none,0,none,0,1.5,1.3,10.3,8.5,7.9,7.3,6.1
F476,Vudalimab,bispecific,5,152,8.1,0.3,138,25,Acetate,5,20,none,0,none,0,none,0,none,0,1.5,1.3,2.5,2,2,2,2
F477,Vudalimab,bispecific,5,152,8.1,0.3,69,25,Acetate,5,20,none,0,none,0,none,0,none,0,1.5,1.3,1.1,1,1,0.9,0.9
F478,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,NaCl,175,none,0,none,0,none,0,1.5,1.3,2.7,2.5,2.5,2.5,2.5
F479,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,none,0,Sucrose,0.5,none,0,none,0,1.5,1.3,7.6,6.1,5.6,5.2,4.3
F480,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,none,0,none,0,tween-80,0.0625,none,0,1.5,1.3,3.4,3.1,3,2.9,2.8
F481,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,NaCl,175,Sucrose,0.5,none,0,none,0,1.5,1.3,10.5,7.4,6.6,5.8,4
F482,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,4.2,3.8,3.7,3.7,3.7
F483,Vudalimab,bispecific,5,152,8.1,0.3,50,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,1.6,1.6,1.3,1.3,1.4
F484,Vudalimab,bispecific,5,152,8.1,0.3,50,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1.5,1.3,1.9,1.5,1.4,1.3,1.3
F485,Vudalimab,bispecific,5,152,8.1,0.3,50,25,Histidine,6,15,none,0,Sucrose,0.8,none,0,none,0,1.5,1.3,1.2,1.2,1.1,1.1,1.2
F486,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Histidine,6,15,NaCl,140,Sucrose,0.8,none,0,none,0,1.5,1.3,3.4,3,2.9,2.8,2.7
F487,Vudalimab,bispecific,5,152,8.1,0.3,150,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1.5,1.3,2.9,2.9,2.9,2.9,2.5"""

df_test = pd.read_csv(io.StringIO(csv_data))

# --- 2. Configuration ---
# Update this path to your actual checkpoint
CHECKPOINT_PATH = "models/experiments/20260127_140127/model_3.pt"


def run_diagnostic():
    print(f"--- DIAGNOSTIC: Testing Adaptive Learning & Physics Inheritance ---")

    # 1. Load Predictor
    print(f"[Init] Loading checkpoint: {CHECKPOINT_PATH}")
    try:
        vp = ViscosityPredictor(CHECKPOINT_PATH)
        vp.hydrate()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model = vp.model
    processor = vp.processor

    # 2. Sanity Check
    if "Vudalimab" in processor.cat_maps["Protein_type"]:
        print(
            "WARNING: 'Vudalimab' is already in the vocabulary. Please use a fresh checkpoint for this test."
        )
        return

    # 4. Prepare Data for `learn`
    # We must provide valid Y targets so the adapter training doesn't crash
    TARGET_COLS = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]
    y_test = df_test[TARGET_COLS].values
    ref_df = pd.read_csv("data/raw/formulation_data_12292025.csv")
    # 5. Run Adaptation (The Function Under Test)
    print(f"\n[Action] Running vp.learn() for Vudalimab...")
    try:
        # We run 0 epochs just to trigger expansion, or 1 to test full flow
        vp.learn(
            df_new=df_test,
            y_new=y_test,
            epochs=100,
            lr=0.02,
            reference_df=ref_df,
        )
    except Exception as e:
        print(f"CRASH during vp.learn(): {e}")
        import traceback

        traceback.print_exc()
        return

    # 6. Inspect Internal State
    print("\n[Analysis] Inspecting Physics Layer Internal State")

    # --- FIX: Look up the CLASS index, not the Protein index ---
    # The Physics layer dim 0 corresponds to 'Protein_class_type'
    target_class_name = "bispecific"  # From your CSV data
    if target_class_name not in processor.cat_maps["Protein_class_type"]:
        print(f"Error: Target class '{target_class_name}' not found in maps.")
        return

    new_class_idx = processor.cat_maps["Protein_class_type"].index(target_class_name)

    # We also need the class of the analog to compare apples to apples
    # Assuming the analog provided (e.g., 'bgg') maps to a class that exists
    # If we don't know the analog's class easily, we can skip the direct comparison
    # or infer it if we had the training data. For now, we'll just check if the NEW class is zero.

    print(f"   Target Class: '{target_class_name}' (Index {new_class_idx})")

    # Locate the Surfactant Physics Layer
    layer_found = False
    for i, name in enumerate(model.cat_feature_names):
        # Look for surfactant/excipient layers
        if "surfactant" in name.lower() or "excipient" in name.lower():
            phys_layer = model.physics_layers[i]
            if hasattr(phys_layer, "static_scores"):
                # 1. Static Prior (Fixed)
                static_val = (
                    phys_layer.static_scores[new_class_idx, :, :].abs().mean().item()
                )

                # 2. Learnable Delta (Active) - THIS is what changes!
                delta_val = phys_layer.delta[new_class_idx, :, :].abs().mean().item()

                # 3. Total Effective Prior
                total_mag = (
                    (
                        phys_layer.static_scores[new_class_idx]
                        + phys_layer.delta[new_class_idx]
                    )
                    .abs()
                    .mean()
                    .item()
                )

                print(f"   Layer: {name}")
                print(f"   - Static Base:   {static_val:.6f} (Fixed)")
                print(f"   - Learned Delta: {delta_val:.6f} <--- Check this!")
                print(f"   - Total Magnitude: {total_mag:.6f}")

    if not layer_found:
        print("WARNING: No active physics layers found to test.")
    eval_csv_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F465,Vudalimab,bispecific,5,152,8.1,0.3,320,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,13,8.8,7.7,6.7,4.6"""

    df_eval = pd.read_csv(io.StringIO(eval_csv_data))
    # 7. Prediction Check on Surfactant Data
    print("\n[Verification] Prediction Check on Row F465 (Contains Tween-80)")
    eval_row = df_eval[df_eval["ID"] == "F465"]

    # We expect reasonable predictions if physics are present.
    # If physics are missing, the Adapter might have over-corrected during the 1 epoch of training.
    res = vp.predict(eval_row)
    print(f"   Target Viscosity: {eval_row['Viscosity_100'].values[0]}")
    print(f"   Model Prediction: {res[0][0]:.4f}")

    eval_csv_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F165,Adalimumab,mAb_IgG1,3.0,148.0,8.7,0.3,100.0,27.5,Acetate,5.0,20,none,0,none,0.0,none,0.0,none,0,1.0,1.0,3.19,2.72,2.48,2.32,1.45"""

    df_eval = pd.read_csv(io.StringIO(eval_csv_data))
    # 7. Prediction Check on Surfactant Data
    print("\n[Verification] Legacy Prediction")
    eval_row = df_eval[df_eval["ID"] == "F165"]

    # We expect reasonable predictions if physics are present.
    # If physics are missing, the Adapter might have over-corrected during the 1 epoch of training.
    res = vp.predict(eval_row)
    print(f"   Target Viscosity: {eval_row['Viscosity_100'].values[0]}")
    print(f"   Model Prediction: {res[0][0]:.4f}")


if __name__ == "__main__":
    run_diagnostic()
