import io
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score  # Added for evaluation

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

# --- 1. Load the Vuda Data ---
csv_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F466,Vuda,bispecific,5,152,8.1,0.3,145,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,3.6,3.4,3.4,3.4,3.4
F467,Vuda,bispecific,5,152,8.1,0.3,80,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,3.3,2.3,2,1.8,1.3
F468,Vuda,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,2.7,2.6,2.5,2.4,2.3
F469,Vuda,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,NaCl,140,none,0,none,0,none,0,1.5,1.3,1.2,1.2,1.2,1.2,1.2
F470,Vuda,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1.5,1.3,2.3,1.9,1.5,1.2,1
F471,Vuda,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,none,0,tween-80,0.05,none,0,1.5,1.3,2.1,2.1,2,1.9,1.9
F472,Vuda,bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,NaCl,140,Sucrose,0.4,none,0,none,0,1.5,1.3,2.5,1.9,1.7,1.6,1.3
F473,Vuda,bispecific,5,152,8.1,0.3,145,25,Histidine,6,15,none,0,none,0,none,0,Arginine,100,1.5,1.3,6.4,4.3,3.9,3.5,2.7
F474,Vuda,bispecific,5,152,8.1,0.3,145,25,Histidine,6,15,none,0,none,0,none,0,Arginine,200,1.5,1.3,5,4.2,4,3.9,3.7
F475,Vuda,bispecific,5,152,8.1,0.3,276,25,Acetate,5,20,none,0,none,0,none,0,none,0,1.5,1.3,10.3,8.5,7.9,7.3,6.1
F476,Vuda,bispecific,5,152,8.1,0.3,138,25,Acetate,5,20,none,0,none,0,none,0,none,0,1.5,1.3,2.5,2,2,2,2
F477,Vuda,bispecific,5,152,8.1,0.3,69,25,Acetate,5,20,none,0,none,0,none,0,none,0,1.5,1.3,1.1,1,1,0.9,0.9
F478,Vuda,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,NaCl,175,none,0,none,0,none,0,1.5,1.3,2.7,2.5,2.5,2.5,2.5
F479,Vuda,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,none,0,Sucrose,0.5,none,0,none,0,1.5,1.3,7.6,6.1,5.6,5.2,4.3
F480,Vuda,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,none,0,none,0,tween-80,0.0625,none,0,1.5,1.3,3.4,3.1,3,2.9,2.8
F481,Vuda,bispecific,5,152,8.1,0.3,150,25,Acetate,5,20,NaCl,175,Sucrose,0.5,none,0,none,0,1.5,1.3,10.5,7.4,6.6,5.8,4
F482,Vuda,bispecific,5,152,8.1,0.3,150,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,4.2,3.8,3.7,3.7,3.7
F483,Vuda,bispecific,5,152,8.1,0.3,50,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.5,1.3,1.6,1.6,1.3,1.3,1.4
F484,Vuda,bispecific,5,152,8.1,0.3,50,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1.5,1.3,1.9,1.5,1.4,1.3,1.3
F485,Vuda,bispecific,5,152,8.1,0.3,50,25,Histidine,6,15,none,0,Sucrose,0.8,none,0,none,0,1.5,1.3,1.2,1.2,1.1,1.1,1.2
F486,Vuda,bispecific,5,152,8.1,0.3,150,25,Histidine,6,15,NaCl,140,Sucrose,0.8,none,0,none,0,1.5,1.3,3.4,3,2.9,2.8,2.7
F487,Vuda,bispecific,5,152,8.1,0.3,150,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1.5,1.3,2.9,2.9,2.9,2.9,2.5"""

df_test = pd.read_csv(io.StringIO(csv_data))

# --- 2. Configuration ---
# Update this path to your actual checkpoint
CHECKPOINT_PATH = "models/experiments/20260128_095004/model_4.pt"


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
    if "Vuda" in processor.cat_maps["Protein_type"]:
        print(
            "WARNING: 'Vuda' is already in the vocabulary. Please use a fresh checkpoint for this test."
        )
        return

    # 4. Prepare Data for `learn`
    TARGET_COLS = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]
    y_test = df_test[TARGET_COLS].values

    # Load Legacy Data
    print(f"[Init] Loading legacy data...")
    ref_df = pd.read_csv("data/raw/formulation_data_01092026.csv")

    # 5. Run Adaptation
    print(f"\n[Action] Running vp.learn() for Vuda...")
    try:
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

    # 6. Inspect Internal State (Physics Check)
    print("\n[Analysis] Inspecting Physics Layer Internal State")
    target_class_name = "bispecific"

    if target_class_name in processor.cat_maps["Protein_class_type"]:
        new_class_idx = processor.cat_maps["Protein_class_type"].index(
            target_class_name
        )
        print(f"   Target Class: '{target_class_name}' (Index {new_class_idx})")

        for i, name in enumerate(model.cat_feature_names):
            if "surfactant" in name.lower() or "excipient" in name.lower():
                phys_layer = model.physics_layers[i]
                if hasattr(phys_layer, "static_scores"):
                    static_val = (
                        phys_layer.static_scores[new_class_idx, :, :]
                        .abs()
                        .mean()
                        .item()
                    )
                    delta_val = (
                        phys_layer.delta[new_class_idx, :, :].abs().mean().item()
                    )
                    print(f"   Layer: {name}")
                    print(f"   - Static Base:   {static_val:.6f}")
                    print(f"   - Learned Delta: {delta_val:.6f}")

    # 7. Comprehensive R2 Evaluation
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE EVALUATION")
    print("=" * 60)

    # A. Vuda (Adaptation Performance)
    print(f"\n[A] Vuda Performance (Adaptation Target, N={len(df_test)}):")
    y_pred_vuda = vp.predict(df_test)
    y_true_vuda = df_test[TARGET_COLS].values

    # Header
    print(f"{'Shear Rate':<20} | {'R2 Score':<10} | {'RMSE':<10}")
    print("-" * 46)

    for i, col in enumerate(TARGET_COLS):
        r2 = r2_score(y_true_vuda[:, i], y_pred_vuda[:, i])
        rmse = np.sqrt(np.mean((y_true_vuda[:, i] - y_pred_vuda[:, i]) ** 2))
        shear_label = col.replace("Viscosity_", "") + " s^-1"
        print(f"{shear_label:<20} | {r2:>8.4f}   | {rmse:>8.4f}")

    # B. Legacy (Stability Check)
    # Ensure we only test on rows with valid targets
    ref_df_clean = ref_df.dropna(subset=TARGET_COLS)
    print(
        f"\n[B] Legacy Data Performance (Catastrophic Forgetting Check, N={len(ref_df_clean)}):"
    )

    y_pred_legacy = vp.predict(ref_df_clean)
    y_true_legacy = ref_df_clean[TARGET_COLS].values

    # Header
    print(f"{'Shear Rate':<20} | {'R2 Score':<10} | {'RMSE':<10}")
    print("-" * 46)

    for i, col in enumerate(TARGET_COLS):
        r2 = r2_score(y_true_legacy[:, i], y_pred_legacy[:, i])
        rmse = np.sqrt(np.mean((y_true_legacy[:, i] - y_pred_legacy[:, i]) ** 2))
        shear_label = col.replace("Viscosity_", "") + " s^-1"
        print(f"{shear_label:<20} | {r2:>8.4f}   | {rmse:>8.4f}")

    print("=" * 60)


if __name__ == "__main__":
    run_diagnostic()
