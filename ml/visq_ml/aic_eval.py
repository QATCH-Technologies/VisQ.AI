import copy
import io
import sys

import numpy as np
import pandas as pd
import torch

# Handle imports consistent with your project structure
try:
    from inference import ViscosityPredictor
except ImportError:
    try:
        from visq_ml.inference import ViscosityPredictor
    except ImportError:
        print("Error: Could not import ViscosityPredictor.")
        sys.exit(1)

# --- 1. Configuration ---
CHECKPOINT_PATH = "models/experiments/20260128_095004/model_4.pt"
NOISE_LEVELS = [0.01, 0.05]  # 1% and 5% perturbation
NUM_SAMPLES_AIC = 23  # Total training samples (used for AIC formula)

# --- 2. Data Loading (Reusing Vuda Data for Consistency) ---
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
TARGET_COLS = [c for c in df_test.columns if "Viscosity_" in c]

# --- 3. Metric Calculations ---


def calculate_aic(model, y_true, y_pred, n_samples, scope_module=None):
    """
    Computes Akaike Information Criterion (AIC) focused on the active components.

    Args:
        model: The full model (for prediction context).
        y_true, y_pred: Arrays of ground truth and predictions.
        n_samples: Number of observations.
        scope_module: (Optional) The specific submodule to count params for (e.g., model.adapter).
                      If None, defaults to counting ALL params with requires_grad=True.
    """
    # 1. Calculate MSE (Fit)
    mse = np.mean((y_true - y_pred) ** 2)

    # 2. Count Trainable Parameters (k)
    # STRATEGY: If a scope is provided (e.g., the adapter), count ONLY that.
    # Otherwise, fall back to the standard "requires_grad" check.
    if scope_module:
        k = sum(p.numel() for p in scope_module.parameters())
        print(
            f"[AIC Debug] Scoped to '{type(scope_module).__name__}': counting {k} params."
        )
    else:
        # Fallback: Count everything that isn't frozen
        k = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Calculate AIC
    if mse <= 1e-9:  # Prevent log(0)
        aic = -np.inf
    else:
        aic = n_samples * np.log(mse) + 2 * k

    return aic, mse, k


def run_stability_test(vp, df, noise_pct=0.01):
    """
    Perturbs numerical inputs by noise_pct and measures output deviation.
    """
    df_perturbed = df.copy()

    # Identify numerical columns to perturb (exclude IDs and categorical)
    numerical_cols = [
        "Protein_conc",
        "Temperature",
        "Buffer_pH",
        "Buffer_conc",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
        "Excipient_conc",
    ]

    # 1. Inject Noise
    for col in numerical_cols:
        if col in df_perturbed.columns:
            # Add Gaussian noise: New = Old + (Old * Noise * RandomNormal)
            sigma = df_perturbed[col] * noise_pct
            noise = np.random.normal(0, 1, size=len(df_perturbed)) * sigma
            df_perturbed[col] = df_perturbed[col] + noise

    # 2. Predict on Clean vs Perturbed
    preds_clean = vp.predict(df)
    preds_noisy = vp.predict(df_perturbed)

    # 3. Calculate Deviation (Mean Absolute Percentage Difference)
    # Avoid div by zero by adding epsilon
    diff = np.abs(preds_clean - preds_noisy)
    mean_preds = np.abs(preds_clean) + 1e-6
    mapd = np.mean(diff / mean_preds) * 100

    return mapd


# --- 4. Main Execution ---


def run_robustness_eval():
    print(f"--- ROBUSTNESS EVALUATION: Stability & AIC ---")

    # 1. Load Predictor
    print(f"[Init] Loading checkpoint: {CHECKPOINT_PATH}")
    try:
        vp = ViscosityPredictor(CHECKPOINT_PATH)
        vp.hydrate()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Adapt the model first (using logic from your diagnostic script)
    # We want to test the ADAPTED model's stability
    y_test = df_test[TARGET_COLS].values
    ref_df = pd.read_csv("data/raw/formulation_data_01092026.csv")

    print(f"[Setup] Adapting model to Vuda (to simulate production state)...")
    vp.learn(df_new=df_test, y_new=y_test, epochs=100, lr=0.02, reference_df=ref_df)

    # --- TEST 1: STABILITY ANALYSIS ---
    print("\n" + "=" * 60)
    print("1. STABILITY ANALYSIS (Perturbation Testing)")
    print("   Goal: Ensure adaptation didn't break physical consistency.")
    print("=" * 60)

    print(f"{'Input Noise':<15} | {'Output Deviation (MAPD)':<25} | {'Status':<10}")
    print("-" * 55)

    for noise in NOISE_LEVELS:
        mapd = run_stability_test(vp, df_test, noise_pct=noise)

        # Heuristic: Output change should be roughly proportional to input noise
        # If 1% noise causes >5% output change, it's unstable
        ratio = mapd / (noise * 100)
        if ratio < 1.5:
            status = "ROBUST"
        elif ratio < 3.0:
            status = "ACCEPTABLE"
        else:
            status = "UNSTABLE"

        print(f"{noise*100:>5.1f}%          | {mapd:>22.4f}%           | {status}")

    # --- TEST 2: AIC EVALUATION ---
    print("\n" + "=" * 60)
    print("2. INFORMATION CRITERION (AIC)")
    print("   Goal: Balance accuracy vs. memorization risk.")
    print("=" * 60)

    y_pred = vp.predict(df_test)
    aic, mse, k = calculate_aic(
        vp.model,
        y_test,
        y_pred,
        n_samples=NUM_SAMPLES_AIC,
        scope_module=vp.adapter,
    )

    print(f"   MSE (Fit):         {mse:.6f}")
    print(f"   Params (k):        {k}")
    print(f"   AIC Score:         {aic:.2f}")
    print("-" * 60)
    print("   INTERPRETATION:")
    print("   - Use this AIC score as a baseline.")
    print("   - If you modify the adapter architecture and AIC drops,")
    print("     the new architecture is better despite parameter changes.")
    print("   - If AIC rises, the accuracy gain wasn't worth the extra complexity.")


if __name__ == "__main__":
    run_robustness_eval()
