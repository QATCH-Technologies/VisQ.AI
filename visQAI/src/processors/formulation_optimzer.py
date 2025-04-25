# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize
# from sklearn.preprocessing import OneHotEncoder
# import os
# from cnn_predictor import CNNPredictor


class FormulationOptimizer:
    pass
    #     def __init__(self,
    #                  predictor,                    # instance of CNNPredictor
    #                  # e.g. {"Protein type":"poly-hIgG", "Buffer":"PBS", …}
    #                  categorical_features: dict,
    #                  # e.g. {"Protein":[0,200], "Sugar (M)":[0,1], …}
    #                  continuous_bounds: dict,
    #                  tolerance: float = 1e-3,
    #                  max_iter: int = 100):
    #         self.predictor = predictor
    #         self.tol = tolerance
    #         self.max_iter = max_iter

    #         # --- Prepare categorical encoder once ---
    #         self.cat_names = list(categorical_features.keys())
    #         self.cat_encoder = OneHotEncoder(handle_unknown="ignore")
    #         # fit on this single sample just to capture the column names
    #         cat_vals = [list(categorical_features.values())]
    #         self.cat_encoder.fit([list(categorical_features.values())])
    #         # store fixed one-hot vector
    #         self.fixed_cat_vec = self.cat_encoder.transform(cat_vals)[0]

    #         # continuous features
    #         self.cont_names = list(continuous_bounds.keys())
    #         self.bounds = [continuous_bounds[name] for name in self.cont_names]

    #     def _make_input_df(self, conc_vector: np.ndarray) -> pd.DataFrame:
    #         """Merge fixed one-hot and current concentrations into a DF ready for predictor."""
    #         # build feature dict
    #         feat = {}
    #         # continuous
    #         for name, val in zip(self.cont_names, conc_vector):
    #             feat[name] = val
    #         # categorical: expand one-hot back to nominal columns
    #         # Note: if your CNNPredictor expects the one-hot columns directly,
    #         # you can skip this and pass the concatenated array instead.
    #         cat_columns = self.cat_encoder.get_feature_names_out(self.cat_names)
    #         for col_name, val in zip(cat_columns, self.fixed_cat_vec):
    #             feat[col_name] = val

    #         return pd.DataFrame([feat])

    #     def _loss(self, conc_vector: np.ndarray, target: np.ndarray) -> float:
    #         """Objective: MSE between predicted and target viscosity profiles."""
    #         df_in = self._make_input_df(conc_vector)
    #         pred_profile = self.predictor.predict(df_in).ravel()
    #         return float(np.mean((pred_profile - target)**2))

    #     def optimize(self,
    #                  target_profile: np.ndarray,
    #                  initial_guess: dict) -> dict:
    #         """
    #         Run Nelder–Mead starting from initial_guess until MSE ≤ tol.
    #         Returns a dict of optimized continuous concentrations.
    #         """
    #         x0 = np.array([initial_guess[name] for name in self.cont_names])

    #         res = minimize(
    #             fun=lambda x: self._loss(x, target_profile),
    #             x0=x0,
    #             method="Nelder-Mead",
    #             options={"xatol": self.tol, "maxiter": self.max_iter}
    #         )

    #         optimized = {name: val for name, val in zip(self.cont_names, res.x)}
    #         return optimized

    # def prompt_target_profile() -> np.ndarray:
    #     """
    #     Interactively prompt for the target viscosity profile values at predefined shear rates.
    #     Returns a numpy array of viscosities in the order:
    #       Viscosity100, Viscosity1000, Viscosity10000, Viscosity100000, Viscosity15000000
    #     """
    #     print("Enter target viscosity profile:")
    #     shear_rates = [100, 1000, 10000, 100000, 15000000]
    #     values = []
    #     for rate in shear_rates:
    #         while True:
    #             val = input(f"  Viscosity at shear rate {rate}: ").strip()
    #             try:
    #                 values.append(float(val))
    #                 break
    #             except ValueError:
    #                 print("    Invalid number, please enter a numeric value.")
    #     return np.array(values)

    # def prompt_initial_formulation() -> (dict, dict):
    #     """
    #     Interactively prompt for initial formulation parameters.

    #     Returns:
    #       categorical_features: dict of categorical feature values
    #       initial_guess: dict of continuous feature initial values
    #     """
    #     print("Enter initial formulation parameters:")
    #     # Categorical features
    #     cat_keys = ['Protein type', 'Buffer', 'Sugar', 'Surfactant']
    #     categorical_features = {}
    #     for key in cat_keys:
    #         categorical_features[key] = input(f"  {key}: ").strip()

    #     # Continuous features
    #     cont_keys = ['Protein', 'Temperature', 'Sugar (M)', 'TWEEN']
    #     initial_guess = {}
    #     for key in cont_keys:
    #         while True:
    #             val = input(f"  {key} (numeric): ").strip()
    #             try:
    #                 initial_guess[key] = float(val)
    #                 break
    #             except ValueError:
    #                 print("    Invalid number, please enter a numeric value.")
    #     return categorical_features, initial_guess

    # def prompt_continuous_bounds() -> dict:
    #     """
    #     Interactively prompt for optimization bounds of continuous features.
    #     Returns a dict mapping each continuous feature to [min, max].
    #     """
    #     print("Enter continuous bounds for optimization:")
    #     cont_keys = ['Protein', 'Temperature', 'Sugar (M)', 'TWEEN']
    #     bounds = {}
    #     for key in cont_keys:
    #         while True:
    #             lb = input(f"  {key} lower bound: ").strip()
    #             ub = input(f"  {key} upper bound: ").strip()
    #             try:
    #                 lbf = float(lb)
    #                 ubf = float(ub)
    #                 if ubf < lbf:
    #                     print("    Upper bound must be >= lower bound. Try again.")
    #                     continue
    #                 bounds[key] = [lbf, ubf]
    #                 break
    #             except ValueError:
    #                 print("    Invalid number, please enter numeric values.")
    #     return bounds

    # def main():
    #     print("=== Formulation Optimizer ===")

    #     # Prompt for target profile and initial formulation
    #     target_profile = prompt_target_profile()
    #     categorical_features, initial_guess = prompt_initial_formulation()

    #     # Prompt for bounds
    #     continuous_bounds = prompt_continuous_bounds()

    #     # Prompt for model directory and output file
    #     model_dir = input("Enter CNNPredictor model directory path: ").strip()
    #     output_path = input(
    #         "Enter output JSON path [default: optimized_formulation.json]: ").strip()
    #     if not output_path:
    #         output_path = "optimized_formulation.json"

    #     # Initialize predictor and optimizer
    #     predictor = CNNPredictor(model_dir=model_dir)
    #     optimizer = FormulationOptimizer(
    #         predictor=predictor,
    #         categorical_features=categorical_features,
    #         continuous_bounds=continuous_bounds
    #     )

    #     # Run optimization
    #     print("Running optimization...")
    #     optimized = optimizer.optimize(
    #         target_profile=target_profile,
    #         initial_guess=initial_guess
    #     )

    #     # Combine categorical and optimized continuous values
    #     result = {**categorical_features, **optimized}
    #     out_dir = os.path.dirname(output_path)
    #     if out_dir and not os.path.exists(out_dir):
    #         os.makedirs(out_dir, exist_ok=True)
    #     with open(output_path, 'w') as f:
    #         json.dump(result, f, indent=4)

    #     print(f"Optimized formulation saved to: {output_path}")

    # if __name__ == '__main__':
    #     main()
