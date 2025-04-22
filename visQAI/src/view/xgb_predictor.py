import os
import pandas as pd
import joblib
import xgboost as xgb


class XGBPredictor:

    def __init__(self, model_dir,
                 feature_columns=None,
                 target_columns=None):
        self.model_dir = model_dir
        self.preprocessor = joblib.load(
            os.path.join(model_dir, "preprocessor.pkl"))

        self.boosters = joblib.load(os.path.join(model_dir, "boosters.pkl"))
        self.feature_columns = feature_columns or ["Protein type", "Protein", "Temperature", "Buffer",
                                                   "Sugar", "Sugar (M)", "Surfactant", "TWEEN"]
        self.target_columns = target_columns or ["Viscosity100", "Viscosity1000",
                                                 "Viscosity10000", "Viscosity100000",
                                                 "Viscosity15000000"]

    def predict(self, df_new):
        X = df_new[self.feature_columns]
        X_mat = self.preprocessor.transform(X)
        dmat = xgb.DMatrix(X_mat)
        preds = {target: booster.predict(dmat)
                 for target, booster in self.boosters.items()}
        return pd.DataFrame(preds, index=df_new.index)
