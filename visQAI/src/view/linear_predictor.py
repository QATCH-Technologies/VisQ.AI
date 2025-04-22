import os
import pandas as pd
import joblib


class LinearPredictor:
    def __init__(self, model_dir,
                 feature_columns=None,
                 target_columns=None):
        self.pipeline = joblib.load(os.path.join(model_dir, "pipeline_lm.pkl"))
        self.feature_columns = feature_columns or [
            "Protein type", "Protein", "Temperature", "Buffer",
            "Sugar", "Sugar (M)", "Surfactant", "TWEEN"
        ]
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000", "Viscosity10000",
            "Viscosity100000", "Viscosity15000000"
        ]

    def predict(self, df_new):
        X = df_new[self.feature_columns]
        preds = self.pipeline.predict(X)
        return pd.DataFrame(preds, columns=self.target_columns, index=df_new.index)
