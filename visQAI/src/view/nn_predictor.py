import os
import pandas as pd
import joblib


class NNPredictor:

    def __init__(self, model_dir,
                 target_columns=None):
        self.pipeline = joblib.load(os.path.join(model_dir, "pipeline_nn.pkl"))
        self.target_columns = target_columns or [
            "Viscosity100", "Viscosity1000", "Viscosity10000",
            "Viscosity100000", "Viscosity15000000"
        ]

    def predict(self, df_new):
        preds = self.pipeline.predict(df_new)
        return pd.DataFrame(preds, columns=self.target_columns, index=df_new.index)
