import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle


class TransformerPipeline:
    CATEGORICAL_FEATURES = [
        'Protein_type', 'Buffer_type', 'Salt_type',
        'Stabilizer_type', 'Surfactant_type',
    ]
    NUMERIC_FEATURES = [
        'MW', 'PI_mean', 'PI_range', 'Protein_conc',
        'Temperature', 'Buffer_pH', 'Buffer_conc',
        'Salt_conc', 'Stabilizer_conc', 'Surfactant_conc',
    ]

    def __init__(self, scaler_path: str = None, encoder_path: str = None):
        # Load or initialize scaler
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = MinMaxScaler()

        # Load or initialize one-hot encoder
        if encoder_path:
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
        else:
            self.encoder = OneHotEncoder(
                handle_unknown='ignore',
            )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Replace missing values in all features with 0
        for col in self.NUMERIC_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

        for col in self.CATEGORICAL_FEATURES:
            out[col] = out[col].fillna("Unknown").astype(str)

        return out

    def fit(self, X: pd.DataFrame, y=None):
        X_clean = self._clean(X)

        self.encoder.fit(X_clean[self.CATEGORICAL_FEATURES])
        self.scaler.fit(X_clean[self.NUMERIC_FEATURES])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = self._clean(X)

        # One-hot encode
        cat_encoded = self.encoder.transform(
            X_clean[self.CATEGORICAL_FEATURES])
        cat_columns = self.encoder.get_feature_names_out(
            self.CATEGORICAL_FEATURES)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_columns, index=X.index)

        # Scale numeric
        num_scaled = self.scaler.transform(X_clean[self.NUMERIC_FEATURES])
        num_df = pd.DataFrame(
            num_scaled, columns=self.NUMERIC_FEATURES, index=X.index)

        # Concatenate all together
        return pd.concat([cat_df, num_df], axis=1)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, scaler_path: str, encoder_path: str):
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.encoder, f)
