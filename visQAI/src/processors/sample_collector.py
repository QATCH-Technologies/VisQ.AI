import pandas as pd


class SampleCollector:

    def __init__(self, known_df: pd.DataFrame = None):
        self.id_field = 'ID'
        self.categorical_features = [
            'Protein_type', 'Buffer_type',
            'Sugar_type', 'Surfactant_type'
        ]
        self.numeric_features = [
            'MW', 'PI_mean', 'PI_range', 'Protein_concentration',
            'Temperature', 'Buffer_pH', 'Buffer_conc', 'NaCl',
            'Sugar_concentration', 'Surfactant_concentration',
            'Viscosity_100', 'Viscosity_1000', 'Viscosity_10000',
            'Viscosity_100000', 'Viscosity_15000000'
        ]
        self._cat_maps = {col: {} for col in self.categorical_features}
        self._next_codes = {col: 0 for col in self.categorical_features}
        if known_df is not None:
            for col in self.categorical_features:
                for val in known_df[col].dropna().unique():
                    if val not in self._cat_maps[col]:
                        self._cat_maps[col][val] = self._next_codes[col]
                        self._next_codes[col] += 1

    def query_user(self) -> dict:
        sample = {}
        sample[self.id_field] = input(f"Enter {self.id_field}: ").strip()
        for col in self.categorical_features:
            sample[col] = input(f"Enter {col} (categorical): ").strip()
        for col in self.numeric_features:
            val = input(f"Enter {col} (numeric): ").strip()
            try:
                sample[col] = float(val)
            except ValueError:
                raise ValueError(f"Field {col} must be numeric, got '{val}'")
        return sample

    def _encode_cat(self, col: str, val: str) -> int:
        cmap = self._cat_maps[col]
        if val not in cmap:
            cmap[val] = self._next_codes[col]
            self._next_codes[col] += 1
        return cmap[val]

    def create_dataframe(self, sample: dict) -> pd.DataFrame:
        row = {self.id_field: sample[self.id_field]}
        for col in self.categorical_features:
            row[col] = self._encode_cat(col, sample[col])
        for col in self.numeric_features:
            row[col] = sample[col]
        return pd.DataFrame([row])

    def collect(self, sample: dict = None) -> pd.DataFrame:
        if sample is None:
            sample = self.query_user()

        expected = ({self.id_field}
                    | set(self.categorical_features)
                    | set(self.numeric_features))
        missing = expected - sample.keys()
        extra = sample.keys() - expected
        if missing:
            raise KeyError(f"Missing fields: {missing}")
        if extra:
            raise KeyError(f"Unexpected fields: {extra}")

        return self.create_dataframe(sample)

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode a whole DataFrame of raw inputs:
          - map each categorical col → its integer code
          - coerce numeric cols to float, filling errors/back-conversions with 0.0
        """
        df_enc = pd.DataFrame()

        # Categoricals → integer codes
        for col in self.categorical_features:
            # assume df[col] exists
            df_enc[col] = df[col].map(
                lambda v: self._encode_cat(col, v)).astype(int)

        # Numerics → to_numeric + fillna(0)
        for col in self.numeric_features:
            # coerce errors (like 'none', 'NaN') to NaN, then fill with 0.0
            df_enc[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .fillna(0.0)
                .astype(float)
            )

        return df_enc


if __name__ == "__main__":
    known = pd.read_csv('content/formulation_data_05272025.csv')
    collector = SampleCollector(known)
    raw = {
        "ID": "exp_1",
        "Protein_type": "EXP",
        "Buffer_type": "EXP",
        "Sugar_type": "EXP",
        "Surfactant_type": "EXP",
        "MW": 100,
        'PI_mean': 10,
        'PI_range': 1,
        'Protein_concentration': 100,
        'Temperature': 25,
        'Buffer_pH': 5,
        'Buffer_conc': 10,
        'NaCl': 10,
        'Sugar_concentration': 0.1,
        'Surfactant_concentration': 0.01,
        'Viscosity_100': 10,
        'Viscosity_1000': 9,
        'Viscosity_10000': 9,
        'Viscosity_100000': 9,
        'Viscosity_15000000': 8
    }
    df_new = collector.collect(raw)
    print(df_new)
