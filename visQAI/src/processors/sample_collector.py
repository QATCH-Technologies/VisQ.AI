import pandas as pd


class SampleCollector:
    def __init__(self,
                 df_known: pd.DataFrame,
                 categorical_cols: list[str]):
        """
        df_known: existing DataFrame of raw samples (with 'ID' column)
        categorical_cols: list of columns to treat as categorical
        """
        self.df_known = df_known.copy()
        self.cat_cols = categorical_cols

        # initialize mapping for each categorical column
        self.cat_maps: dict[str, dict[str, int]] = {}
        for c in self.cat_cols:
            uniques = list(self.df_known[c].dropna().unique())
            self.cat_maps[c] = {cat: idx for idx, cat in enumerate(uniques)}

    def _encode_value(self, col: str, val: str) -> int:
        """
        Map a category to its integer code, adding new categories as needed.
        """
        cmap = self.cat_maps[col]
        if val not in cmap:
            cmap[val] = len(cmap)
        return cmap[val]

    def query_user_sample(self) -> pd.DataFrame:
        """
        Prompt the user for each non-ID field, auto-assign an ID,
        encode categorical values, and return a one-row DataFrame
        with a fixed set of columns (including 'ID').
        """
        # fields to prompt (excluding 'ID')
        fields = [
            "Protein_type", "MW", "PI_mean", "PI_range",
            "Protein_concentration", "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "NaCl", "Sugar_type", "Sugar_concentration",
            "Surfactant_type", "Surfactant_concentration",
            "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
            "Viscosity_100000", "Viscosity_15000000"
        ]

        raw: dict[str, str] = {}
        # auto-generate a new numeric ID
        raw['ID'] = len(self.df_known) + 1

        for f in fields:
            v = input(f"{f}: ").strip()
            if f not in self.cat_cols:
                # numeric field
                try:
                    float(v)
                except ValueError:
                    raise ValueError(f"Expected numeric for '{f}', got '{v}'")
            raw[f] = v

        # encode raw to numeric values
        enc: dict[str, float | int] = {}
        for col, val in raw.items():
            if col in self.cat_cols:
                enc[col] = self._encode_value(col, val)
            elif col == 'ID':  # leave ID as integer
                enc[col] = val  # already int
            else:
                enc[col] = float(val)

        # build DataFrame ensuring fixed column order
        cols_full = ['ID'] + fields
        row_df = pd.DataFrame([enc], columns=cols_full)

        # append the raw sample for future mapping
        self.df_known = pd.concat(
            [self.df_known, pd.DataFrame([raw])],
            ignore_index=True
        )

        return row_df


# ─── Usage ───
if __name__ == "__main__":
    # load existing raw samples (must include 'ID')
    df_known = pd.read_csv("content/formulation_data_05272025.csv")

    collector = SampleCollector(
        df_known=df_known,
        categorical_cols=[
            "Protein_type", "Buffer_type",
            "Sugar_type", "Surfactant_type"
        ]
    )

    print("Enter your new sample (ID will be assigned automatically):")
    sample = collector.query_user_sample()
    print("\nEncoded sample (fixed columns including ID):")
    print(sample)
