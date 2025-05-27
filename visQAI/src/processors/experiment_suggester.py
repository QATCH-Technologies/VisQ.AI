import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sample_collector import SampleCollector


def generate_local_candidates(
    sample: pd.Series,
    df_known: pd.DataFrame,
    n_candidates: int = 200,
    frac_window: float = 0.1
) -> pd.DataFrame:
    """
    Create n_candidates by perturbing numeric features of `sample`
    within ±frac_window of the global span in df_known.
    Categorical features are held fixed at their sample value.
    """
    # 1) Identify numeric vs categorical features
    num_cols = df_known.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in sample.index if c not in num_cols]

    # 2) Precompute bounds
    lows = df_known[num_cols].min()
    highs = df_known[num_cols].max()

    rows = []
    for _ in range(n_candidates):
        row = {}
        # 3) Perturb each numeric column individually
        for c in num_cols:
            span = highs[c] - lows[c]
            noise = np.random.uniform(-frac_window * span,
                                      frac_window * span)
            val = sample[c] + noise
            # 4) Clip to [low, high]
            row[c] = float(np.clip(val, lows[c], highs[c]))
        # 5) Copy over all categoricals unchanged
        for c in cat_cols:
            row[c] = sample[c]

        rows.append(row)

    # 6) Build a DataFrame — pandas will automatically align columns
    return pd.DataFrame(rows, columns=sample.index)


def suggest_informative_samples(
    sample,               # can be a DataFrame or Series
    df_known: pd.DataFrame,
    categorical_cols: list,
    target_cols: list,
    n_candidates: int = 200,
    n_suggestions: int = 5,
    frac_window: float = 0.1
) -> pd.DataFrame:
    # ——— 0) make sure `sample` is a flat Series of scalars ———
    if isinstance(sample, pd.DataFrame):
        # squeeze 1-row DataFrame into a Series
        sample = sample.squeeze()
    # now sample[c] is a float/string, not a Series

    # 1) define your feature columns in the known data
    feat_cols = df_known.columns.drop(target_cols + ["ID"])
    X_known_feats = df_known[feat_cols]
    sample_feats = sample.reindex(feat_cols)

    # 2) generate candidates
    X_cand = generate_local_candidates(
        sample=sample_feats,
        df_known=X_known_feats,
        n_candidates=n_candidates,
        frac_window=frac_window
    )

    # 5) encode both known + candidate sets
    from sample_collector import SampleCollector  # adjust your import
    collector = SampleCollector(
        df_known=df_known,
        categorical_cols=categorical_cols
    )
    X_known_enc = collector.encode_df(X_known_feats)
    X_cand_enc = collector.encode_df(X_cand)

    # 6) train on known viscosities
    y_known = df_known[target_cols]
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_known_enc, y_known)

    # 7) measure per-tree variance as uncertainty
    all_preds = np.stack([t.predict(X_cand_enc)
                         for t in rf.estimators_], axis=0)
    cand_uncert = all_preds.std(axis=0).mean(axis=1)

    # 8) pick the top N by uncertainty
    top_idx = np.argsort(-cand_uncert)[:n_suggestions]
    return X_cand.iloc[top_idx].reset_index(drop=True)


if __name__ == "__main__":
    # 1) load your data
    df_known = pd.read_csv("content/formulation_data_05272025.csv")
    categorical_cols = ["Protein_type", "Buffer_type",
                        "Sugar_type", "Surfactant_type"]
    target_cols = [
        "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
        "Viscosity_100000", "Viscosity_15000000"
    ]

    # 2) collect user sample
    collector = SampleCollector(
        df_known=df_known, categorical_cols=categorical_cols)
    print("Enter your new sample (ID will be assigned automatically):")
    sample = collector.query_user_sample()

    # 3) suggest top 5 informative experiments
    suggestions = suggest_informative_samples(
        sample=sample,
        df_known=df_known,
        categorical_cols=categorical_cols,
        target_cols=target_cols,
        n_candidates=500,
        n_suggestions=5,
        frac_window=0.1
    )

    print("\nTop 5 suggested experiments:")
    print(suggestions)
