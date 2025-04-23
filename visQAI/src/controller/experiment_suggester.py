from collections import Counter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Any

Sample = Dict[str, Any]


class ExperimentSuggester:
    """
    Encapsulates the sequential suggestion workflow:
    1) Add measured samples via add_sample()
    2) Retrieve next experiment suggestion via next_suggestion()
    3) Optionally get a summary of all measurements
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        viscosity_cols: List[str],
        k: int = 50,
        random_state: int = 0
    ):
        self.train_df = train_df.copy()
        self.viscosity_cols = viscosity_cols
        self.k = k
        self.random_state = random_state
        self.measured: List[Sample] = []
        self.current_sample: Sample = None
        self.current_suggestion: Dict[str, Any] = None

    def add_sample(self, sample: Sample) -> None:
        """
        Add a new measured sample. Sample must include all tunable params
        and a 'visc_profile' key for the 5-point measurement.
        """
        self.current_sample = sample
        self.measured.append(sample)

    def next_suggestion(self) -> Dict[str, Any]:
        """
        Compute and return the next experiment suggestion based on
        the latest sample and measurements.
        """
        # build combined dataset: original + all measured (with viscosity cols)
        measured_df = pd.DataFrame([
            {**m, **dict(zip(self.viscosity_cols, m['visc_profile']))}
            for m in self.measured
        ])
        combined = pd.concat([self.train_df, measured_df], ignore_index=True)

        # find neighbors around the latest sample
        neighbors = self._k_nearest_visc_neighbors(
            self.current_sample, combined
        )

        # get one suggestion
        suggestion = self._suggest_via_kmeans_params(
            neighbors, self.current_sample, n_suggestions=1
        )[0]

        self.current_suggestion = suggestion
        return suggestion

    def summary(self) -> None:
        """
        Print a summary of all measurements entered so far.
        """
        print("\n=== SUMMARY OF MEASUREMENTS ENTERED ===")
        for i, m in enumerate(self.measured, 1):
            vp = ", ".join(f"{v:.3g}" for v in m['visc_profile'])
            print(
                f"{i}: Protein={m['Protein']}, Buffer={m['Buffer']}, "
                f"Surfactant={m['Surfactant']}, Sugar(M)={m['Sugar (M)']}, "
                f"TWEEN={m['TWEEN']} → [{vp}]"
            )

    def _k_nearest_visc_neighbors(
        self,
        sample: Sample,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return k rows whose viscosity profiles are closest to sample['visc_profile'].
        """
        V = df[self.viscosity_cols].values
        target = np.array(sample['visc_profile']).reshape(1, -1)
        dists = np.linalg.norm(V - target, axis=1)
        idx = np.argsort(dists)[:self.k]
        return df.iloc[idx].copy()

    def _suggest_via_kmeans_params(
        self,
        neighbors: pd.DataFrame,
        sample: Sample,
        n_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Cluster neighbors in the tunable parameter space and return medoid(s).
        If sample contains unseen categories, force one suggestion to include it.
        """
        num_cols = ['Protein', 'Sugar (M)', 'TWEEN']
        cat_cols = ['Buffer', 'Surfactant']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ], remainder='drop')
        X = preprocessor.fit_transform(neighbors)
        km = KMeans(n_clusters=n_suggestions, random_state=self.random_state)
        km.fit(X)

        # medoids
        medoids = []
        for center in km.cluster_centers_:
            dists = np.linalg.norm(X - center, axis=1)
            medoids.append(neighbors.iloc[np.argmin(dists)])

        suggestions = [{
            'Protein':    row['Protein'],
            'Buffer':     row['Buffer'],
            'Sugar (M)':  row['Sugar (M)'],
            'Surfactant': row['Surfactant'],
            'TWEEN':      row['TWEEN']
        } for row in medoids]

        # detect unseen categories
        seen_buffers = set(neighbors['Buffer'])
        seen_surfactants = set(neighbors['Surfactant'])
        new_buffer = sample['Buffer'] not in seen_buffers
        new_surfactant = sample['Surfactant'] not in seen_surfactants

        if new_buffer or new_surfactant:
            from collections import Counter
            counts = Counter(km.labels_)
            top_cluster = counts.most_common(1)[0][0]
            top_medoid = neighbors.iloc[np.where(
                km.labels_ == top_cluster)[0][0]]

            forced = {
                'Protein':    top_medoid['Protein'],
                'Buffer':     sample['Buffer'] if new_buffer else top_medoid['Buffer'],
                'Sugar (M)':  top_medoid['Sugar (M)'],
                'Surfactant': sample['Surfactant'] if new_surfactant else top_medoid['Surfactant'],
                'TWEEN':      top_medoid['TWEEN'],
            }
            suggestions.append(forced)

        return suggestions


# Example of interactive wrapper (optional)
if __name__ == "__main__":
    import pandas as pd
    from typing import Optional

    def get_sample_from_input() -> Optional[Sample]:
        sample = {
            'Protein':    float(input("Protein concentration: ")),
            'Buffer':     input("Buffer type: "),
            'Surfactant': input("Surfactant type: "),
            'Sugar (M)':  float(input("Sugar concentration (M): ")),
            'TWEEN':      float(input("TWEEN concentration: ")),
        }
        raw = input(
            "Enter 5-point viscosity profile (comma-separated), or 'quit' to stop: "
        )
        if raw.strip().lower() in ('quit', 'q'):
            return None
        sample['visc_profile'] = np.array([float(x) for x in raw.split(',')])
        return sample

    df = pd.read_csv('content/formulation_data_04222025_2.csv')
    viscosity_cols = [
        'Viscosity100', 'Viscosity1000',
        'Viscosity10000', 'Viscosity100000', 'Viscosity15000000'
    ]

    engine = ExperimentSuggester(df, viscosity_cols)
    init = get_sample_from_input()
    if init is None:
        print("No initial measurement; exiting.")
    else:
        engine.add_sample(init)
        while True:
            suggestion = engine.next_suggestion()
            print("\n--- NEXT EXPERIMENT SUGGESTION ---")
            print(
                f"Protein={suggestion['Protein']}, Buffer={suggestion['Buffer']}, "
                f"Sugar(M)={suggestion['Sugar (M)']}, Surfactant={suggestion['Surfactant']}, "
                f"TWEEN={suggestion['TWEEN']}"
            )
            raw = input(
                "\nEnter measured 5-point viscosity profile (comma-sep), "
                "or 'quit' to end: "
            )
            if raw.strip().lower() in ('quit', 'q'):
                print("Ending sequential suggestions.")
                break
            profile = np.array([float(x) for x in raw.split(',')])
            engine.add_sample({**suggestion, 'visc_profile': profile})

        engine.summary()
