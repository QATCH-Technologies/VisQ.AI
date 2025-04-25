# from collections import Counter
# from typing import List, Dict, Any
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from viscosity_profile import ViscosityProfile
# from excipient import Protein, Buffer, Surfactant, Sugar, ConcentrationUnit
# from formulation import Formulation


class ExperimentSuggester:
    pass
    #     def __init__(
    #         self,
    #         train_df: pd.DataFrame,
    #         viscosity_cols: List[str],
    #         k: int = 50,
    #         random_state: int = 0
    #     ):
    #         self.train_df = train_df.copy()
    #         self.viscosity_cols = viscosity_cols
    #         self.k = k
    #         self.random_state = random_state
    #         self.measured: List[Formulation] = []
    #         self.current_sample: Formulation = None
    #         self.current_suggestion: Formulation = None

    #     def add_sample(self, formulation: Formulation) -> None:
    #         self.current_sample = formulation
    #         self.measured.append(formulation)

    #     def next_suggestion(self) -> Formulation:
    #         measured_dicts = [
    #             {
    #                 **{e.category().capitalize(): e.get_concentration()
    #                    for e in f.get_excipients()},
    #                 **dict(zip(self.viscosity_cols,
    #                            f.get_viscosity_profile().viscosities()))
    #             } for f in self.measured
    #         ]
    #         measured_df = pd.DataFrame(measured_dicts)
    #         combined = pd.concat([self.train_df, measured_df], ignore_index=True)

    #         neighbors = self._k_nearest_visc_neighbors(
    #             self.current_sample, combined)
    #         suggestion_dict = self._suggest_via_kmeans_params(
    #             neighbors, self.current_sample, 1)[0]
    #         self.current_suggestion = self._build_formulation_from_dict(
    #             suggestion_dict)
    #         return self.current_suggestion

    #     def summary(self) -> None:
    #         print("\n=== SUMMARY OF MEASUREMENTS ENTERED ===")
    #         for i, f in enumerate(self.measured, 1):
    #             print(f"{i}:")
    #             print(f.summary())

    #     def _k_nearest_visc_neighbors(self, sample: Formulation, df: pd.DataFrame) -> pd.DataFrame:
    #         V = df[self.viscosity_cols].values
    #         target = np.array(
    #             sample.get_viscosity_profile().viscosities()).reshape(1, -1)
    #         dists = np.linalg.norm(V - target, axis=1)
    #         idx = np.argsort(dists)[:self.k]
    #         return df.iloc[idx].copy()

    #     def _suggest_via_kmeans_params(
    #         self,
    #         neighbors: pd.DataFrame,
    #         sample: Formulation,
    #         n_suggestions: int = 5
    #     ) -> List[Dict[str, Any]]:
    #         num_cols = ['Protein', 'Sugar (M)', 'TWEEN']
    #         cat_cols = ['Buffer', 'Surfactant']

    #         preprocessor = ColumnTransformer([
    #             ('num', StandardScaler(), num_cols),
    #             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    #         ])
    #         X = preprocessor.fit_transform(neighbors)
    #         km = KMeans(n_clusters=n_suggestions, random_state=self.random_state)
    #         km.fit(X)

    #         medoids = []
    #         for center in km.cluster_centers_:
    #             dists = np.linalg.norm(X - center, axis=1)
    #             medoids.append(neighbors.iloc[np.argmin(dists)])

    #         suggestions = [row.to_dict() for row in medoids]

    #         seen_buffers = set(neighbors['Buffer'])
    #         seen_surfactants = set(neighbors['Surfactant'])
    #         sample_dict = self._formulation_to_dict(sample)
    #         new_buffer = sample_dict['Buffer'] not in seen_buffers
    #         new_surfactant = sample_dict['Surfactant'] not in seen_surfactants

    #         if new_buffer or new_surfactant:
    #             counts = Counter(km.labels_)
    #             top_cluster = counts.most_common(1)[0][0]
    #             top_medoid = neighbors.iloc[np.where(
    #                 km.labels_ == top_cluster)[0][0]]
    #             forced = top_medoid.to_dict()
    #             if new_buffer:
    #                 forced['Buffer'] = sample_dict['Buffer']
    #             if new_surfactant:
    #                 forced['Surfactant'] = sample_dict['Surfactant']
    #             suggestions.append(forced)

    #         return suggestions

    #     def _formulation_to_dict(self, f: Formulation) -> Dict[str, Any]:
    #         return {
    #             **{e.category().capitalize(): e.get_concentration() for e in f.get_excipients()}
    #         }

    #     def _build_formulation_from_dict(self, d: Dict[str, Any]) -> Formulation:
    #         f = Formulation("Suggested")
    #         f.add_excipient(
    #             Protein("Protein", d['Protein'], ConcentrationUnit.MILLIGRAM_PER_ML))
    #         f.add_excipient(Buffer(d['Buffer'], 10, ConcentrationUnit.MILLIMOLAR))
    #         f.add_excipient(
    #             Sugar("Sugar", d['Sugar (M)'], ConcentrationUnit.MOLAR))
    #         f.add_excipient(Surfactant(
    #             d['Surfactant'], d['TWEEN'], ConcentrationUnit.PERCENT_V_V))
    #         return f
