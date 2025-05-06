import pickle
import joblib
from pathlib import Path
from typing import Any
import os


class MockPredictor:
    def predict(self, *args, **kwargs) -> str:
        return '[PLACEHOLDER]'


def save_predictor(predictor: MockPredictor, path: str) -> None:
    file_path = Path(path)
    ext = file_path.suffix.lower()

    if ext in ('.pkl', '.pickle'):
        with open(file_path, 'wb') as f:
            pickle.dump(predictor, f)
    elif ext == '.joblib':
        joblib.dump(predictor, file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use '.pkl', '.pickle', or '.joblib'."
        )


def load_predictor(path: str) -> Any:

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Predictor file not found: {path}")

    ext = file_path.suffix.lower()

    if ext in ('.pkl', '.pickle'):
        with open(file_path, 'rb') as f:
            predictor = pickle.load(f)
    elif ext == '.joblib':
        predictor = joblib.load(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use '.pkl', '.pickle', or '.joblib'."
        )

    if not hasattr(predictor, 'predict'):
        raise ValueError(
            f"Loaded object from {path} is not a valid predictor.")

    return predictor


if __name__ == '__main__':
    predictor = MockPredictor()
    for ext in ['.pkl', '.pickle', '.joblib']:
        file_path = os.path.join('visQAI', 'objects', f'mock_predictor{ext}')
        save_predictor(predictor, file_path)

        loaded = load_predictor(file_path)
        print(f"{file_path}: {loaded.predict()}")  # Outputs: PLACEHOLDER"}]}
