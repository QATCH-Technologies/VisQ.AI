import os
import joblib
from abc import ABC, abstractmethod


class AbstractPredictor(ABC):
    """
    Abstract base class for all predictors.
    Defines the interface for predict and update methods, and manages a name attribute.
    """

    def __init__(self, name: str = None):
        # Assign a default name based on class if none provided
        self._name = name or self.__class__.__name__  # internal storage

    @property
    def name(self) -> str:
        """
        Get the predictor's name.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the predictor's name.
        """
        self._name = value

    @abstractmethod
    def predict(self, data):
        """
        Generate a viscosity profile prediction from input data.
        :param data: Raw or preprocessed input data.
        :return: Predicted viscosity profile.
        """
        pass

    @abstractmethod
    def update(self, data):
        """
        Update the predictor with new data (e.g., for online learning).
        :param data: New data to incorporate into the model.
        """
        pass


class GenericPredictor:
    """
    Generic predictor loader and container for user-defined predictors.

    Provides accessors and mutators for metadata:
      - predictor_name
      - model_directory
      - preprocessor_obj
      - base_model_obj
      - predictor_class_obj
      - full metadata dict via get_metadata()

    Expects the following files in `model_dir`:
      - `preprocessor.pkl`: a fitted preprocessing pipeline
      - `model.pkl`: the trained predictive model object
      - `predictor_class.pkl`: the user-defined predictor class (not instance) that inherits from AbstractPredictor

    The user-defined predictor class should implement __init__(preprocessor, base_model), __init__(model_dir), or __init__(name).
    """

    def __init__(self, model_dir: str, name: str = None):
        self._model_dir = model_dir
        self._preprocessor = self._load_pickle('preprocessor.pkl')
        self._base_model = self._load_pickle('model.pkl')
        self._predictor_cls = self._load_pickle('predictor_class.pkl')
        self._validate_predictor_class()

        # instantiate the user-defined predictor
        self._predictor: AbstractPredictor = self._instantiate_predictor(name)

    def _load_pickle(self, filename: str):
        path = os.path.join(self._model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return joblib.load(path)

    def _validate_predictor_class(self):
        if not issubclass(self._predictor_cls, AbstractPredictor):
            raise TypeError(
                "Loaded predictor_class must inherit from AbstractPredictor"
            )

    def _instantiate_predictor(self, name: str = None) -> AbstractPredictor:
        """
        Instantiate the user-defined predictor class.
        Supports signatures:
          - __init__(preprocessor, base_model, name)
          - __init__(preprocessor, base_model)
          - __init__(model_dir, name)
          - __init__(model_dir)
          - __init__(name)
        """
        cls = self._predictor_cls
        try:
            return cls(self._preprocessor, self._base_model, name)
        except TypeError:
            try:
                return cls(self._preprocessor, self._base_model)
            except TypeError:
                try:
                    return cls(self._model_dir, name)
                except TypeError:
                    try:
                        return cls(self._model_dir)
                    except TypeError:
                        # fallback: only name
                        return cls(name)

    def predict(self, raw_data):
        """
        Preprocess raw_data if possible and delegate to user predictor's predict.
        :param raw_data: Raw input data (e.g., pandas DataFrame).
        :return: Predicted viscosity profile.
        """
        try:
            processed = self._preprocessor.transform(raw_data)
        except Exception:
            processed = raw_data
        return self._predictor.predict(processed)

    def update(self, raw_data):
        """
        Delegate update to user predictor.
        :param raw_data: New data for online update.
        :return: Result of update call.
        """
        return self._predictor.update(raw_data)

    # --- Accessors / Mutators ---

    @property
    def predictor_name(self) -> str:
        """Get the name of the underlying predictor."""
        return self._predictor.name

    @predictor_name.setter
    def predictor_name(self, value: str):
        """Set the name of the underlying predictor."""
        self._predictor.name = value

    @property
    def model_directory(self) -> str:
        """Path to the saved model directory."""
        return self._model_dir

    @property
    def preprocessor_obj(self):
        """Access the loaded preprocessor object."""
        return self._preprocessor

    @property
    def base_model_obj(self):
        """Access the loaded base model object."""
        return self._base_model

    @property
    def predictor_class_obj(self):
        """Access the loaded predictor class."""
        return self._predictor_cls

    def get_metadata(self) -> dict:
        """
        Retrieve metadata about this generic predictor and its components.
        :return: Dictionary with keys: name, model_dir, predictor_class,
                 preprocessor_type, base_model_type
        """
        return {
            'name': self.predictor_name,
            'model_dir': self.model_directory,
            'predictor_class': self._predictor_cls.__name__,
            'preprocessor_type': type(self._preprocessor).__name__,
            'base_model_type': type(self._base_model).__name__,
        }

    def set_metadata(self, name: str = None):
        """
        Set metadata properties. Currently only supports updating the name.
        :param name: New name for the predictor.
        """
        if name:
            self.predictor_name = name
