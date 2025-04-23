from abc import ABC, abstractmethod
from enum import Enum


class ConcentrationUnit(Enum):
    MOLAR = "M"
    MILLIMOLAR = "mM"
    MICROGRAM_PER_ML = "μg/mL"
    MILLIGRAM_PER_ML = "mg/mL"
    PERCENT_W_V = "% w/v"
    PERCENT_V_V = "% v/v"

    def __str__(self):
        return self.value


class Excipient(ABC):
    def __init__(self, name: str, concentration: float, unit: ConcentrationUnit = ConcentrationUnit.MOLAR):
        self._name = name
        self._concentration = concentration
        self._unit = unit

    def get_name(self) -> str:
        return self._name

    def get_concentration(self) -> float:
        return self._concentration

    def get_unit(self) -> ConcentrationUnit:
        return self._unit

    def set_name(self, name: str):
        self._name = name

    def set_concentration(self, concentration: float):
        self._concentration = concentration

    def set_unit(self, unit: ConcentrationUnit):
        if not isinstance(unit, ConcentrationUnit):
            raise ValueError(
                "Unit must be an instance of ConcentrationUnit enum.")
        self._unit = unit

    @abstractmethod
    def get_function(self) -> str:
        pass

    @abstractmethod
    def category(self) -> str:
        pass

    def __repr__(self):
        return (f"{self.__class__.__name__}(name='{self._name}', "
                f"concentration={self._concentration} {self._unit})")

    def info(self) -> str:
        return (f"{self._name} ({self.category()}): {self._concentration} {self._unit} "
                f"- {self.get_function()}")


class Surfactant(Excipient):
    def get_function(self) -> str:
        return "Reduces surface tension and prevents aggregation."

    def category(self) -> str:
        return "surfactant"


class Protein(Excipient):
    def get_function(self) -> str:
        return "Acts as the active pharmaceutical ingredient or stabilizer."

    def category(self) -> str:
        return "protein"


class Buffer(Excipient):
    def get_function(self) -> str:
        return "Maintains pH and stabilizes the chemical environment."

    def category(self) -> str:
        return "buffer"


class Sugar(Excipient):
    def get_function(self) -> str:
        return "Stabilizes proteins and prevents aggregation during stress."

    def category(self) -> str:
        return "sugar"
