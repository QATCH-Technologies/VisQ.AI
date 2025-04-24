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


class Excipient:
    def __init__(self, name: str, excipient_type: str, concentration: float, unit: ConcentrationUnit = ConcentrationUnit.MOLAR):
        self._name = name
        self._excipient_type = excipient_type
        self._concentration = concentration
        self._unit = unit

    def get_excipient_type(self):
        return self._excipient_type

    def set_excipient_type(self, excipient_type: str):
        self._excipient_type = excipient_type

    def get_name(self) -> str:
        return self._name

    def set_name(self, name: str):
        self._name = name

    def get_concentration(self) -> float:
        return self._concentration

    def get_unit(self) -> ConcentrationUnit:
        return self._unit

    def set_concentration(self, concentration: float):
        self._concentration = concentration

    def set_unit(self, unit: ConcentrationUnit):
        if not isinstance(unit, ConcentrationUnit):
            raise ValueError(
                "Unit must be an instance of ConcentrationUnit enum.")
        self._unit = unit
