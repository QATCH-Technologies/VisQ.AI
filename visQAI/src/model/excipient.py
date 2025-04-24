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
    def __init__(self, name: str, etype: str, concentration: float, unit: ConcentrationUnit = ConcentrationUnit.MOLAR):
        self._id = id
        self._name = name
        self._etype = etype
        self._concentration = concentration
        self._unit = unit

    def set_id(self, id: int):
        self._id = id

    def get_id(self):
        return self._id

    def get_excipient_type(self):
        return self._etype

    def set_excipient_type(self, etype: str):
        self._etype = etype

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
