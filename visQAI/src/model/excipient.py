#!/usr/bin/env python3
"""
Module: excipient

This module provides data models for managing excipients and their concentration-specific variations in the VisQ.AI system. It defines:
  - `ConcentrationUnit`: supported units for concentration values
  - `BaseExcipient`: foundational excipient with a UUID and name
  - `VisQExcipient`: specialized excipient that adds type, concentration, unit, and optional id
  - `ExcipientProfile`: grouping container for multiple `VisQExcipient` variants under a base excipient

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-04-25

Version:
    1.0.1
"""
import uuid
from enum import Enum
from typing import List, Set, Optional


class ConcentrationUnit(Enum):
    """
    Enumeration of supported concentration units for excipients.

    Attributes:
        MOLAR: Molar concentration (M).
        MILLIMOLAR: Millimolar concentration (mM).
        MICROGRAM_PER_ML: Microgram per milliliter (μg/mL).
        MILLIGRAM_PER_ML: Milligram per milliliter (mg/mL).
        PERCENT_W_V: Percent weight/volume (% w/v).
        PERCENT_V_V: Percent volume/volume (% v/v).
    """
    MOLAR = "M"
    MILLIMOLAR = "mM"
    MICROGRAM_PER_ML = "μg/mL"
    MILLIGRAM_PER_ML = "mg/mL"
    PERCENT_W_V = "% w/v"
    PERCENT_V_V = "% v/v"

    def __str__(self) -> str:
        """Return the unit symbol as a string."""
        return self.value


class BaseExcipient:
    """
    Represents the base concept of an excipient with a unique identifier and a name.

    Attributes:
        _id: UUID identifier for the excipient.
        _name: Name of the excipient.
    """

    def __init__(self, name: str, id: Optional[uuid.UUID] = None):
        """
        Initialize a BaseExcipient.

        Args:
            name: Non-empty string name of the excipient.
            id: Optional UUID; if omitted, a new UUID is generated.

        Raises:
            TypeError: If `name` is not a string or `id` is not a UUID.
            ValueError: If `name` is empty or whitespace.
        """
        self._validate_name(name)
        self._name: str = name.strip()
        if id is not None and not isinstance(id, uuid.UUID):
            raise TypeError(
                "BaseExcipient 'id' must be a uuid.UUID if provided.")
        self._id: uuid.UUID = id or uuid.uuid4()

    @staticmethod
    def _validate_name(name: object) -> None:
        """
        Validates the excipient name.

        Args:
            name: Value to validate as a string name.

        Raises:
            TypeError: If name is not a string.
            ValueError: If name is empty or whitespace.
        """
        if not isinstance(name, str):
            raise TypeError("BaseExcipient 'name' must be a string.")
        if not name.strip():
            raise ValueError(
                "BaseExcipient 'name' must be a non-empty string.")

    @property
    def name(self) -> str:
        """str: The name of the excipient."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Sets the name of the excipient after validation.

        Args:
            value: New non-empty string name.

        Raises:
            TypeError: If value is not a string.
            ValueError: If value is empty or whitespace.
        """
        self._validate_name(value)
        self._name = value.strip()

    @property
    def id(self) -> uuid.UUID:
        """uuid.UUID: Read-only unique identifier of the excipient."""
        return self._id

    @id.setter
    def id(self, id: uuid.UUID) -> None:
        """uuid.UUID: Read-only unique identifier of the excipient."""
        self._id = id

    def __eq__(self, other: object) -> bool:
        """
        Determine equality based on UUID.
        """
        return isinstance(other, BaseExcipient) and self._id == other._id

    def __hash__(self) -> int:
        """Returns hash based on the excipient id."""
        return hash(self._id)


class VisQExcipient(BaseExcipient):
    """
    Specialized excipient including type, concentration, and unit.
    Attributes:
        _etype: Category/type of the excipient.
        _concentration: Numeric concentration value.
        _unit: Unit of concentration as ConcentrationUnit.
        _id: Inherited from BaseExcipient.
    """

    def __init__(
        self,
        name: str,
        etype: str,
        concentration: float,
        unit: ConcentrationUnit = ConcentrationUnit.MOLAR,
        id: Optional[uuid.UUID] = None
    ):
        """
        Initialize a VisQExcipient variation.

        Args:
            name: Base name.
            etype: Non-empty string for excipient type/category.
            concentration: Non-negative number for concentration.
            unit: Unit enum for concentration.
            id: Optional UUID; if omitted, a new UUID is generated.

        Raises:
            TypeError: If types of args are invalid.
            ValueError: If etype is empty or concentration is negative.
        """
        super().__init__(name, id=id)
        self._validate_etype(etype)
        self._etype: str = etype.strip()
        self._validate_concentration(concentration)
        self._concentration: float = float(concentration)
        self._validate_unit(unit)
        self._unit: ConcentrationUnit = unit

    @staticmethod
    def _validate_etype(etype: object) -> None:
        """
        Validates the excipient type.
        """
        if not isinstance(etype, str):
            raise TypeError("VisQExcipient 'etype' must be a string.")
        if not etype.strip():
            raise ValueError(
                "VisQExcipient 'etype' must be a non-empty string.")

    @staticmethod
    def _validate_concentration(conc: object) -> None:
        """
        Validates the concentration value.
        """
        if not isinstance(conc, (int, float)):
            raise TypeError("VisQExcipient 'concentration' must be a number.")
        if conc < 0:
            raise ValueError(
                "VisQExcipient 'concentration' must be non-negative.")

    @staticmethod
    def _validate_unit(unit: object) -> None:
        """
        Validates the concentration unit.
        """
        if not isinstance(unit, ConcentrationUnit):
            raise TypeError(
                "VisQExcipient 'unit' must be a ConcentrationUnit.")

    @property
    def etype(self) -> str:
        """str: The type/category of the excipient."""
        return self._etype

    @etype.setter
    def etype(self, value: str) -> None:
        self._validate_etype(value)
        self._etype = value.strip()

    @property
    def concentration(self) -> float:
        """float: The numeric concentration value."""
        return self._concentration

    @concentration.setter
    def concentration(self, value: float) -> None:
        self._validate_concentration(value)
        self._concentration = float(value)

    @property
    def unit(self) -> ConcentrationUnit:
        """ConcentrationUnit: The unit of the concentration."""
        return self._unit

    @unit.setter
    def unit(self, value: ConcentrationUnit) -> None:
        self._validate_unit(value)
        self._unit = value

    def __eq__(self, other: object) -> bool:
        """
        Equality based on name, type, concentration, and unit.
        """
        return (
            isinstance(other, VisQExcipient)
            and self.name == other.name
            and self.etype == other.etype
            and self.concentration == other.concentration
            and self.unit == other.unit
        )

    def __hash__(self) -> int:
        """Hash computed from name, type, concentration, and unit."""
        return hash((self.name, self.etype, self.concentration, self.unit))


class ExcipientProfile:
    """
    Manages a set of VisQExcipient variations under a common base excipient.
    """

    def __init__(self, base_excipient: BaseExcipient):
        """
        Initialize the profile with a base excipient.
        """
        if not isinstance(base_excipient, BaseExcipient):
            raise TypeError(
                "ExcipientProfile requires a BaseExcipient instance.")
        self._base_excipient: BaseExcipient = base_excipient
        self._variations: Set[VisQExcipient] = set()

    @property
    def base_excipient(self) -> BaseExcipient:
        """BaseExcipient: The base excipient of this profile."""
        return self._base_excipient

    @base_excipient.setter
    def base_excipient(self, value: BaseExcipient) -> None:
        if not isinstance(value, BaseExcipient):
            raise TypeError(
                "ExcipientProfile 'base_excipient' must be a BaseExcipient.")
        self._base_excipient = value

    def add_variation(
        self,
        etype: str,
        concentration: float,
        unit: ConcentrationUnit
    ) -> None:
        """
        Adds a new VisQExcipient variation to the profile.
        """
        variant = VisQExcipient(
            name=self.base_excipient.name,
            etype=etype,
            concentration=concentration,
            unit=unit
        )
        if variant in self._variations:
            raise ValueError(
                f"Variation {concentration}{unit} already exists for {self.base_excipient.name}."
            )
        self._variations.add(variant)

    def get_variations(self) -> List[VisQExcipient]:
        """
        Retrieves a list of all variations.
        """
        return list(self._variations)
