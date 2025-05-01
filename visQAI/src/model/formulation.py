# visQAI/src/model/formulation.py
"""
Module: formulation

Domain model for a formulation, managing its name, a list of concentration-specific excipients,
optional notes, and an optional viscosity profile.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-04-25

Version:
    1.0.0
"""
from typing import List, Optional
from .excipient import VisQExcipient
from .viscosity import ViscosityProfile


class Formulation:
    """
    Domain model for a formulation:
      - name: formulation identifier
      - excipients: list of concentration-specific VisQExcipients
      - notes: optional textual notes
      - viscosity_profile: optional ViscosityProfile
    """

    def __init__(self, name: str):
        """
        Initialize a Formulation.

        Args:
            name: Non-empty string identifying the formulation.

        Raises:
            TypeError: If name is not a string.
            ValueError: If name is empty or whitespace.
        """
        self._validate_name(name)
        self._name: str = name.strip()
        self._excipients: List[VisQExcipient] = []
        self._notes: str = ""
        self._viscosity_profile: Optional[ViscosityProfile] = None

    @staticmethod
    def _validate_name(name: object) -> None:
        """
        Validates the formulation name.

        Args:
            name: Value to validate.

        Raises:
            TypeError: If name is not a string.
            ValueError: If name is empty or whitespace.
        """
        if not isinstance(name, str):
            raise TypeError("Formulation 'name' must be a string.")
        if not name.strip():
            raise ValueError("Formulation 'name' must be a non-empty string.")

    @property
    def name(self) -> str:
        """The formulation's name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Sets a new name after validation.

        Args:
            value: Non-empty string.

        Raises:
            TypeError: If value is not a string.
            ValueError: If value is empty or whitespace.
        """
        self._validate_name(value)
        self._name = value.strip()

    @property
    def excipients(self) -> List[VisQExcipient]:
        """List of added excipients."""
        return list(self._excipients)

    def add_excipient(self, exc: VisQExcipient) -> None:
        """
        Add a VisQExcipient to the formulation, avoiding duplicates.

        Args:
            exc: VisQExcipient instance to add.

        Raises:
            TypeError: If exc is not a VisQExcipient.
        """
        if not isinstance(exc, VisQExcipient):
            raise TypeError("add_excipient requires a VisQExcipient instance.")
        if exc not in self._excipients:
            self._excipients.append(exc)

    def remove_excipient_by_name(self, name: str) -> None:
        """
        Remove all excipients matching the given name.

        Args:
            name: Name string to match.

        Raises:
            TypeError: If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("remove_excipient_by_name requires a string.")
        self._excipients = [e for e in self._excipients if e.name != name]

    def get_excipients_by_type(self, etype: str) -> List[VisQExcipient]:
        """
        Filter excipients by their type (case-insensitive).

        Args:
            etype: Type/category string.

        Returns:
            List of matching VisQExcipients.

        Raises:
            TypeError: If etype is not a string.
        """
        if not isinstance(etype, str):
            raise TypeError("get_excipients_by_type requires a string.")
        return [e for e in self._excipients if e.etype.lower() == etype.lower()]

    @property
    def notes(self) -> str:
        """Optional notes about the formulation."""
        return self._notes

    @notes.setter
    def notes(self, value: str) -> None:
        """
        Sets notes for the formulation.

        Args:
            value: Notes string (may be empty).

        Raises:
            TypeError: If value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("Formulation 'notes' must be a string.")
        self._notes = value

    @property
    def viscosity_profile(self) -> Optional[ViscosityProfile]:
        """Optional ViscosityProfile for the formulation."""
        return self._viscosity_profile

    @viscosity_profile.setter
    def viscosity_profile(self, profile: Optional[ViscosityProfile]) -> None:
        """
        Sets the viscosity profile.

        Args:
            profile: ViscosityProfile instance or None.

        Raises:
            TypeError: If profile is not a ViscosityProfile or None.
        """
        if profile is not None and not isinstance(profile, ViscosityProfile):
            raise TypeError(
                "viscosity_profile must be a ViscosityProfile or None.")
        self._viscosity_profile = profile

    def summary(self) -> str:
        """
        Returns a textual summary of the formulation, including excipients and viscosity.
        """
        lines = [f"Formulation: {self.name}", "-" * 40]
        if not self._excipients:
            lines.append("No excipients added.")
        else:
            for e in self._excipients:
                lines.append(
                    f"  • {e.name} ({e.etype}) @ {e.concentration}{e.unit}")
        if self._notes:
            lines.append(f"Notes: {self._notes}")
        if self._viscosity_profile:
            log_slope = self._viscosity_profile.compute_log_slope()
            lines.append(f"\nViscosity Profile (log-slope: {log_slope:.2f}):")
            for rate, visc in self._viscosity_profile.get_profile().items():
                lines.append(f"  • {rate} 1/s: {visc:.2f} cP")
        else:
            lines.append("\nNo viscosity profile set.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Human-readable representation.
        """
        return f"<Formulation {self.name!r}: {len(self._excipients)} excipients>"
