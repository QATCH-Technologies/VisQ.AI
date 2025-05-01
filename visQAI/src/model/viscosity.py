"""
Module: viscosity

Provides data models and CRUD operations for managing viscosity measurements
in a profile, including individual points and profile-level operations, as well as
serialization helpers.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-04-25

Version:
    1.2.0
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class ViscosityPoint:
    """
    Represents a viscosity measurement at a specific shear rate.

    Attributes:
        shear_rate: Non-negative shear rate value (1/s).
        viscosity: Non-negative viscosity value (cP).
    """
    shear_rate: float
    viscosity: float

    def __post_init__(self):
        """
        Validate that shear_rate and viscosity are non-negative numbers.

        Raises:
            TypeError: If values are not numeric.
            ValueError: If values are negative.
        """
        if not isinstance(self.shear_rate, (int, float)):
            raise TypeError("ViscosityPoint 'shear_rate' must be a number.")
        if not isinstance(self.viscosity, (int, float)):
            raise TypeError("ViscosityPoint 'viscosity' must be a number.")
        if self.shear_rate < 0:
            raise ValueError(
                "ViscosityPoint 'shear_rate' must be non-negative.")
        if self.viscosity < 0:
            raise ValueError(
                "ViscosityPoint 'viscosity' must be non-negative.")


class ViscosityProfile:
    """
    Maintains a collection of ViscosityPoint objects and provides CRUD operations
    and dictionary serialization.

    Attributes:
        _points: Internal list of ViscosityPoint instances.
    """

    def __init__(self):
        """
        Initialize an empty viscosity profile.
        """
        self._points: List[ViscosityPoint] = []

    @property
    def points(self) -> List[ViscosityPoint]:
        """
        List[ViscosityPoint]: Read-only list of viscosity points, sorted by shear_rate.
        """
        return sorted(self._points.copy(), key=lambda p: p.shear_rate)

    def add_point(self, shear_rate: float, viscosity: float) -> None:
        """
        Add a new viscosity point.

        Args:
            shear_rate: Non-negative shear rate.
            viscosity: New non-negative viscosity.

        Raises:
            ValueError: If a point at the same shear rate already exists.
            TypeError: If inputs are not numeric.
        """
        new_point = ViscosityPoint(shear_rate, viscosity)
        if any(p.shear_rate == new_point.shear_rate for p in self._points):
            raise ValueError(
                f"A point at shear rate {shear_rate} already exists.")
        self._points.append(new_point)

    def get_point(self, shear_rate: float) -> Optional[ViscosityPoint]:
        """
        Retrieve a viscosity point by shear rate.

        Args:
            shear_rate: Shear rate to look up.

        Returns:
            The matching ViscosityPoint or None if not found.

        Raises:
            TypeError: If shear_rate is not numeric.
        """
        if not isinstance(shear_rate, (int, float)):
            raise TypeError("get_point 'shear_rate' must be a number.")
        for p in self._points:
            if p.shear_rate == shear_rate:
                return p
        return None

    def update_point(self, shear_rate: float, viscosity: float) -> None:
        """
        Update the viscosity for an existing shear rate.

        Args:
            shear_rate: Shear rate of the point to update.
            viscosity: New non-negative viscosity value.

        Raises:
            KeyError: If no point exists at the given shear rate.
            TypeError: If inputs are not numeric.
        """
        if not isinstance(shear_rate, (int, float)):
            raise TypeError("update_point 'shear_rate' must be a number.")
        if not isinstance(viscosity, (int, float)):
            raise TypeError("update_point 'viscosity' must be a number.")
        for idx, p in enumerate(self._points):
            if p.shear_rate == shear_rate:
                self._points[idx] = ViscosityPoint(shear_rate, viscosity)
                return
        raise KeyError(f"No viscosity point found at shear rate {shear_rate}.")

    def remove_point(self, shear_rate: float) -> None:
        """
        Remove a viscosity point by shear rate.

        Args:
            shear_rate: Shear rate of the point to delete.

        Raises:
            KeyError: If no point exists at the given shear rate.
            TypeError: If shear_rate is not numeric.
        """
        if not isinstance(shear_rate, (int, float)):
            raise TypeError("remove_point 'shear_rate' must be a number.")
        for idx, p in enumerate(self._points):
            if p.shear_rate == shear_rate:
                del self._points[idx]
                return
        raise KeyError(f"No viscosity point found at shear rate {shear_rate}.")

    def clear(self) -> None:
        """
        Remove all viscosity points from the profile.
        """
        self._points.clear()

    def to_dict(self) -> Dict[float, float]:
        """
        Convert the viscosity profile to a dictionary mapping shear_rate to viscosity.

        Returns:
            Dict where keys are shear rates and values are viscosities.
        """
        return {p.shear_rate: p.viscosity for p in self.points}

    @classmethod
    def from_dict(cls, data: Dict[Any, Any]) -> "ViscosityProfile":
        """
        Create a ViscosityProfile from a dictionary mapping shear_rate to viscosity.

        Args:
            data: Dict with shear_rate as key (numeric or str) and viscosity as value.

        Returns:
            A populated ViscosityProfile instance.

        Raises:
            ValueError: If dict contains invalid rates or viscosities.
        """
        vp = cls()
        for rate_key, visc in data.items():
            try:
                rate = float(rate_key)
                visc_val = float(visc)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid data in dict: {rate_key}: {visc}")
            vp.add_point(rate, visc_val)
        return vp
