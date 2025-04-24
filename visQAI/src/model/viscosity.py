from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ViscosityPoint:
    """
    Represents a viscosity measurement at a specific shear rate.
    """
    shear_rate: float
    viscosity: float


class ViscosityProfile:
    """
    Maintains a collection of viscosity points (shear_rate -> viscosity) and provides CRUD operations.
    """

    def __init__(self):
        self._profile: List[ViscosityPoint] = []

    def add_point(self, shear_rate: float, viscosity: float) -> None:
        """
        Create a new viscosity point. Raises ValueError if a point at the same shear rate exists.
        """
        if any(p.shear_rate == shear_rate for p in self._profile):
            raise ValueError(
                f"A point at shear rate {shear_rate} already exists.")
        self._profile.append(ViscosityPoint(shear_rate, viscosity))

    def get_point(self, shear_rate: float) -> Optional[ViscosityPoint]:
        """
        Retrieve the viscosity point for the given shear rate, or None if not found.
        """
        for p in self._profile:
            if p.shear_rate == shear_rate:
                return p
        return None

    def update_point(self, shear_rate: float, viscosity: float) -> None:
        """
        Update the viscosity value for an existing shear rate. Raises KeyError if not found.
        """
        for p in self._profile:
            if p.shear_rate == shear_rate:
                p.viscosity = viscosity
                return
        raise KeyError(f"No viscosity point found at shear rate {shear_rate}.")

    def remove_point(self, shear_rate: float) -> None:
        """
        Delete the viscosity point at the specified shear rate. Raises KeyError if not found.
        """
        for i, p in enumerate(self._profile):
            if p.shear_rate == shear_rate:
                del self._profile[i]
                return
        raise KeyError(f"No viscosity point found at shear rate {shear_rate}.")

    def list_points(self) -> List[ViscosityPoint]:
        return sorted(self._profile, key=lambda p: p.shear_rate)

    def clear(self) -> None:
        """
        Remove all viscosity points from the profile.
        """
        self._profile.clear()
