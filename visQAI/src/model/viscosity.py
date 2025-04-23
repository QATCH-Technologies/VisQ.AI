import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from scipy.interpolate import interp1d


class ViscosityProfile:
    """
    Represents a viscosity profile composed of multiple ViscosityPoints,
    with log-log interpolation support.
    """

    class ViscosityPoint:
        """
        Represents a single viscosity value at a specific shear rate.
        """

        def __init__(self, shear_rate: float, viscosity: float):
            self.shear_rate = shear_rate
            self.viscosity = viscosity

        def __repr__(self):
            return f"({self.shear_rate} 1/s, {self.viscosity} cP)"

    def __init__(self, viscosities: List[float], shear_rates: List[float] = None):
        shear_rates = shear_rates or [100, 1000, 10000, 100000, 15000000]
        if len(viscosities) != len(shear_rates):
            raise ValueError(
                "Viscosities and shear_rates must be the same length.")

        self._points = [
            self.ViscosityPoint(sr, vis) for sr, vis in zip(shear_rates, viscosities)
        ]
        self._update_interpolator()

    def _update_interpolator(self):
        shear_rates = [p.shear_rate for p in self._points]
        viscosities = [p.viscosity for p in self._points]
        self._log_interp = interp1d(
            np.log10(shear_rates),
            np.log10(viscosities),
            bounds_error=False,
            fill_value="extrapolate"
        )

    def get_viscosity(self, shear_rate: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Returns interpolated viscosity at one or more shear rates.
        """
        if isinstance(shear_rate, list):
            log_interp_vals = self._log_interp(np.log10(shear_rate))
            return list(10 ** log_interp_vals)
        else:
            return float(10 ** self._log_interp(np.log10(shear_rate)))

    def get_profile(self) -> List['ViscosityProfile.ViscosityPoint']:
        return self._points

    def set_profile(self, viscosities: List[float], shear_rates: List[float]):
        if len(viscosities) != len(shear_rates):
            raise ValueError(
                "Mismatch between viscosity and shear rate lengths.")
        self._points = [
            self.ViscosityPoint(sr, vis) for sr, vis in zip(shear_rates, viscosities)
        ]
        self._update_interpolator()

    def get_point(self, shear_rate: float) -> Union['ViscosityProfile.ViscosityPoint', None]:
        for point in self._points:
            if point.shear_rate == shear_rate:
                return point
        return None

    def add_point(self, shear_rate: float, viscosity: float):
        self._points.append(self.ViscosityPoint(shear_rate, viscosity))
        self._points.sort(key=lambda p: p.shear_rate)
        self._update_interpolator()

    def plot(self, loglog: bool = True, title: str = "Viscosity Profile"):
        x = [p.shear_rate for p in self._points]
        y = [p.viscosity for p in self._points]

        plt.figure(figsize=(6, 4))
        if loglog:
            plt.loglog(x, y, marker='o')
        else:
            plt.plot(x, y, marker='o')

        plt.xlabel("Shear Rate (1/s)")
        plt.ylabel("Viscosity (cP)")
        plt.title(title)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    def compute_log_slope(self) -> float:
        x = np.log10([p.shear_rate for p in self._points])
        y = np.log10([p.viscosity for p in self._points])
        slope, _ = np.polyfit(x, y, deg=1)
        return slope

    def compare_to(self, other: 'ViscosityProfile') -> Dict[float, float]:
        """
        Compare to another profile at this profile's shear rates.
        """
        return {
            p.shear_rate: abs(p.viscosity - other.get_viscosity(p.shear_rate))
            for p in self._points
        }

    def __repr__(self):
        return f"ViscosityProfile({self._points})"
