from typing import List, Optional
from .excipient import Excipient
from .viscosity import ViscosityProfile


class Formulation:
    def __init__(self, name: str):
        self.name = name
        self.excipients: List[Excipient] = []
        self.notes: Optional[str] = ""
        self.viscosity_profile: Optional[ViscosityProfile] = None

    def add_notes(self, notes: str):
        self.notes = notes

    def remove_notes(self):
        self.notes = ""

    def add_excipient(self, excipient: Excipient):
        self.excipients.append(excipient)

    def remove_excipient_by_name(self, name: str):
        self.excipients = [
            e for e in self.excipients if e.get_name() != name]

    def get_excipients(self) -> List[Excipient]:
        return self.excipients

    def get_excipients_by_category(self, category: str) -> List[Excipient]:
        return [e for e in self.excipients if e.category() == category.lower()]

    def set_viscosity_profile(self, profile: ViscosityProfile):
        self.viscosity_profile = profile

    def get_viscosity_profile(self) -> Optional[ViscosityProfile]:
        return self.viscosity_profile

    def summary(self) -> str:
        lines = [f"Formulation: {self.name}", "-" * 40]
        if not self.excipients:
            lines.append("No excipients added.")
        else:
            for exc in self.excipients:
                lines.append(f"  • {exc.info()}")

        if self.viscosity_profile:
            lines.append("\nViscosity Profile (log-slope: "
                         f"{self.viscosity_profile.compute_log_slope():.2f}):")
            for rate, visc in self.viscosity_profile.get_profile().items():
                lines.append(f"  • {rate} 1/s: {visc:.2f} cP")
        else:
            lines.append("\nNo viscosity profile set.")

        return "\n".join(lines)

    def __repr__(self):
        return f"Formulation(name='{self.name}', excipients={len(self.excipients)})"
