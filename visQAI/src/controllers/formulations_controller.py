# src/controllers/formulation_controller.py

from src.db.sqlite_db import SQLiteDB
from src.model.formulation import Formulation
from excipients_controller import ExcipientsController
from src.model.viscosity import ViscosityProfile
import json


class FormulationsController:
    def __init__(self):
        self.db = SQLiteDB()
        self.exc_ctrl = ExcipientsController()

    def all(self):
        results = []
        for r in self.db.list_formulations():
            results.append(self._row_to_obj(r))
        return results

    def get(self, form_id) -> Formulation:
        row = self.db.get_formulation(form_id)
        return None if not row else self._row_to_obj(row)

    def add(self, formulation: Formulation):
        # persist the base row
        vis = (formulation.get_viscosity_profile()
               and formulation.get_viscosity_profile().to_dict())
        fid = self.db.add_formulation(
            name=formulation.name,
            notes=formulation.notes,
            viscosity=vis,
            excipient_ids=[e._id for e in formulation.get_excipients()]
        )
        formulation._id = fid
        return formulation

    def edit(self, formulation: Formulation):
        vis = (formulation.get_viscosity_profile()
               and formulation.get_viscosity_profile().to_dict())
        self.db.update_formulation(
            formulation._id,
            name=formulation.name,
            notes=formulation.notes,
            viscosity=vis,
            excipient_ids=[e._id for e in formulation.get_excipients()]
        )

    def delete(self, form_id):
        self.db.delete_formulation(form_id)

    def _row_to_obj(self, row: dict) -> Formulation:
        f = Formulation(name=row["name"])
        f._id = row["id"]
        f.notes = row.get("notes", "")
        # rebuild viscosity_profile
        if row.get("viscosity_json"):
            data = json.loads(row["viscosity_json"])
            f.viscosity_profile = ViscosityProfile.from_dict(data)
        exc_ids = self.db.get_excipient_ids_for(f._id)
        for eid in exc_ids:
            f.add_excipient(self.exc_ctrl.get(eid))
        return f
