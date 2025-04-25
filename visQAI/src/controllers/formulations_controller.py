# src/controllers/formulation_controller.py

from src.db.sqlite_db import SQLiteDB
from src.model.formulation import Formulation
from .excipients_controller import ExcipientsController
from src.model.viscosity import ViscosityProfile
import json
from typing import List, Optional


class FormulationsController:
    """
    Controller to manage Formulation objects, each of which contains
    concentration-specific (VisQ) excipients.
    """

    def __init__(self):
        self.db = SQLiteDB()
        self.exc_ctrl = ExcipientsController()

    def all(self) -> List[Formulation]:
        records = self.db.list_formulations()
        formulations: List[Formulation] = []
        for rec in records:
            fid = rec['id']
            exc_ids = self.db.get_excipient_ids_for(fid)
            excs = [self.exc_ctrl.get(eid) for eid in exc_ids]
            viscosity = json.loads(rec['viscosity_json']) if rec.get(
                'viscosity_json') else None
            formulations.append(
                Formulation(
                    name=rec['name'],
                    excipients=excs,
                    notes=rec.get('notes'),
                    viscosity=viscosity,
                    _id=fid
                )
            )
        return formulations

    def get(self, form_id: int) -> Optional[Formulation]:
        rec = self.db.get_formulation(form_id)
        if not rec:
            return None
        exc_ids = self.db.get_excipient_ids_for(form_id)
        excs = [self.exc_ctrl.get(eid) for eid in exc_ids]
        viscosity = json.loads(rec['viscosity_json']) if rec.get(
            'viscosity_json') else None
        return Formulation(
            name=rec['name'],
            excipients=excs,
            notes=rec.get('notes'),
            viscosity=viscosity,
            _id=form_id
        )

    def add(self, form: Formulation) -> Formulation:
        exc_ids = [str(exc.id) for exc in form.excipients]
        new_id = self.db.add_formulation(
            name=form.name,
            notes=form.notes,
            viscosity=form.viscosity,
            excipient_ids=exc_ids
        )
        form.id = new_id
        return form

    def edit(self, form: Formulation) -> None:
        exc_ids = [str(exc.id) for exc in form.excipients]
        self.db.update_formulation(
            formulation_id=form.id,
            name=form.name,
            notes=form.notes,
            viscosity=form.viscosity,
            excipient_ids=exc_ids
        )

    def delete(self, form_id: int) -> None:
        self.db.delete_formulation(form_id)
