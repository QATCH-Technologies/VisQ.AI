# src/controllers/formulations_controller.py

from uuid import UUID
import json
from typing import List, Optional

from src.db.sqlite_db import SQLiteDB
from src.model.formulation import Formulation
from .excipients_controller import ExcipientsController
from src.model.viscosity import ViscosityProfile


class FormulationsController:
    """
    Controller to manage Formulation objects, each of which contains
    concentration-specific (VisQ) excipients, identified by UUIDs.
    """

    def __init__(self):
        self.db = SQLiteDB()
        self.exc_ctrl = ExcipientsController()

    def all(self) -> List[Formulation]:
        """
        Retrieve all formulations from the database.
        """
        records = self.db.list_formulations()
        formulations: List[Formulation] = []
        for rec in records:
            # parse UUID from stored string
            form_id = UUID(rec['id'])
            form = Formulation(name=rec['name'], id=form_id)

            # notes
            form.notes = rec.get('notes', '') or ''

            # viscosity profile
            vis_json = rec.get('viscosity_json')
            if vis_json:
                data = json.loads(vis_json)
                vp = ViscosityProfile()
                for rate, visc in data.items():
                    vp.add_point(float(rate), float(visc))
                form.viscosity_profile = vp

            # excipients
            exc_ids = self.db.get_excipient_ids_for(str(form_id))
            for eid in exc_ids:
                exc = self.exc_ctrl.get(eid)
                if exc:
                    form.add_excipient(exc)

            formulations.append(form)
        return formulations

    def get(self, form_id: UUID) -> Optional[Formulation]:
        """
        Retrieve a single formulation by UUID.
        """
        rec = self.db.get_formulation(str(form_id))
        if not rec:
            return None
        form = Formulation(name=rec['name'], id=form_id)
        form.notes = rec.get('notes', '') or ''

        vis_json = rec.get('viscosity_json')
        if vis_json:
            data = json.loads(vis_json)
            vp = ViscosityProfile()
            for rate, visc in data.items():
                vp.add_point(float(rate), float(visc))
            form.viscosity_profile = vp

        exc_ids = self.db.get_excipient_ids_for(str(form_id))
        for eid in exc_ids:
            exc = self.exc_ctrl.get(eid)
            if exc:
                form.add_excipient(exc)

        return form

    def add(self, form: Formulation) -> Formulation:
        """
        Insert a new formulation into the database and assign its UUID.
        """
        exc_ids = [str(exc.id) for exc in form.excipients]
        new_id_str = self.db.add_formulation(
            name=form.name,
            notes=form.notes,
            viscosity=form.viscosity_profile.to_dict(),
            excipient_ids=exc_ids
        )
        new_uuid = UUID(new_id_str)
        form.id = new_uuid
        return form

    def edit(self, form: Formulation) -> None:
        """
        Update an existing formulation in the database by UUID.
        """
        exc_ids = [str(exc.id) for exc in form.excipients]
        self.db.update_formulation(
            formulation_id=str(form.id),
            name=form.name,
            notes=form.notes,
            viscosity=form.viscosity_profile.to_dict(),
            excipient_ids=exc_ids
        )

    def delete(self, form_id: UUID) -> None:
        """
        Delete a formulation from the database by UUID.
        """
        self.db.delete_formulation(str(form_id))
