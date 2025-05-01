#!/usr/bin/env python3
"""
Module: formulations_controller

Provides CRUD operations for the Formulation domain model.

All SQL interactions are delegated to the SQLiteDB abstraction layer.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-05-01

Version:
    1.1.0
"""
import json
from uuid import UUID
from typing import List, Optional

from src.db.sqlite_db import SQLiteDB
from src.model.formulation import Formulation
from src.model.viscosity import ViscosityProfile
from src.controllers.excipients_controller import ExcipientsController


class FormulationsController:
    """
    Controller for managing Formulation objects.
    Delegates all raw SQL queries to the SQLiteDB layer for separation of concerns.
    """

    def __init__(self, db: Optional[SQLiteDB] = None):
        """
        Initialize controller and ensure DB schema is in place.

        Args:
            db: Optional SQLiteDB instance. If omitted, a new one is created.
        """
        self.db = db or SQLiteDB()
        self.ex_ctrl = ExcipientsController()
        # Ensure tables exist via DB methods
        self.db.create_formulations_table()
        self.db.create_formulation_excipients_table()

    def list_formulations(self) -> List[Formulation]:
        """
        Retrieve all formulations.

        Returns:
            List of Formulation instances.
        """
        rows = self.db.fetch_all_formulations()
        return [self._row_to_formulation(row) for row in rows]

    def get_formulation(self, formulation_id: str) -> Formulation:
        """
        Fetch a single formulation by its UUID.

        Args:
            formulation_id: UUID string.

        Returns:
            Formulation instance.

        Raises:
            TypeError: If the ID is not a valid UUID.
            ValueError: If not found.
        """
        try:
            fid = UUID(formulation_id)
        except Exception:
            raise TypeError("get_formulation requires a valid UUID string.")

        row = self.db.fetch_formulation_by_id(str(fid))
        if not row:
            raise ValueError(
                f"Formulation with id {formulation_id} not found.")
        return self._row_to_formulation(row)

    def add_formulation(self, formulation: Formulation) -> Formulation:
        """
        Persist a new formulation and its excipient links.

        Args:
            formulation: Formulation instance.

        Returns:
            The same Formulation (ID pre-set).

        Raises:
            TypeError: If input is not Formulation.
        """
        if not isinstance(formulation, Formulation):
            raise TypeError("add_formulation requires a Formulation instance.")

        vis_json = (
            json.dumps(formulation.viscosity_profile.to_dict())
            if formulation.viscosity_profile else None
        )

        self.db.insert_formulation(
            id=str(formulation.id),
            name=formulation.name,
            notes=formulation.notes,
            viscosity_json=vis_json
        )
        for exc in formulation.excipients:
            self.db.add_formulation_excipient(
                formulation_id=str(formulation.id),
                excipient_id=str(exc.id)
            )
        return formulation

    def update_formulation(self, formulation: Formulation) -> None:
        """
        Update metadata, excipients, and viscosity for an existing formulation.

        Args:
            formulation: Updated Formulation instance.

        Raises:
            TypeError: If not Formulation.
            ValueError: If formulation does not exist.
        """
        if not isinstance(formulation, Formulation):
            raise TypeError(
                "update_formulation requires a Formulation instance.")

        if not self.db.exists_formulation(str(formulation.id)):
            raise ValueError(
                f"Formulation with id {formulation.id} does not exist.")

        vis_json = (
            json.dumps(formulation.viscosity_profile.to_dict())
            if formulation.viscosity_profile else None
        )

        self.db.update_formulation(
            id=str(formulation.id),
            name=formulation.name,
            notes=formulation.notes,
            viscosity_json=vis_json
        )
        # Reset excipient links
        self.db.remove_formulation_excipients(str(formulation.id))
        for exc in formulation.excipients:
            self.db.add_formulation_excipient(
                formulation_id=str(formulation.id),
                excipient_id=str(exc.id)
            )

    def delete_formulation(self, formulation_id: str) -> None:
        """
        Remove a formulation and its linked excipients.

        Args:
            formulation_id: UUID string to delete.

        Raises:
            TypeError: If ID invalid.
            ValueError: If not found.
        """
        try:
            fid = UUID(formulation_id)
        except Exception:
            raise TypeError("delete_formulation requires a valid UUID string.")

        if not self.db.exists_formulation(str(fid)):
            raise ValueError(
                f"Formulation with id {formulation_id} does not exist.")

        self.db.delete_formulation(str(fid))

    def _row_to_formulation(self, row: tuple) -> Formulation:
        """
        Map a DB row to a Formulation, loading excipients and viscosity.

        Args:
            row: Tuple of fields from fetch.

        Returns:
            Populated Formulation.
        """
        fid_str, name, notes, vis_json = row
        formulation = Formulation(name=name, id=UUID(fid_str))
        formulation.notes = notes or ""

        if vis_json:
            try:
                data = json.loads(vis_json)
                formulation.viscosity_profile = ViscosityProfile()
                formulation.viscosity_profile.from_dict(data)
            except Exception as e:
                raise ValueError(f"Error parsing viscosity JSON: {e}")

        # Load excipient IDs and fetch each variation
        exc_ids = self.db.fetch_excipients_for_formulation(fid_str)
        for eid in exc_ids:
            exc = self.ex_ctrl.get_variation(eid)
            formulation.add_excipient(exc)

        return formulation
