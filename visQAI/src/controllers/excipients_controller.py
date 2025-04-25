# visQAI/src/controllers/excipients_controller.py
"""
Module: excipients_controller

This controller provides CRUD operations and profile assembly for BaseExcipients and VisQExcipient variations
in the VisQ.AI SQLite database.

Author: Paul MacNichol (paul.macnichol@qatchtech.com)
Date: 2025-04-25
Version: 1.0.0
"""
import uuid
from typing import List, Optional
from src.db.sqlite_db import SQLiteDB
from src.model.excipient import (
    BaseExcipient,
    VisQExcipient,
    ExcipientProfile,
    ConcentrationUnit
)


class ExcipientsController:
    """
    Controller for managing BaseExcipients, VisQExcipient variations, and profiles in SQLite.

    Provides CRUD operations with input validation and error checking.
    """

    def __init__(self):
        """
        Initializes the ExcipientsController with a SQLiteDB instance.
        """
        self.db = SQLiteDB()

    def list_base_excipients(self) -> List[BaseExcipient]:
        """
        Retrieve all base excipients from the database.

        Returns:
            List[BaseExcipient]: A list of BaseExcipient objects.
        """
        records = self.db.list_base_excipients()
        bases: List[BaseExcipient] = []
        for r in records:
            bid = uuid.UUID(r['id'])
            base = BaseExcipient(name=r['name'], id=bid)
            bases.append(base)
        return bases

    def get_base_excipient(self, base_id: str) -> Optional[BaseExcipient]:
        """
        Retrieve a single base excipient by its UUID string.

        Args:
            base_id (str): UUID string of the base excipient.

        Returns:
            BaseExcipient: The matching BaseExcipient or None if not found.

        Raises:
            TypeError: If base_id is not a string.
            ValueError: If base_id is not a valid UUID.
        """
        if not isinstance(base_id, str):
            raise TypeError("get_base_excipient 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(
                f"get_base_excipient 'base_id' is not a valid UUID: {base_id}")
        r = self.db.get_base_excipient(base_id)
        if r is None:
            return None
        bid = uuid.UUID(r['id'])
        return BaseExcipient(name=r['name'], id=bid)

    def add_base_excipient(self, base: BaseExcipient) -> BaseExcipient:
        """
        Add a new base excipient.

        Args:
            base (BaseExcipient): BaseExcipient instance to add.

        Returns:
            BaseExcipient: Newly created BaseExcipient with generated ID.

        Raises:
            TypeError: If base is not a BaseExcipient instance.
        """
        if not isinstance(base, BaseExcipient):
            raise TypeError(
                "add_base_excipient requires a BaseExcipient instance.")
        new_id = self.db.add_base_excipient(name=base.name)
        return BaseExcipient(name=base.name, id=uuid.UUID(new_id))

    def delete_base_excipient(self, base_id: str) -> None:
        """
        Delete a base excipient by its UUID string.

        Args:
            base_id (str): UUID string of the base excipient to delete.

        Raises:
            TypeError: If base_id is not a string.
            ValueError: If base_id is not a valid UUID.
        """
        if not isinstance(base_id, str):
            raise TypeError(
                "delete_base_excipient 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(
                f"delete_base_excipient 'base_id' is not a valid UUID: {base_id}")
        self.db.delete_base_excipient(base_id)

    def list_variations(self) -> List[VisQExcipient]:
        """
        Retrieve all VisQExcipient variations.

        Returns:
            List[VisQExcipient]: A list of VisQExcipient objects.
        """
        records = self.db.list_excipients()
        result: List[VisQExcipient] = []
        for r in records:
            vid = uuid.UUID(r['id'])
            unit = ConcentrationUnit(r['unit']) if r['unit'] else None
            exc = VisQExcipient(
                name=r['base_name'],
                etype=r['etype'],
                concentration=r['concentration'],
                unit=unit,
                id=vid
            )
            result.append(exc)
        return result

    def get_variation(self, var_id: str) -> Optional[VisQExcipient]:
        """
        Retrieve a single VisQExcipient variation by its UUID string.

        Args:
            var_id (str): UUID string of the variation.

        Returns:
            VisQExcipient: The matching VisQExcipient or None if not found.

        Raises:
            TypeError: If var_id is not a string.
            ValueError: If var_id is not a valid UUID.
        """
        if not isinstance(var_id, str):
            raise TypeError("get_variation 'var_id' must be a string.")
        try:
            uuid.UUID(var_id)
        except ValueError:
            raise ValueError(
                f"get_variation 'var_id' is not a valid UUID: {var_id}")
        r = self.db.get_excipient(var_id)
        if r is None:
            return None
        vid = uuid.UUID(r['id'])
        unit = ConcentrationUnit(r['unit']) if r['unit'] else None
        return VisQExcipient(
            name=r['base_name'],
            etype=r['etype'],
            concentration=r['concentration'],
            unit=unit,
            id=vid
        )

    def add_variation(self, base_id: str, exc: VisQExcipient) -> VisQExcipient:
        """
        Add a new VisQExcipient variation under a specified base.

        Args:
            base_id (str): UUID string of the BaseExcipient.
            exc (VisQExcipient): VisQExcipient instance to add.

        Returns:
            VisQExcipient: Newly created VisQExcipient with assigned ID.

        Raises:
            TypeError: If base_id is not a string or exc is not a VisQExcipient.
            ValueError: If base_id is not a valid UUID.
        """
        if not isinstance(base_id, str):
            raise TypeError("add_variation 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(
                f"add_variation 'base_id' is not a valid UUID: {base_id}")
        if not isinstance(exc, VisQExcipient):
            raise TypeError("add_variation requires a VisQExcipient instance.")
        new_id = self.db.add_excipient(
            base_id=base_id,
            etype=exc.etype,
            concentration=exc.concentration,
            unit=str(exc.unit)
        )
        return VisQExcipient(
            name=exc.name,
            etype=exc.etype,
            concentration=exc.concentration,
            unit=exc.unit,
            id=uuid.UUID(new_id)
        )

    def update_variation(self, exc: VisQExcipient) -> None:
        """
        Update an existing VisQExcipient variation.

        Args:
            exc (VisQExcipient): VisQExcipient with updated values and valid ID.

        Raises:
            TypeError: If exc is not a VisQExcipient.
            ValueError: If exc.id is not a valid UUID or does not exist.
        """
        if not isinstance(exc, VisQExcipient):
            raise TypeError(
                "update_variation requires a VisQExcipient instance.")
        var_id = exc.id
        if not isinstance(var_id, uuid.UUID):
            raise ValueError("update_variation 'exc.id' must be a UUID.")
        if not self.db.get_excipient(str(var_id)):
            raise ValueError(f"Variation with id {var_id} does not exist.")
        self.db.update_excipient(
            str(var_id),
            etype=exc.etype,
            concentration=exc.concentration,
            unit=str(exc.unit)
        )

    def delete_variation(self, var_id: str) -> None:
        """
        Delete a variation by its UUID string.

        Args:
            var_id (str): UUID string of the variation to delete.

        Raises:
            TypeError: If var_id is not a string.
            ValueError: If var_id is not a valid UUID.
        """
        if not isinstance(var_id, str):
            raise TypeError("delete_variation 'var_id' must be a string.")
        try:
            uuid.UUID(var_id)
        except ValueError:
            raise ValueError(
                f"delete_variation 'var_id' is not a valid UUID: {var_id}")
        self.db.delete_excipient(var_id)

    def get_profile(self, base_id: str) -> Optional[ExcipientProfile]:
        """
        Retrieve an ExcipientProfile for a given base, including its variations.

        Args:
            base_id (str): UUID string of the BaseExcipient.

        Returns:
            ExcipientProfile or None if base not found.

        Raises:
            TypeError: If base_id is not a string.
            ValueError: If base_id is not a valid UUID.
        """
        if not isinstance(base_id, str):
            raise TypeError("get_profile 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(
                f"get_profile 'base_id' is not a valid UUID: {base_id}")
        base = self.get_base_excipient(base_id)
        if base is None:
            return None
        profile = ExcipientProfile(base)
        records = self.db.list_excipients()
        for r in records:
            if r['base_id'] == base_id:
                unit = ConcentrationUnit(r['unit']) if r['unit'] else None
                profile.add_variation(
                    etype=r['etype'],
                    concentration=r['concentration'],
                    unit=unit
                )
        return profile
