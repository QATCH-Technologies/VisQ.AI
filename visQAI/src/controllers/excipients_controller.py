#!/usr/bin/env python3
"""
Module: excipients_controller

This controller provides CRUD operations and profile assembly for BaseExcipient and its VisQExcipient variations
in the VisQ.AI SQLite database, now leveraging the unified BaseExcipient (name+etype).

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-05-01

Version:
    1.1.0
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
    Controller for managing BaseExcipient entries, VisQExcipient variations, and profiles in SQLite.

    Provides full CRUD with input validation and error checking.
    """

    def __init__(self):
        self.db = SQLiteDB()

    def list_base_excipients(self) -> List[BaseExcipient]:
        """
        Retrieve all base excipients (with name+etype) from the database.
        """
        records = self.db.list_base_excipients()
        bases: List[BaseExcipient] = []
        for r in records:
            bid = uuid.UUID(r['id'])
            # BaseExcipient now requires etype
            base = BaseExcipient(
                name=r['name'],
                etype=r['etype'],
                id=bid
            )
            bases.append(base)
        return bases

    def get_base_excipient(self, base_id: str) -> Optional[BaseExcipient]:
        """
        Retrieve a single BaseExcipient by UUID string.
        """
        if not isinstance(base_id, str):
            raise TypeError("get_base_excipient 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(f"Invalid UUID for base_id: {base_id}")

        r = self.db.get_base_excipient(base_id)
        if r is None:
            return None
        bid = uuid.UUID(r['id'])
        return BaseExcipient(
            name=r['name'],
            etype=r['etype'],
            id=bid
        )

    def add_base_excipient(self, base: BaseExcipient) -> BaseExcipient:
        """
        Add a new BaseExcipient (with name+etype) to the database.
        """
        if not isinstance(base, BaseExcipient):
            raise TypeError(
                "add_base_excipient requires a BaseExcipient instance.")
        # ensure etype is passed through
        new_id = self.db.add_base_excipient(
            name=base.name,
            etype=base.etype
        )
        return BaseExcipient(
            name=base.name,
            etype=base.etype,
            id=uuid.UUID(new_id)
        )

    def delete_base_excipient(self, base_id: str) -> None:
        """
        Delete a BaseExcipient by UUID string.
        """
        if not isinstance(base_id, str):
            raise TypeError(
                "delete_base_excipient 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(f"Invalid UUID for base_id: {base_id}")
        self.db.delete_base_excipient(base_id)

    def list_variations(self) -> List[VisQExcipient]:
        """
        Retrieve all VisQExcipient variations.
        """
        records = self.db.list_excipients()
        result: List[VisQExcipient] = []
        for r in records:
            vid = uuid.UUID(r['id'])
            unit = ConcentrationUnit(r['unit'])
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
        Retrieve a single VisQExcipient variation by UUID string.
        """
        if not isinstance(var_id, str):
            raise TypeError("get_variation 'var_id' must be a string.")
        try:
            uuid.UUID(var_id)
        except ValueError:
            raise ValueError(f"Invalid UUID for var_id: {var_id}")
        r = self.db.get_excipient(var_id)
        if r is None:
            return None
        vid = uuid.UUID(r['id'])
        unit = ConcentrationUnit(r['unit'])
        return VisQExcipient(
            name=r['base_name'],
            etype=r['etype'],
            concentration=r['concentration'],
            unit=unit,
            id=vid
        )

    def add_variation(self, base_id: str, exc: VisQExcipient) -> VisQExcipient:
        """
        Add a new VisQExcipient variation under an existing BaseExcipient.
        """
        if not isinstance(base_id, str):
            raise TypeError("add_variation 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(f"Invalid UUID for base_id: {base_id}")
        if not isinstance(exc, VisQExcipient):
            raise TypeError("add_variation requires a VisQExcipient instance.")
        new_id = self.db.add_excipient(
            base_id=base_id,
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
        """
        if not isinstance(exc, VisQExcipient):
            raise TypeError(
                "update_variation requires a VisQExcipient instance.")
        var_id = exc.id
        if not isinstance(var_id, uuid.UUID):
            raise ValueError("update_variation 'exc.id' must be a UUID.")
        if not self.db.get_excipient(str(var_id)):
            raise ValueError(f"Variation with id {var_id} does not exist.")
        # Only concentration and unit are stored in excipients table
        print(self.get_variation(var_id=var_id))
        self.db.update_excipient(
            str(var_id),
            concentration=exc.concentration,
            unit=str(exc.unit)
        )
        print(self.get_variation(var_id=var_id))

    def delete_variation(self, var_id: str) -> None:
        """
        Delete a VisQExcipient variation by UUID string.
        """
        if not isinstance(var_id, str):
            raise TypeError("delete_variation 'var_id' must be a string.")
        try:
            uuid.UUID(var_id)
        except ValueError:
            raise ValueError(f"Invalid UUID for var_id: {var_id}")
        self.db.delete_excipient(var_id)

    def get_profile(self, base_id: str) -> Optional[ExcipientProfile]:
        """
        Retrieve a profile (ExcipientProfile) for a BaseExcipient, including all its variations.
        """
        if not isinstance(base_id, str):
            raise TypeError("get_profile 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except ValueError:
            raise ValueError(f"Invalid UUID for base_id: {base_id}")
        base = self.get_base_excipient(base_id)
        if base is None:
            return None
        profile = ExcipientProfile(base)
        records = self.db.list_excipients()
        for r in records:
            if r['base_id'] == base_id:
                unit = ConcentrationUnit(r['unit'])
                # add_variation now inherits etype from base
                profile.add_variation(
                    concentration=r['concentration'],
                    unit=unit
                )
        return profile

    def update_base_excipient(self, base: BaseExcipient) -> None:
        """
        Update an existing BaseExcipient's name and etype in the database.

        Args:
            base: the BaseExcipient instance with the new name and/or etype and a valid UUID.

        Raises:
            TypeError: if `base` is not a BaseExcipient.
            ValueError: if `base.id` is not a valid UUID or if no such record exists.
        """
        if not isinstance(base, BaseExcipient):
            raise TypeError(
                "update_base_excipient requires a BaseExcipient instance.")
        base_id = base.id
        if not isinstance(base_id, uuid.UUID):
            raise ValueError("update_base_excipient 'base.id' must be a UUID.")
        existing = self.db.get_base_excipient(str(base_id))
        if existing is None:
            raise ValueError(
                f"BaseExcipient with id {base_id} does not exist.")

        self.db.update_base_excipient(
            base_id=str(base_id),
            name=base.name,
            etype=base.etype
        )
