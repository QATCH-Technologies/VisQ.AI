# src/controllers/excipient_controller.py

from src.db.sqlite_db import SQLiteDB
from src.model.excipient import Excipient


class ExcipientsController:
    def __init__(self):
        self.db = SQLiteDB()

    def all(self):
        rows = self.db.list_excipients()
        return [Excipient(
            etype=r["etype"],
            name=r["name"],
            concentration=r["concentration"],
            unit=r["unit"],
            _id=r["id"]
        ) for r in rows]

    def get(self, exc_id):
        r = self.db.get_excipient(exc_id)
        return None if not r else Excipient(
            etype=r["etype"],
            name=r["name"],
            concentration=r["concentration"],
            unit=r["unit"],
            _id=r["id"]
        )

    def add(self, exc: Excipient):
        new_id = self.db.add_excipient(
            etype=exc.get_excipient_type(),
            name=exc.get_name(),
            concentration=exc.get_concentration(),
            unit=exc.get_unit()
        )
        exc.set_id(new_id)
        return exc

    def edit(self, exc: Excipient):
        self.db.update_excipient(
            exc.get_id(),
            etype=exc.get_excipient_type,
            name=exc.get_name(),
            concentration=exc.get_concentration(),
            unit=exc.get_unit()
        )

    def delete(self, exc_id):
        self.db.delete_excipient(exc_id)
