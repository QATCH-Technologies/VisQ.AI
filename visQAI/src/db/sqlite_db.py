#!/usr/bin/env python3
"""
Module: sqlite_db

SQLite database supporting BaseExcipient, VisQExcipient variations, and Formulations.

Updated to store excipient type in base_excipients only and inherit etype for variations.
"""
import os
import json
import uuid
from pathlib import Path

# try:
#     from pysqlcipher3 import dbapi2 as sqlite3
#     _USE_ENCRYPTION = True
# except ImportError:
import sqlite3
_USE_ENCRYPTION = False


def _enable_foreign_keys(conn):
    conn.execute("PRAGMA foreign_keys = ON;")


DB_PATH = Path(__file__).parent.parent / 'data' / 'app.db'


class SQLiteDB:
    """
    SQLite database supporting BaseExcipients, VisQExcipient variations, and Formulations.
    """

    def __init__(self, db_path: Path = DB_PATH, encryption_key: str = None):
        self.db_path = db_path
        self.encryption_key = encryption_key or os.getenv('DB_ENCRYPTION_KEY')
        self._ensure_db_dir()
        self.conn = sqlite3.connect(str(self.db_path))
        if _USE_ENCRYPTION:
            if not self.encryption_key:
                raise ValueError(
                    "Encryption key required for encrypted database")
            self.conn.execute(f"PRAGMA key = '{self.encryption_key}';")
        _enable_foreign_keys(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _ensure_db_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_tables(self):
        with self.conn:
            # Base excipients hold name+etype
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS base_excipients (
                    id         TEXT PRIMARY KEY,
                    etype      TEXT NOT NULL,
                    name       TEXT NOT NULL UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Variations table: inherits etype via foreign key
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS excipients (
                    id            TEXT PRIMARY KEY,
                    base_id       TEXT NOT NULL,
                    concentration REAL NOT NULL,
                    unit          TEXT NOT NULL,
                    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(base_id) REFERENCES base_excipients(id) ON DELETE CASCADE
                );
            """)
            # Formulations unchanged
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS formulations (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    name           TEXT    NOT NULL UNIQUE,
                    notes          TEXT,
                    viscosity_json TEXT,
                    created_at     TEXT    DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS formulation_excipients (
                    formulation_id INTEGER NOT NULL,
                    excipient_id   TEXT    NOT NULL,
                    PRIMARY KEY (formulation_id, excipient_id),
                    FOREIGN KEY(formulation_id) REFERENCES formulations(id) ON DELETE CASCADE,
                    FOREIGN KEY(excipient_id)   REFERENCES excipients(id)   ON DELETE CASCADE
                );
            """)

    # Base Excipients CRUD
    def add_base_excipient(self, etype: str, name: str) -> str:
        """
        Create a new BaseExcipient with type+name.
        """
        bid = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                "INSERT INTO base_excipients (id, etype, name) VALUES (?, ?, ?)",
                (bid, etype, name)
            )
        return bid

    def list_base_excipients(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM base_excipients ORDER BY created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_base_excipient(self, base_id: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT * FROM base_excipients WHERE id=?", (base_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def delete_base_excipient(self, base_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM base_excipients WHERE id=?", (base_id,)
            )

    # VisQ Excipients CRUD
    def add_excipient(self, base_id: str, concentration: float, unit: str) -> str:
        """
        Create a new VisQExcipient variation under a BaseExcipient.
        """
        eid = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                "INSERT INTO excipients (id, base_id, concentration, unit) VALUES (?, ?, ?, ?)",
                (eid, base_id, concentration, unit)
            )
        return eid

    def list_excipients(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT e.id, e.base_id, e.concentration, e.unit, e.created_at,"
            " b.name AS base_name, b.etype"
            " FROM excipients e"
            " JOIN base_excipients b ON e.base_id = b.id"
            " ORDER BY e.created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_excipient(self, exc_id: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT e.id, e.base_id, e.concentration, e.unit, e.created_at,"
            " b.name AS base_name, b.etype"
            " FROM excipients e"
            " JOIN base_excipients b ON e.base_id = b.id"
            " WHERE e.id = ?", (exc_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_excipient(self, exc_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [exc_id]
        with self.conn:
            self.conn.execute(
                f"UPDATE excipients SET {cols} WHERE id=?", vals
            )

    def delete_excipient(self, exc_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM excipients WHERE id=?", (exc_id,)
            )

    # Formulations CRUD (unchanged)
    def add_formulation(self, name: str, notes: str = None, viscosity: dict = None, excipient_ids: list = None) -> int:
        vis_json = json.dumps(viscosity) if viscosity is not None else None
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO formulations (name, notes, viscosity_json) VALUES (?, ?, ?)",
                (name, notes, vis_json)
            )
            fid = cur.lastrowid
            if excipient_ids:
                self._link_excipients(fid, excipient_ids)
            return fid

    def update_formulation(self, formulation_id: int, **fields) -> None:
        exc_ids = fields.pop("excipient_ids", None)
        if "viscosity" in fields:
            fields["viscosity_json"] = json.dumps(fields.pop("viscosity"))
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            vals = list(fields.values()) + [formulation_id]
            with self.conn:
                self.conn.execute(
                    f"UPDATE formulations SET {cols} WHERE id=?", vals
                )
        if exc_ids is not None:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM formulation_excipients WHERE formulation_id=?", (
                        formulation_id,)
                )
                self._link_excipients(formulation_id, exc_ids)

    def _link_excipients(self, formulation_id: int, excipient_ids: list) -> None:
        for eid in excipient_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO formulation_excipients (formulation_id, excipient_id) VALUES (?, ?)",
                (formulation_id, eid)
            )

    def delete_formulation(self, formulation_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "DELETE FROM formulations WHERE id=?", (formulation_id,)
            )

    def list_formulations(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM formulations ORDER BY created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_formulation(self, formulation_id: int) -> dict | None:
        cur = self.conn.execute(
            "SELECT * FROM formulations WHERE id=?", (formulation_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_excipient_ids_for(self, formulation_id: int) -> list[str]:
        cur = self.conn.execute(
            "SELECT excipient_id FROM formulation_excipients WHERE formulation_id=?", (
                formulation_id,)
        )
        return [r[0] for r in cur.fetchall()]

    def drop_database(self) -> None:
        """
        Remove the current database file and recreate an empty one with tables.
        """
        try:
            self.close()
        except Exception:
            pass
        try:
            self.db_path.unlink()
        except FileNotFoundError:
            pass
        self._ensure_db_dir()
        self.conn = sqlite3.connect(str(self.db_path))
        if _USE_ENCRYPTION:
            if not self.encryption_key:
                raise ValueError(
                    "Encryption key required for encrypted database")
            self.conn.execute(f"PRAGMA key = '{self.encryption_key}';")
        _enable_foreign_keys(self.conn)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def update_base_excipient(self, base_id: str, name: str, etype: str) -> None:
        """
        Update the name and etype of an existing BaseExcipient.

        Args:
            base_id: UUID string of the BaseExcipient to update.
            name:   New name.
            etype:  New etype.

        Raises:
            TypeError: if base_id is not a string.
            ValueError: if base_id is not a valid UUID or no record exists.
        """
        if not isinstance(base_id, str):
            raise TypeError(
                "update_base_excipient 'base_id' must be a string.")
        try:
            uuid.UUID(base_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid UUID for base_id: {base_id}")

        # ensure record exists
        cursor = self.conn.execute(
            "SELECT 1 FROM base_excipients WHERE id = ?;",
            (base_id,)
        )
        if cursor.fetchone() is None:
            raise ValueError(f"No BaseExcipient found with id {base_id}")

        # perform the update
        self.conn.execute(
            """
            UPDATE base_excipients
               SET name  = ?,
                   etype = ?
             WHERE id    = ?;
            """,
            (name, etype, base_id)
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
