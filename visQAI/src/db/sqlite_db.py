#!/usr/bin/env python3
"""
Module: sqlite_db

SQLite database supporting BaseExcipient, VisQExcipient variations, and Formulations.

Updated to store excipient type in base_excipients only and inherit etype for variations.
All Formulations CRUD and join-table operations are abstracted here for use by controllers.
"""
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional

# try:
#     from pysqlcipher3 import dbapi2 as sqlite3
#     _USE_ENCRYPTION = True
# except ImportError:
import sqlite3
_USE_ENCRYPTION = False


def _enable_foreign_keys(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")


DB_PATH = Path(__file__).parent.parent / 'data' / 'app.db'


class SQLiteDB:
    """
    SQLite database supporting BaseExcipients, VisQExcipient variations, and Formulations.
    Provides table creation and all CRUD operations for formulations and excipients.
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

    def _ensure_db_dir(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_tables(self) -> None:
        """
        Create all necessary tables for excipients and formulations.
        """
        with self.conn:
            # Base excipients
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS base_excipients (
                    id         TEXT PRIMARY KEY,
                    etype      TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, etype)
                );
                """
            )
            # Variations table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS excipients (
                    id            TEXT PRIMARY KEY,
                    base_id       TEXT NOT NULL,
                    concentration REAL NOT NULL,
                    unit          TEXT NOT NULL,
                    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(base_id) REFERENCES base_excipients(id) ON DELETE CASCADE
                );
                """
            )
            # Formulations
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS formulations (
                    id             TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    notes          TEXT,
                    viscosity_json TEXT,
                    created_at     TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Join table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS formulation_excipients (
                    formulation_id TEXT NOT NULL,
                    excipient_id   TEXT NOT NULL,
                    PRIMARY KEY (formulation_id, excipient_id),
                    FOREIGN KEY(formulation_id) REFERENCES formulations(id) ON DELETE CASCADE,
                    FOREIGN KEY(excipient_id)   REFERENCES excipients(id)   ON DELETE CASCADE
                );
                """
            )

    # ---------- BaseExcipients ----------

    def add_base_excipient(self, etype: str, name: str) -> str:
        """
        Create a new BaseExcipient.
        """
        bid = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                "INSERT INTO base_excipients (id, etype, name) VALUES (?, ?, ?)",
                (bid, etype, name)
            )
        return bid

    def update_base_excipient(self, base_id: str, etype: str, name: str) -> None:
        if not isinstance(base_id, str):
            raise TypeError("base_id must be a string")
        try:
            uuid.UUID(base_id)
        except Exception:
            raise ValueError(f"Invalid UUID: {base_id}")
        with self.conn:
            cur = self.conn.execute(
                "UPDATE base_excipients SET etype=?, name=? WHERE id=?",
                (etype, name, base_id)
            )
            if cur.rowcount == 0:
                raise ValueError(f"BaseExcipient not found: {base_id}")

    def list_base_excipients(self) -> List[dict]:
        cur = self.conn.execute(
            "SELECT id, etype, name, created_at FROM base_excipients ORDER BY created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_base_excipient(self, base_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT id, etype, name, created_at FROM base_excipients WHERE id=?",
            (base_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_base_excipient_by_name(self, name: str) -> Optional[dict]:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        name = name.strip()
        if not name:
            raise ValueError("name cannot be empty")
        cur = self.conn.execute(
            "SELECT id, etype, name, created_at FROM base_excipients WHERE name=?",
            (name,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def delete_base_excipient(self, base_id: str) -> None:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM base_excipients WHERE id=?",
                (base_id,)
            )
            if cur.rowcount == 0:
                raise ValueError(f"BaseExcipient not found: {base_id}")

    # ---------- VisQExcipients ----------

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

    def update_excipient(self, exc_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [exc_id]
        with self.conn:
            cur = self.conn.execute(
                f"UPDATE excipients SET {cols} WHERE id=?", vals
            )
            if cur.rowcount == 0:
                raise ValueError(f"Excipient not found: {exc_id}")

    def list_excipients(self) -> List[dict]:
        cur = self.conn.execute(
            "SELECT e.id, e.base_id, e.concentration, e.unit, e.created_at,"
            " b.name AS base_name, b.etype"
            " FROM excipients e"
            " JOIN base_excipients b ON e.base_id=b.id"
            " ORDER BY e.created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def get_excipient(self, exc_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT e.id, e.base_id, e.concentration, e.unit, e.created_at,"
            " b.name AS base_name, b.etype"
            " FROM excipients e"
            " JOIN base_excipients b ON e.base_id=b.id"
            " WHERE e.id=?",
            (exc_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def delete_excipient(self, exc_id: str) -> None:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM excipients WHERE id=?",
                (exc_id,)
            )
            if cur.rowcount == 0:
                raise ValueError(f"Excipient not found: {exc_id}")

    # ---------- Formulations ----------

    def create_formulations_table(self) -> None:
        """Ensure the formulations table exists."""
        self._create_tables()

    def create_formulation_excipients_table(self) -> None:
        """Ensure the formulation_excipients join table exists."""
        self._create_tables()

    def insert_formulation(self, id: str, name: str, notes: str, viscosity_json: Optional[str]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO formulations (id, name, notes, viscosity_json) VALUES (?, ?, ?, ?)",
                (id, name, notes, viscosity_json)
            )

    def fetch_all_formulations(self) -> List[tuple]:
        cur = self.conn.execute(
            "SELECT id, name, notes, viscosity_json FROM formulations ORDER BY created_at DESC"
        )
        return [tuple(r) for r in cur.fetchall()]

    def fetch_formulation_by_id(self, id: str) -> Optional[tuple]:
        cur = self.conn.execute(
            "SELECT id, name, notes, viscosity_json FROM formulations WHERE id=?",
            (id,)
        )
        row = cur.fetchone()
        return tuple(row) if row else None

    def exists_formulation(self, id: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM formulations WHERE id=?", (id,)
        )
        return cur.fetchone() is not None

    def update_formulation(self, id: str, name: str, notes: str, viscosity_json: Optional[str]) -> None:
        with self.conn:
            cur = self.conn.execute(
                "UPDATE formulations SET name=?, notes=?, viscosity_json=? WHERE id=?",
                (name, notes, viscosity_json, id)
            )
            if cur.rowcount == 0:
                raise ValueError(f"Formulation not found: {id}")

    def remove_formulation_excipients(self, formulation_id: str) -> None:
        self.conn.execute(
            "DELETE FROM formulation_excipients WHERE formulation_id=?",
            (formulation_id,)
        )

    def add_formulation_excipient(self, formulation_id: str, excipient_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO formulation_excipients (formulation_id, excipient_id) VALUES (?, ?)",
            (formulation_id, excipient_id)
        )

    def fetch_excipients_for_formulation(self, formulation_id: str) -> List[str]:
        cur = self.conn.execute(
            "SELECT excipient_id FROM formulation_excipients WHERE formulation_id=?",
            (formulation_id,)
        )
        return [r[0] for r in cur.fetchall()]

    def delete_formulation(self, id: str) -> None:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM formulations WHERE id=?",
                (id,)
            )
            if cur.rowcount == 0:
                raise ValueError(f"Formulation not found: {id}")

    def drop_database(self) -> None:
        """
        Remove the current database file and recreate an empty one with tables.
        """
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.db_path.unlink()
        except FileNotFoundError:
            pass
        self.__init__(self.db_path, self.encryption_key)

    def close(self) -> None:
        self.conn.close()
