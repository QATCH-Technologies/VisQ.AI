import os
import json
from pathlib import Path

try:
    from pysqlcipher3 import dbapi2 as sqlite3
    _USE_ENCRYPTION = True
except ImportError:
    import sqlite3
    _USE_ENCRYPTION = False


def _enable_foreign_keys(conn):
    conn.execute("PRAGMA foreign_keys = ON;")


DB_PATH = Path(__file__).parent.parent / 'data' / 'app.db'


class SQLiteDB:
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
            # Excipients table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS excipients (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                etype         TEXT    NOT NULL,
                name          TEXT    NOT NULL,
                concentration REAL    NOT NULL,
                unit          TEXT    NOT NULL,
                created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Formulations table
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS formulations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT    NOT NULL UNIQUE,
                notes          TEXT,
                viscosity_json TEXT,
                created_at     TEXT    DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Join table linking formulations to excipients
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS formulation_excipients (
                formulation_id INTEGER NOT NULL,
                excipient_id   INTEGER NOT NULL,
                PRIMARY KEY (formulation_id, excipient_id),
                FOREIGN KEY(formulation_id)
                  REFERENCES formulations(id) ON DELETE CASCADE,
                FOREIGN KEY(excipient_id)
                  REFERENCES excipients(id)   ON DELETE CASCADE
            );
            """)

    # ─── Excipients CRUD ─────────────────────────────────────────────────────────
    def add_excipient(self, etype: str, name: str, concentration: float, unit: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO excipients (etype, name, concentration, unit) VALUES (?, ?, ?, ?)",
                (etype, name, concentration, unit)
            )
            return cur.lastrowid

    def update_excipient(self, excipient_id: int, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [excipient_id]
        with self.conn:
            self.conn.execute(f"UPDATE excipients SET {cols} WHERE id=?", vals)

    def delete_excipient(self, excipient_id: int):
        with self.conn:
            self.conn.execute(
                "DELETE FROM excipients WHERE id=?", (excipient_id,))

    def list_excipients(self) -> list:
        cur = self.conn.execute(
            "SELECT * FROM excipients ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def get_excipient(self, excipient_id: int) -> dict:
        cur = self.conn.execute(
            "SELECT * FROM excipients WHERE id=?", (excipient_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ─── Formulations CRUD ────────────────────────────────────────────────────────
    def add_formulation(
        self,
        name: str,
        notes: str = None,
        viscosity: dict = None,
        excipient_ids: list = None
    ) -> int:
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

    def update_formulation(self, formulation_id: int, **fields):
        exc_ids = fields.pop("excipient_ids", None)
        if "viscosity" in fields:
            fields["viscosity_json"] = json.dumps(fields.pop("viscosity"))
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            vals = list(fields.values()) + [formulation_id]
            with self.conn:
                self.conn.execute(
                    f"UPDATE formulations SET {cols} WHERE id=?", vals)
        if exc_ids is not None:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM formulation_excipients WHERE formulation_id=?",
                    (formulation_id,)
                )
                self._link_excipients(formulation_id, exc_ids)

    def _link_excipients(self, formulation_id: int, excipient_ids: list):
        for eid in excipient_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO formulation_excipients (formulation_id, excipient_id) VALUES (?, ?)",
                (formulation_id, eid)
            )

    def delete_formulation(self, formulation_id: int):
        with self.conn:
            self.conn.execute(
                "DELETE FROM formulations WHERE id=?", (formulation_id,))

    def list_formulations(self) -> list:
        cur = self.conn.execute(
            "SELECT * FROM formulations ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def get_formulation(self, formulation_id: int) -> dict:
        cur = self.conn.execute(
            "SELECT * FROM formulations WHERE id=?", (formulation_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_excipient_ids_for(self, formulation_id: int) -> list:
        cur = self.conn.execute(
            "SELECT excipient_id FROM formulation_excipients WHERE formulation_id=?", (
                formulation_id,)
        )
        return [r[0] for r in cur.fetchall()]

    def close(self):
        self.conn.close()
