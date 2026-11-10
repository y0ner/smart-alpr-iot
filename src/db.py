"""Capa de persistencia y repositorio SQLite para Smart ALPR IoT.

Implementa la gestión de conexiones y operaciones transaccionales para
vehículos, accesos y cálculo de permanencia.
"""

import sqlite3
from pathlib import Path
from typing import Any

from src.config import SCHEMA_SQL_PATH, SQLITE_DB_PATH


class DatabaseManager:
    """Administrador de conexiones y operaciones con SQLite."""

    def __init__(self, db_path: Path = SQLITE_DB_PATH) -> None:
        self.db_path = db_path

    def get_connection(self) -> sqlite3.Connection:
        """Obtiene una conexión configurada con soporte para claves foráneas."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self, schema_path: Path = SCHEMA_SQL_PATH) -> None:
        """Inicializa las tablas e índices definidos en el archivo schema.sql."""
        if not schema_path.exists():
            raise FileNotFoundError(f"Esquema SQL no encontrado en {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_script = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_script)

    def ping(self) -> dict[str, Any]:
        """Verifica la conectividad con la base de datos."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            version = cursor.fetchone()[0]
            return {"status": "connected", "sqlite_version": version}
