"""Capa de persistencia y repositorio SQLite para Smart ALPR IoT.

Implementa la gestión de conexiones transaccionales, registro inteligente de
accesos con alternancia de sentido, cálculo exacto de tiempos de permanencia y
consultas analíticas agregadas para el cliente.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.config import SCHEMA_SQL_PATH, SQLITE_DB_PATH

VehicleType = Literal["CARRO", "MOTO"]
AccessDirection = Literal["ENTRADA", "SALIDA"]


class DatabaseManager:
    """Administrador de conexiones y operaciones relacionales con SQLite."""

    def __init__(self, db_path: Path | str = SQLITE_DB_PATH) -> None:
        self.db_path_str: str = str(db_path)
        self._is_memory: bool = self.db_path_str == ":memory:"
        self._shared_conn: sqlite3.Connection | None = None

        if self._is_memory:
            # En bases de datos en memoria, mantener una conexión abierta persistente
            # mientras viva la instancia para evitar que SQLite libere la RAM entre llamadas
            self._shared_conn = sqlite3.connect(self.db_path_str)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            Path(self.db_path_str).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager que entrega una conexión con FK activas y autocommit/rollback."""
        if self._is_memory and self._shared_conn is not None:
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path_str)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def close(self) -> None:
        """Cierra la conexión compartida en memoria si existe."""
        if self._shared_conn is not None:
            self._shared_conn.close()
            self._shared_conn = None

    def init_db(self, schema_path: Path = SCHEMA_SQL_PATH) -> None:
        """Inicializa las tablas e índices definidos en el archivo schema.sql."""
        if not schema_path.exists():
            raise FileNotFoundError(f"Esquema SQL no encontrado en {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_script = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_script)

    def ping(self) -> dict[str, Any]:
        """Verifica la conectividad y versión de SQLite."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version();")
            row = cursor.fetchone()
            version = row[0] if row else "unknown"
            return {"status": "connected", "sqlite_version": version}

    # -------------------------------------------------------------------------
    # OPERACIONES DE VEHÍCULOS
    # -------------------------------------------------------------------------

    def upsert_vehiculo(self, placa: str, tipo: VehicleType) -> None:
        """Inserta o actualiza un vehículo asegurando integridad referencial."""
        placa_clean = placa.strip().upper()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO vehiculos (placa, tipo)
                VALUES (?, ?)
                ON CONFLICT(placa) DO UPDATE SET tipo = excluded.tipo;
                """,
                (placa_clean, tipo),
            )

    def get_vehiculo(self, placa: str) -> dict[str, Any] | None:
        """Obtiene los datos de un vehículo por su placa."""
        placa_clean = placa.strip().upper()
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT placa, tipo, creado_en FROM vehiculos WHERE placa = ?;",
                (placa_clean,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # GESTIÓN INTELIGENTE DE ACCESOS Y PERMANENCIA
    # -------------------------------------------------------------------------

    def get_ultimo_acceso(self, placa: str) -> dict[str, Any] | None:
        """Obtiene el registro más reciente de acceso de una placa."""
        placa_clean = placa.strip().upper()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, placa, momento, sentido, confianza, ruta_captura, preprocesado
                FROM accesos
                WHERE placa = ?
                ORDER BY momento DESC, id DESC
                LIMIT 1;
                """,
                (placa_clean,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def registrar_acceso(
        self,
        placa: str,
        tipo: VehicleType,
        confianza: float,
        sentido: AccessDirection | None = None,
        ruta_captura: str | None = None,
        preprocesado: bool = True,
        momento: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Registra un acceso con alternancia inteligente y cálculo de permanencia.

        Si sentido es None, determina automáticamente el sentido:
        - Si el último movimiento fue ENTRADA, alterna a SALIDA.
        - Si no tiene registros previos o el último fue SALIDA, asigna ENTRADA.

        Si el movimiento resulta en SALIDA, calcula el tiempo de permanencia
        respecto a la última ENTRADA correspondiente.
        """
        placa_clean = placa.strip().upper()

        # Normalizar timestamp a formato ISO estándar compatible con SQLite
        if momento is None:
            momento_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(momento, datetime):
            momento_str = momento.strftime("%Y-%m-%d %H:%M:%S")
        else:
            momento_str = str(momento)

        ultimo_acceso = self.get_ultimo_acceso(placa_clean)

        # Alternancia inteligente de sentido si no se especifica explícitamente
        if sentido is None:
            if ultimo_acceso and ultimo_acceso["sentido"] == "ENTRADA":
                sentido_final: AccessDirection = "SALIDA"
            else:
                sentido_final = "ENTRADA"
        else:
            sentido_final = sentido

        # Asegurar existencia del vehículo en tabla vehiculos (Foreign Key)
        self.upsert_vehiculo(placa_clean, tipo)

        # Inserción atómica del acceso
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO accesos (placa, momento, sentido, confianza, ruta_captura, preprocesado)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    placa_clean,
                    momento_str,
                    sentido_final,
                    float(confianza),
                    ruta_captura,
                    1 if preprocesado else 0,
                ),
            )
            acceso_id = cursor.lastrowid

        # Cálculo de permanencia si corresponde a SALIDA
        permanencia_minutos: float | None = None
        permanencia_texto: str | None = None

        if sentido_final == "SALIDA":
            # Buscar la última entrada previa a este momento
            with self.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT momento FROM accesos
                    WHERE placa = ? AND sentido = 'ENTRADA' AND momento <= ? AND id != ?
                    ORDER BY momento DESC, id DESC
                    LIMIT 1;
                    """,
                    (placa_clean, momento_str, acceso_id),
                )
                row_entrada = cursor.fetchone()

            if row_entrada:
                entrada_str = row_entrada["momento"]
                try:
                    dt_entrada = datetime.fromisoformat(entrada_str)
                    dt_salida = datetime.fromisoformat(momento_str)
                    segundos = max(0.0, (dt_salida - dt_entrada).total_seconds())
                    permanencia_minutos = round(segundos / 60.0, 2)

                    horas = int(permanencia_minutos // 60)
                    minutos = int(permanencia_minutos % 60)
                    if horas > 0:
                        permanencia_texto = f"{horas}h {minutos}m"
                    else:
                        permanencia_texto = f"{minutos}m"
                except ValueError:
                    permanencia_minutos = None
                    permanencia_texto = None

        return {
            "id": acceso_id,
            "placa": placa_clean,
            "tipo": tipo,
            "momento": momento_str,
            "sentido": sentido_final,
            "confianza": float(confianza),
            "ruta_captura": ruta_captura,
            "preprocesado": preprocesado,
            "permanencia_minutos": permanencia_minutos,
            "permanencia_texto": permanencia_texto,
        }

    # -------------------------------------------------------------------------
    # CONSULTAS AGREGADAS Y REPORTES DE NEGOCIO
    # -------------------------------------------------------------------------

    def get_vehiculos_en_parqueadero(self) -> list[dict[str, Any]]:
        """Devuelve los vehículos que se encuentran actualmente dentro (último estado = ENTRADA)."""
        query = """
        WITH UltimosAccesos AS (
            SELECT
                a.id,
                a.placa,
                a.momento,
                a.sentido,
                a.confianza,
                a.ruta_captura,
                v.tipo,
                ROW_NUMBER() OVER (PARTITION BY a.placa ORDER BY a.momento DESC, a.id DESC) as rn
            FROM accesos a
            JOIN vehiculos v ON a.placa = v.placa
        )
        SELECT id, placa, tipo, momento as entrada_momento, confianza, ruta_captura
        FROM UltimosAccesos
        WHERE rn = 1 AND sentido = 'ENTRADA'
        ORDER BY momento DESC;
        """
        now = datetime.now()
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            filas = [dict(r) for r in cursor.fetchall()]

        for item in filas:
            try:
                dt_in = datetime.fromisoformat(item["entrada_momento"])
                mins = max(0.0, round((now - dt_in).total_seconds() / 60.0, 1))
                item["minutos_dentro"] = mins
            except ValueError:
                item["minutos_dentro"] = 0.0

        return filas

    def get_historial_accesos(
        self, limite: int = 50, placa: str | None = None
    ) -> list[dict[str, Any]]:
        """Consulta el historial cronológico de accesos con datos del vehículo."""
        where_clause = ""
        params: list[Any] = []
        if placa:
            where_clause = "WHERE a.placa = ?"
            params.append(placa.strip().upper())

        query = f"""
        SELECT
            a.id,
            a.placa,
            v.tipo,
            a.momento,
            a.sentido,
            a.confianza,
            a.ruta_captura,
            a.preprocesado
        FROM accesos a
        JOIN vehiculos v ON a.placa = v.placa
        {where_clause}
        ORDER BY a.momento DESC, a.id DESC
        LIMIT ?;
        """
        params.append(limite)
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]

    def get_flujo_por_hora(self, fecha: str | None = None) -> list[dict[str, Any]]:
        """Calcula el volumen de entradas y salidas agrupado por hora para una fecha (YYYY-MM-DD)."""
        fecha_filtro = fecha or datetime.now().strftime("%Y-%m-%d")
        query = """
        SELECT
            strftime('%H', momento) as hora,
            SUM(CASE WHEN sentido = 'ENTRADA' THEN 1 ELSE 0 END) as entradas,
            SUM(CASE WHEN sentido = 'SALIDA' THEN 1 ELSE 0 END) as salidas,
            COUNT(*) as total_movimientos
        FROM accesos
        WHERE date(momento) = date(?)
        GROUP BY strftime('%H', momento)
        ORDER BY hora ASC;
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (fecha_filtro,))
            return [dict(r) for r in cursor.fetchall()]

    def get_distribucion_vehicular(self) -> dict[str, int]:
        """Devuelve el total de vehículos registrados clasificados por tipo (CARRO vs MOTO)."""
        query = """
        SELECT tipo, COUNT(*) as cantidad
        FROM vehiculos
        GROUP BY tipo;
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            resultado = {"CARRO": 0, "MOTO": 0}
            for row in cursor.fetchall():
                resultado[row["tipo"]] = row["cantidad"]
            return resultado

    def get_metricas_dashboard(self) -> dict[str, Any]:
        """Consolida KPIs esenciales para la consola SCADA de misión crítica."""
        activos = self.get_vehiculos_en_parqueadero()
        carros_dentro = sum(1 for v in activos if v["tipo"] == "CARRO")
        motos_dentro = sum(1 for v in activos if v["tipo"] == "MOTO")

        hoy_str = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            # Total accesos hoy
            cursor = conn.execute(
                "SELECT COUNT(*) FROM accesos WHERE date(momento) = date(?);",
                (hoy_str,),
            )
            total_hoy = cursor.fetchone()[0]

            # Confianza promedio general
            cursor = conn.execute("SELECT AVG(confianza) FROM accesos;")
            avg_conf = cursor.fetchone()[0] or 0.0

        return {
            "ocupacion_total": len(activos),
            "carros_dentro": carros_dentro,
            "motos_dentro": motos_dentro,
            "total_accesos_hoy": total_hoy,
            "confianza_promedio": round(float(avg_conf), 4),
        }
