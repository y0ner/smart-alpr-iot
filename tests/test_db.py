"""Pruebas unitarias completas de la capa de persistencia SQLite y lógica de permanencia.

Cubre inicialización de esquema, integridad referencial (FK), alternancia inteligente
de sentido, cálculo exacto de tiempo de estadía y consultas analíticas agregadas.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from src.config import SCHEMA_SQL_PATH
from src.db import DatabaseManager


@pytest.fixture
def db() -> DatabaseManager:
    """Fixture que entrega una base de datos SQLite en memoria con esquema cargado."""
    manager = DatabaseManager(db_path=Path(":memory:"))
    manager.init_db(schema_path=SCHEMA_SQL_PATH)
    return manager


def test_schema_init_and_foreign_key_enforcement(db: DatabaseManager) -> None:
    """Verifica que el esquema se inicialice y que las claves foráneas bloqueen placas huérfanas."""
    status = db.ping()
    assert status["status"] == "connected"

    # Intentar insertar directamente un acceso sin registrar previamente el vehículo
    # debe fallar con sqlite3.IntegrityError debido a PRAGMA foreign_keys = ON
    with pytest.raises(sqlite3.IntegrityError), db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO accesos (placa, sentido, confianza)
            VALUES ('HUR999', 'ENTRADA', 0.95);
            """
        )


def test_schema_file_not_found() -> None:
    """Verifica que init_db lance FileNotFoundError si no existe el schema.sql."""
    mgr = DatabaseManager(db_path=Path(":memory:"))
    with pytest.raises(FileNotFoundError):
        mgr.init_db(schema_path=Path("/ruta/inexistente/schema.sql"))


def test_upsert_vehiculo(db: DatabaseManager) -> None:
    """Verifica la inserción y actualización (idempotencia) de vehículos."""
    assert db.get_vehiculo("NOEXISTE") is None

    db.upsert_vehiculo("ABC123", "CARRO")
    v1 = db.get_vehiculo("ABC123")
    assert v1 is not None
    assert v1["placa"] == "ABC123"
    assert v1["tipo"] == "CARRO"

    # Actualizar tipo si se reclasifica
    db.upsert_vehiculo("ABC123", "MOTO")
    v2 = db.get_vehiculo("ABC123")
    assert v2 is not None
    assert v2["tipo"] == "MOTO"


def test_registrar_acceso_alternancia_inteligente(db: DatabaseManager) -> None:
    """Verifica que sin sentido explícito, el sistema alterne ENTRADA -> SALIDA -> ENTRADA."""
    assert db.get_ultimo_acceso("COL123") is None

    # Primer evento sin registrar -> Debe ser ENTRADA
    r1 = db.registrar_acceso("COL123", "CARRO", confianza=0.94)
    assert r1["sentido"] == "ENTRADA"
    assert r1["permanencia_minutos"] is None

    # Segundo evento consecutivo -> Debe alternar automáticamente a SALIDA
    r2 = db.registrar_acceso("COL123", "CARRO", confianza=0.91)
    assert r2["sentido"] == "SALIDA"
    assert r2["permanencia_minutos"] is not None

    # Tercer evento -> Debe volver a alternar a ENTRADA
    r3 = db.registrar_acceso("COL123", "CARRO", confianza=0.96)
    assert r3["sentido"] == "ENTRADA"
    assert r3["permanencia_minutos"] is None


def test_calculo_exacto_tiempo_permanencia(db: DatabaseManager) -> None:
    """Verifica el cálculo matemático exacto de permanencia entre timestamps definidos."""
    momento_in = "2026-11-09 08:00:00"
    momento_out = "2026-11-09 10:15:30"  # 2 horas, 15 minutos y 30 segundos = 135.5 minutos

    # Registro de ENTRADA
    res_in = db.registrar_acceso(
        placa="VIV100",
        tipo="CARRO",
        confianza=0.98,
        sentido="ENTRADA",
        momento=momento_in,
    )
    assert res_in["sentido"] == "ENTRADA"
    assert res_in["permanencia_minutos"] is None

    # Registro de SALIDA
    res_out = db.registrar_acceso(
        placa="VIV100",
        tipo="CARRO",
        confianza=0.95,
        sentido="SALIDA",
        momento=momento_out,
    )
    assert res_out["sentido"] == "SALIDA"
    # 2 horas = 120 mins + 15 mins + 0.5 mins = 135.5 mins
    assert res_out["permanencia_minutos"] == 135.5
    assert res_out["permanencia_texto"] == "2h 15m"


def test_permanencia_corta_minutos_unicamente(db: DatabaseManager) -> None:
    """Verifica que estadías menores a una hora solo muestren minutos."""
    db.registrar_acceso("MIN001", "CARRO", 0.9, sentido="ENTRADA", momento="2026-11-09 08:00:00")
    res = db.registrar_acceso("MIN001", "CARRO", 0.9, sentido="SALIDA", momento="2026-11-09 08:25:00")
    assert res["permanencia_minutos"] == 25.0
    assert res["permanencia_texto"] == "25m"


def test_vehiculos_en_parqueadero_y_ocupacion(db: DatabaseManager) -> None:
    """Verifica el listado de vehículos activos dentro del parqueadero y KPIs."""
    t_now = datetime.now()
    # Ingreso de un Carro y una Moto
    db.registrar_acceso(
        "CAR001",
        "CARRO",
        confianza=0.92,
        sentido="ENTRADA",
        momento=t_now,
    )
    db.registrar_acceso(
        "MOT001",
        "MOTO",
        confianza=0.89,
        sentido="ENTRADA",
        momento=t_now,
    )

    dentro = db.get_vehiculos_en_parqueadero()
    assert len(dentro) == 2
    placas_dentro = [v["placa"] for v in dentro]
    assert "CAR001" in placas_dentro
    assert "MOT001" in placas_dentro

    # Dar salida al Carro
    db.registrar_acceso(
        "CAR001",
        "CARRO",
        confianza=0.95,
        sentido="SALIDA",
        momento=t_now,
    )

    dentro_post = db.get_vehiculos_en_parqueadero()
    assert len(dentro_post) == 1
    assert dentro_post[0]["placa"] == "MOT001"
    assert dentro_post[0]["tipo"] == "MOTO"

    metricas = db.get_metricas_dashboard()
    assert metricas["ocupacion_total"] == 1
    assert metricas["carros_dentro"] == 0
    assert metricas["motos_dentro"] == 1
    assert metricas["total_accesos_hoy"] >= 3


def test_historial_accesos_y_filtro_por_placa(db: DatabaseManager) -> None:
    """Verifica que el historial ordene cronológicamente y filtre por placa."""
    db.registrar_acceso("AAA111", "CARRO", confianza=0.90, momento="2026-11-09 09:00:00")
    db.registrar_acceso("BBB222", "MOTO", confianza=0.88, momento="2026-11-09 09:05:00")
    db.registrar_acceso("AAA111", "CARRO", confianza=0.92, momento="2026-11-09 09:30:00")

    todos = db.get_historial_accesos(limite=10)
    assert len(todos) == 3

    solo_aaa = db.get_historial_accesos(placa="AAA111")
    assert len(solo_aaa) == 2
    for r in solo_aaa:
        assert r["placa"] == "AAA111"


def test_flujo_por_hora_y_distribucion(db: DatabaseManager) -> None:
    """Verifica la agregación de flujo por hora y conteo total por tipo."""
    fecha_test = "2026-11-09"
    # 2 entradas en la hora 08
    db.registrar_acceso("A10101", "CARRO", confianza=0.9, sentido="ENTRADA", momento=f"{fecha_test} 08:15:00")
    db.registrar_acceso("B20202", "MOTO", confianza=0.9, sentido="ENTRADA", momento=f"{fecha_test} 08:45:00")
    # 1 salida en la hora 09
    db.registrar_acceso("A10101", "CARRO", confianza=0.9, sentido="SALIDA", momento=f"{fecha_test} 09:30:00")

    flujo = db.get_flujo_por_hora(fecha=fecha_test)
    assert len(flujo) == 2  # horas 08 y 09

    h08 = next(item for item in flujo if item["hora"] == "08")
    assert h08["entradas"] == 2
    assert h08["salidas"] == 0

    h09 = next(item for item in flujo if item["hora"] == "09")
    assert h09["entradas"] == 0
    assert h09["salidas"] == 1

    dist = db.get_distribucion_vehicular()
    assert dist["CARRO"] == 1
    assert dist["MOTO"] == 1


def test_transaction_rollback_on_error(db: DatabaseManager) -> None:
    """Verifica que una excepción dentro del bloque con get_connection() ejecute rollback."""
    with pytest.raises(RuntimeError), db.get_connection() as conn:
        conn.execute("INSERT INTO vehiculos (placa, tipo) VALUES ('ROLL01', 'CARRO');")
        raise RuntimeError("Error forzado de transacción")

    assert db.get_vehiculo("ROLL01") is None


def test_close_database_manager(db: DatabaseManager) -> None:
    """Verifica que el método close cierre la conexión compartida sin errores."""
    db.close()
    assert db._shared_conn is None
