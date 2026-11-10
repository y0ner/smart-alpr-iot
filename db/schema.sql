-- =============================================================================
-- SMART ALPR IOT - ESQUEMA DE BASE DE DATOS RELACIONAL (SQLite)
-- Proyecto #4: Sistema Inteligente Lector de Placas Vehiculares
-- Universidad de La Guajira - Ingeniería de Sistemas
-- =============================================================================

-- Tabla de vehículos registrados / detectados
CREATE TABLE IF NOT EXISTS vehiculos (
    placa TEXT PRIMARY KEY,
    tipo TEXT NOT NULL CHECK(tipo IN ('CARRO', 'MOTO')),
    creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla transaccional de accesos (entradas y salidas)
CREATE TABLE IF NOT EXISTS accesos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placa TEXT NOT NULL REFERENCES vehiculos(placa),
    momento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentido TEXT NOT NULL CHECK(sentido IN ('ENTRADA', 'SALIDA')),
    confianza REAL NOT NULL,
    ruta_captura TEXT,
    preprocesado BOOLEAN DEFAULT 1
);

-- Índices para optimización de consultas de auditoría y permanencia
CREATE INDEX IF NOT EXISTS idx_accesos_placa ON accesos(placa);
CREATE INDEX IF NOT EXISTS idx_accesos_momento ON accesos(momento);
CREATE INDEX IF NOT EXISTS idx_accesos_sentido ON accesos(sentido);
