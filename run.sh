#!/usr/bin/env bash
# =============================================================================
# SMART ALPR IOT - SCRIPT DE LANZAMIENTO LOCAL (LINUX MINT / UBUNTU)
# Universidad de La Guajira - Visión Artificial
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "  INICIANDO CONSOLA INDUSTRIAL SMART ALPR IOT"
echo "  Entorno: Linux Mint | Puerto: 8501"
echo "=========================================================="

# 1. Verificar o crear entorno virtual .venv
if [ ! -d ".venv" ]; then
    echo "[!] Creando entorno virtual .venv..."
    python3 -m venv .venv
    echo "[!] Instalando dependencias desde requirements.txt..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# 2. Asegurar directorios de persistencia
mkdir -p data/raw data/processed data/samples db models

# 3. Lanzar aplicación Streamlit con configuración industrial
echo "[✓] Lanzando consola Streamlit en http://localhost:8501 ..."
exec .venv/bin/streamlit run src/app.py \
    --server.port=8501 \
    --server.headless=false \
    --theme.base="dark" \
    --theme.primaryColor="#10B981" \
    --theme.backgroundColor="#090A0F" \
    --theme.secondaryBackgroundColor="#11131A" \
    --theme.textColor="#F3F4F6"
