# =============================================================================
# SMART ALPR IOT - DOCKERFILE INDUSTRIAL (DESPLIEGUE FUERA DE COLAB +0.5 PTS)
# =============================================================================

FROM python:3.12-slim

# Metadatos del Contenedor
LABEL maintainer="Facultad de Ingeniería - Universidad de La Guajira"
LABEL description="Consola de Telemetría Industrial Smart ALPR IoT"

# Evitar prompts interactivos durante apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema requeridas por OpenCV y PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiar requirements e instalar dependencias de Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y artefactos del proyecto
COPY . /app/

# Crear directorios para persistencia de datos y modelos
RUN mkdir -p /app/data/processed /app/data/raw /app/data/samples /app/models /app/db

# Exponer el puerto de Streamlit
EXPOSE 8501

# Healthcheck para verificar disponibilidad del servicio
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de ejecución de la consola de telemetría industrial
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
