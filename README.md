# Smart ALPR IoT 🚗🔍📡

> **Sistema Inteligente de Lectura de Placas Vehiculares con Visión Artificial e IoT**  
> *Proyecto Final — Visión Artificial | Universidad de La Guajira*

---

## 📌 Descripción

**Smart ALPR IoT** es una solución integral de **Reconocimiento Automático de Placas Vehiculares (ALPR / ANPR)** diseñada para el control automatizado de acceso en parqueaderos, conjuntos residenciales o centros comerciales. 

El sistema integra captura de video/cámara IoT, detección de regiones de interés mediante modelos de Deep Learning (YOLO), técnicas clásicas de procesamiento digital de imágenes para maximizar contraste y nitidez, reconocimiento óptico de caracteres (OCR) con validación sintáctica adaptada a la normativa vehicular colombiana, y persistencia en base de datos de auditoría de entradas y salidas.

---

## 🏗️ Arquitectura del Pipeline

El flujo de procesamiento sigue una secuencia modular optimizada:

```
[ Cámara / Dispositivo IoT ]
             │ (Stream de video / RTSP)
             ▼
[ 1. Detección de Placa (YOLO) ]
             │ Bounding Box + Margen de seguridad
             ▼
[ 2. Preprocesamiento Morfológico ]
             │ • Escala de grises
             │ • Bilateral Filter (reducción de ruido preservando bordes)
             │ • Umbralización adaptativa / Otsu
             │ • Operaciones morfológicas (apertura/cierre)
             ▼
[ 3. Extracción OCR (EasyOCR) ]
             │ Allowlist: A-Z, 0-9
             ▼
[ 4. Validación Sintáctica (Formato Colombia) ]
             │ • Carros: AAA-123 ([A-Z]{3}[0-9]{3})
             │ • Motos:  AAA-12D ([A-Z]{3}[0-9]{2}[A-Z])
             │ • Corrección de ambigüedades (O/0, I/1)
             ▼
[ 5. Filtro de Votación Temporal ]
             │ Consenso de lecturas en ráfaga de cuadros
             ▼
[ 6. Base de Datos & Control de Acceso ]
             └─ Registro de eventos (Placa, Timestamp, Captura, Permanencia)
```

---

## 📂 Estructura del Repositorio

```text
smart-alpr-iot/
├── data/                  # Datasets locales (raw, etiquetado y splits)
├── docs/                  # Guías del curso y especificaciones del proyecto
│   ├── Guia_paso_a_paso_proyectos.pdf
│   └── Proyecto_final_Vision_Artificial.pdf
├── models/                # Pesos de modelos entrenados (*.pt, *.onnx)
├── src/                   # Código fuente modular
│   ├── __init__.py
│   ├── capture.py         # Interfaz de captura y streaming IoT / Cámara
│   ├── detector.py        # Módulo de detección de placa con YOLO
│   ├── preprocessor.py    # Filtros digitales, umbralización y morfología
│   ├── ocr_engine.py      # Extracción y corrección sintáctica de texto
│   ├── voter.py           # Votación inter-cuadros para estabilización
│   ├── database.py        # Capa de datos (SQLite / PostgreSQL)
│   └── main.py            # Orquestador del sistema en tiempo real
├── requirements.txt       # Dependencias de Python
└── README.md
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone https://github.com/y0ner/smart-alpr-iot.git
cd smart-alpr-iot
```

### 2. Crear entorno virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 📋 Requerimientos y Entregables

- [x] Inicialización del repositorio y estructura modular.
- [ ] Etiquetado y entrenamiento del modelo YOLO para placas colombianas.
- [ ] Implementación de canal de preprocesamiento (comparativa con vs sin preprocesamiento).
- [ ] Integración y ajuste de EasyOCR con restricciones de caracteres.
- [ ] Motor de validación de placas colombianas (motos y vehículos particulares/públicos).
- [ ] Votador temporal inter-frames.
- [ ] Base de datos de entradas y salidas con cálculo de tiempos de estancia.
- [ ] Pruebas en condiciones reales de iluminación y reporte técnico.

---

## 👨‍💻 Autor
- **Yoner Isaza** ([@y0ner](https://github.com/y0ner))
