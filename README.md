# Smart ALPR IoT 🚗🔍📡

> **Sistema Inteligente de Lectura de Placas Vehiculares con Visión Artificial e IoT**  
> *Proyecto Final — Visión Artificial | Universidad de La Guajira*

---

## 📌 Descripción

**Smart ALPR IoT** es una solución integral de **Reconocimiento Automático de Placas Vehiculares (ALPR / ANPR)** diseñada para el control automatizado de acceso en parqueaderos, conjuntos residenciales o centros comerciales (caso de estudio: *Centro Comercial Viva Wajiira / Campus Universidad de La Guajira*, Riohacha). 

El sistema integra captura de video/cámara IoT, detección de regiones de interés mediante modelos de Deep Learning (YOLO), técnicas clásicas de procesamiento digital de imágenes para maximizar contraste y nitidez, reconocimiento óptico de caracteres (OCR) con validación sintáctica adaptada a la normativa vehicular colombiana, y persistencia en base de datos SQLite con auditoría de entradas, salidas y cálculo de tiempo de permanencia en tiempo real.

---

## 🏗️ Arquitectura del Pipeline

El flujo de procesamiento sigue una secuencia modular optimizada:

```
[ Cámara / Dispositivo IoT ]
             │ (Stream de video / RTSP / Imagen)
             ▼
[ 1. Detección de Placa (YOLO) ]
             │ Bounding Box + Margen porcentual (12%) + Escalado 3x
             ▼
[ 2. Preprocesamiento Morfológico (OpenCV) ]
             │ • Escala de grises
             │ • Filtro Bilateral (reducción de ruido preservando bordes)
             │ • Umbralización adaptativa / Otsu
             │ • Operaciones morfológicas (apertura y cierre de caracteres)
             ▼
[ 3. Extracción OCR (EasyOCR) ]
             │ Allowlist: A-Z, 0-9
             ▼
[ 4. Validación Sintáctica Colombiana & Corrección Heurística ]
             │ • Carros: AAA-123 ([A-Z]{3}[0-9]{3})
             │ • Motos:  AAA-12D ([A-Z]{3}[0-9]{2}[A-Z])
             │ • Corrección posicional de ambigüedades (O/0, I/1, Z/2, S/5, B/8)
             ▼
[ 5. Filtro de Votación Temporal (PlateVoterBuffer) ]
             │ Ventana deslizante (N=7, k=4) y cooldown para descarte de duplicados
             ▼
[ 6. Base de Datos SQLite & Control de Acceso ]
             └─ Registro transaccional, cálculo de permanencia y telemetría
```

---

## 📂 Estructura del Repositorio

```text
smart-alpr-iot/
├── .agents/                    # Configuración de agentes, skills y reglas de diseño
│   ├── rules/
│   │   └── industrial-ui-standard.md
│   └── skills/
│       ├── frontend-design/
│       └── ui-ux-pro-max/
├── data/                       # Datasets locales organizados
│   ├── processed/
│   ├── raw/
│   └── samples/
├── db/                         # Persistencia relacional
│   └── schema.sql
├── docs/                       # Documentación técnica y académica
│   ├── analisis_tecnico_A_E.md
│   ├── ficha_cliente.md
│   ├── Guia_paso_a_paso_proyectos.pdf
│   └── Proyecto_final_Vision_Artificial.pdf
├── models/                     # Pesos de modelos de detección (*.pt, *.onnx)
├── notebooks/                  # Notebooks de experimentación y entrenamiento Colab
├── src/                        # Código fuente modular tipado estrictamente
│   ├── __init__.py
│   ├── app.py                 # Consola SCADA / Telemetría vial (Streamlit)
│   ├── config.py              # Hiperparámetros y rutas absolutas
│   ├── db.py                  # Repositorio SQLite y cálculo de permanencia
│   ├── detector.py            # Orquestador YOLO + OCR
│   ├── preprocessor.py        # Pipeline de visión clásica OpenCV
│   └── validator.py           # Sintaxis de placas colombianas y votación temporal
├── tests/                      # Suite de pruebas unitarias automatizadas (pytest)
│   ├── __init__.py
│   └── test_scaffolding.py
├── requirements.txt            # Dependencias del proyecto
└── README.md
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio y navegar a la rama de desarrollo
```bash
git clone https://github.com/y0ner/smart-alpr-iot.git
cd smart-alpr-iot
git checkout dev
```

### 2. Configurar el entorno virtual en Linux Mint
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Ejecutar pruebas unitarias y análisis estático
```bash
pytest -v
ruff check src tests
```

### 4. Lanzar la consola de telemetría industrial
```bash
streamlit run src/app.py
```

---

## 📋 Plan de Desarrollo por Etapas (Roadmap)

- [x] **Etapa 0:** Skills de diseño industrial (`ui-ux-pro-max`, `frontend-design`), scaffolding oficial, entorno Python `.venv` y linters.
- [ ] **Etapa 1:** Ficha de cliente empresarial (Riohacha) y capa relacional SQLite (`schema.sql` y `src/db.py`).
- [ ] **Etapa 2:** Validación de placa colombiana (Carro/Moto), corrección heurística y votador multi-cuadro.
- [ ] **Etapa 3:** Pipeline de visión clásica OpenCV, extracción morfológica y demostración matemática 3x3 paso a paso.
- [ ] **Etapa 4:** Motor de inferencia YOLO + EasyOCR, latencias, benchmark A/B (reto extra) y notebook Colab.
- [ ] **Etapa 5:** Interfaz de control SCADA/Telemetría en tiempo real (`src/app.py`) y dockerización.
- [ ] **Etapa 6:** Documentación técnica completa (Secciones A a E), auditoría de fallas y entrega final.
