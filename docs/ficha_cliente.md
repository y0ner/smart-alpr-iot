# FICHA TÉCNICA Y DE NEGOCIO DEL CLIENTE (BUSINESS CONTEXT)
## Sistema Inteligente Lector de Placas Vehiculares (Smart ALPR IoT)
*Proyecto Final — Visión Artificial | Facultad de Ingeniería — Universidad de La Guajira*

---

### 1. Perfil de la Organización y Entorno Operativo

- **Razón Social / Organización:** Centro Comercial Viva Wajiira / Consorcio Parqueaderos del Norte S.A.S.  
  *(En coexistencia con el piloto de acceso institucional en la Universidad de La Guajira - Campus Riohacha).*
- **Ubicación Geográfica:** Calle 15 con Carrera 11, Riohacha, Departamento de La Guajira, Colombia.
- **Sector de Actividad:** Infraestructura Comercial, Gestión de Estacionamientos y Seguridad Vial Perimetral.
- **Capacidad Instalada:** 420 bahías para automóviles y 180 bahías para motocicletas.
- **Flujo Vehicular Promedio:** 1,850 ingresos diarios entre semana; hasta 3,200 ingresos/día los fines de semana.
- **Usuarios y Operadores del Sistema:**
  1. *Operador de Garita / Talanquera:* Supervisa el flujo en pantalla y gestiona excepciones o vehículos no identificados.
  2. *Auditor de Seguridad y Control Interno:* Revisa discrepancias de tiempo de permanencia, vehículos con pernocta no autorizada y registros fotográficos.
  3. *Administrador General de Operaciones:* Analiza reportes consolidados de afluencia, horas pico, distribución vehicular (Carros vs. Motos) y facturación por estancia.

---

### 2. Diagnóstico del Problema de Negocio Cuantificado

Previo a la concepción del sistema **Smart ALPR IoT**, el proceso de control de acceso se realizaba mediante expedición manual de boletos de papel y digitación por teclado por parte de personal de seguridad en garita:

1. **Tiempos de Espera y Congestión en Accesos (Cuellos de Botella):**
   - El tiempo de registro promedio por vehículo (saludo, digitación manual de 6 caracteres, impresión de ticket y apertura de barrera) oscilaba entre **32 y 45 segundos**.
   - En horas pico (11:45 AM – 1:30 PM y 5:30 PM – 7:30 PM), las filas vehiculares bloqueaban la Calle 15 de Riohacha, generando multas por afectación vial y quejas sistemáticas de los clientes.

2. **Tasa Crítica de Error Humano en Digitación:**
   - Auditorías internas revelaron un **14.8% de errores de transcripción** en planillas y sistemas manuales, provocados principalmente por fatiga visual y confusión de caracteres de morfología similar:
     - `O` (letra) confundida con `0` (número).
     - `I` (letra) confundida con `1` (número).
     - `B` (letra) confundida con `8` (número).
     - `Z` (letra) confundida con `2` (número).
     - `G` (letra) confundida con `6` (número).

3. **Pérdida de Trazabilidad y Control del Tiempo de Permanencia:**
   - La pérdida frecuente del ticket de papel por parte del usuario impedía determinar con certeza la hora exacta de ingreso, derivando en cobros arbitrarios o pérdidas económicas para el establecimiento al no poder calcular con exactitud la permanencia (minutos y horas de estancia).

4. **Desafíos Ambientales Propios de La Guajira:**
   - Fuerte radiación solar cenital (>1,000 W/m²) que produce deslumbramiento especular en las láminas reflectivas amarillas y blancas de las placas colombianas.
   - Polvo en suspensión y cambios de iluminación entre sombra de garita y luz solar directa.

---

### 3. Indicadores Clave de Éxito (KPIs Cuantitativos)

El éxito del proyecto y la viabilidad técnica para producción se miden frente a las siguientes metas auditables:

| Indicador (KPI) | Situación Previa (Manual) | Meta Smart ALPR IoT | Método de Medición |
| :--- | :---: | :---: | :--- |
| **Tiempo de Registro por Vehículo** | 35 – 45 s | **< 2.5 s** | Cronometrado desde ingreso al campo de visión hasta persistencia en BD |
| **Exactitud de Lectura de Placa** | 85.2% (14.8% error) | **≥ 92.0%** | Matriz de confusión sobre dataset de prueba real en Riohacha |
| **Cálculo Automático de Permanencia** | 0% (requiere ticket) | **100% automatizado** | Cálculo en SQLite entre timestamp de ENTRADA y SALIDA |
| **Disponibilidad Operativa Local** | Dependiente de humano | **≥ 99.5%** | Arquitectura local SQLite + inferencia edge sin caída por internet |
| **Tasa de Rechazo de Ruido / Duplicados** | N/A (duplicados manuales) | **≥ 98.0%** | Buffer de votación temporal ($N=7$, $k=4$) y ventana de enfriamiento |

---

### 4. Especificación de Requerimientos del Sistema

#### Requerimientos Funcionales (RF)
- **RF-01 (Detección de Región de Interés):** Localizar con un modelo Deep Learning (YOLO) la placa delantera o trasera en imágenes y stream de video con una confianza mínima del 45%.
- **RF-02 (Acondicionamiento Óptico Digital):** Recortar la matrícula con un margen de seguridad porcentual (12%) y aplicar un pipeline OpenCV: conversión a grises, filtrado bilateral (preservación de aristas y atenuación de ruido), binarización adaptativa/Otsu y morfología rectangular.
- **RF-03 (Extracción OCR Restringida):** Extraer caracteres exclusivamente del conjunto `[A-Z0-9]` mediante EasyOCR, evitando caracteres de control o símbolos especiales.
- **RF-04 (Validación Sintáctica Colombiana):** Validar contra el formato vehicular nacional:
  - Automóviles y Camionetas: 3 letras y 3 números (`^[A-Z]{3}[0-9]{3}$`).
  - Motocicletas: 3 letras, 2 números y 1 letra (`^[A-Z]{3}[0-9]{2}[A-Z]$`).
  - Corrección posicional de ambigüedades morfológicas antes de rechazar.
- **RF-05 (Consenso Temporal Multi-Cuadro):** Implementar un buffer temporal (`PlateVoterBuffer`) con ventana deslizante ($N=7$ frames) que emita un resultado validado solo cuando una placa alcance al menos $k=4$ votos consistentes, bloqueando duplicados durante una ventana de enfriamiento de 15 segundos.
- **RF-06 (Persistencia Relacional SQLite):** Registrar cada evento en SQLite con integridad referencial, asociando la placa al catálogo de vehículos y calculando el tiempo de permanencia si el evento corresponde a una salida.
- **RF-07 (Consola Industrial SCADA):** Proveer un tablero de control sin estéticas genéricas de IA, con visualización de stream, telemetría de latencias, inspector morfológico en 5 etapas y tabla transaccional de auditoría.

#### Requerimientos No Funcionales (RNF)
- **RNF-01 (Latencia):** Tiempo total de inferencia de pipeline (YOLO + OpenCV + OCR + BD) inferior a 250 ms por cuadro en hardware con aceleración GPU o inferior a 600 ms en CPU.
- **RNF-02 (Robustez y Persistencia):** Base de datos relacional transaccional (ACID) con claves foráneas activas (`PRAGMA foreign_keys = ON;`) y almacenamiento persistente en disco.
- **RNF-03 (Mantenibilidad y Código Limpio):** Tipado estricto en Python (`typing`), cumplimiento del linter `ruff` sin excepciones y cobertura de pruebas unitarias (`pytest`) superior al 90%.

---

### 5. Parámetros de Instalación Física y Hardware Recomendado

- **Cámara:** Cámara IP PoE con resolución mínima Full HD (1920×1080) a 30 FPS, lente varifocal (5–50 mm) y obturador rápido (1/1000s) para evitar desenfoque por movimiento (*motion blur*).
- **Geometría de Captura:**
  - Altura de montaje: entre 1.80 m y 2.50 m sobre el nivel del suelo.
  - Distancia al vehículo: entre 3.5 m y 6.0 m.
  - Ángulo de incidencia horizontal y vertical: inferior a 25° respecto al eje longitudinal del carril.
- **Iluminación:** Foco LED auxiliar de luz blanca o iluminación infrarroja (850 nm) con filtro IR-Cut sincronizado para mitigar el contraste extremo de sol de Riohacha y garantizar operatividad 24/7.
- **Hardware de Procesamiento:** Estación de trabajo local o SBC industrial (x86_64 Linux Mint / Ubuntu), 16 GB RAM, GPU NVIDIA RTX serie 3000/4000 o procesador multi-núcleo Intel Core i7/AMD Ryzen 7 para ejecución autónoma sin dependencia de la nube.
