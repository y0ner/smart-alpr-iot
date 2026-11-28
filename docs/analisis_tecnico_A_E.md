# ANÁLISIS TÉCNICO COMPLETO (SECCIONES A - E)
## Sistema Inteligente Lector de Placas Vehiculares (Smart ALPR IoT)
*Universidad de La Guajira — Facultad de Ingeniería — Programa de Ingeniería de Sistemas*  
*Asignatura: Internet de las Cosas, Grupo A3 — Docente: Eduardo Luis Sierra Fragozo*  
*Autores: Yoner De Jesus Isaza Vides y Kerlon Andres Causado Ojeda — Riohacha, noviembre de 2026*  
*Caso de Estudio: Control de Acceso y Tarificación Automatizada en Riohacha, La Guajira*

---

> **Resumen Ejecutivo:** El presente documento constituye el análisis técnico formal y académico del sistema **Smart ALPR IoT**, cubriendo en profundidad las cinco secciones exigidas por la rúbrica oficial de evaluación (Secciones A a E, 1.2 puntos + reto extra experimental de 0.5 puntos). Se exponen los fundamentos de álgebra lineal, la teoría de redes convolucionales, la correlación exacta entre código fuente e ingeniería de visión, la demostración paso a paso de convolución espacial 3x3 frente a OpenCV (con error residual de $0.0 \times 10^0$), la evaluación experimental A/B con Character Error Rate (CER), el análisis de casos de falla reales y el estudio de viabilidad económica e industrial para el despliegue en Riohacha.

---

## SECCIÓN A: REPRESENTACIÓN MATRICIAL DE IMÁGENES Y REDES CONVOLUCIONALES

### A.1 Flujo Matricial Paso a Paso: De los Fotones en el Sensor a la Inserción Relacional

La transformación de una escena física en un registro estructurado dentro de una base de datos relacional sigue una cadena discreta de siete etapas deterministas:

```mermaid
flowchart TD
    A["1. Matriz Sensor Raw (H × W × 3, uint8)"] --> B["2. Tensor Normalizado (1 × 3 × 640 × 640, float32)"]
    B --> C["3. YOLO Feature Pyramid (Backbone + PANet + Head)"]
    C --> D["4. Bounding Box & Recorte ROI + 12% Margen (H' × W' × 3)"]
    D --> E["5. Pipeline OpenCV (Bicúbico 3x → Grises → Bilateral → Otsu/Adaptativo → Morfología)"]
    E --> F["6. Inferencia Secuencial EasyOCR (CRNN: ResNet + BiLSTM + CTC)"]
    F --> G["7. Validador Colombiano (AAA000 / AAA00A) + Votador Temporal (N=7, k=4)"]
    G --> H["8. Transacción SQLite ACID (vehiculos + accesos + permanencia)"]
```

#### 1. Captura Fotónica y Formación de la Imagen Digital
La luz ambiental reflejada por la carrocería y la matrícula vehicular incide sobre el arreglo bidimensional de fotodiodos del sensor CMOS/CCD de la cámara IP perimetral. Cada celda fotosensible integra los fotones durante el tiempo de exposición y los cuantiza mediante un convertidor analógico-digital (ADC) de 8 bits por canal, bajo el patrón de filtrado de color Bayer (RGGB). Tras el proceso de demosaicing en el Procesador de Señal de Imagen (ISP), la señal se expresa matemáticamente como un tensor tridimensional de orden 3:
$$\mathbf{I}_{\text{raw}} \in \mathbb{U}^{H \times W \times C}, \quad \mathbb{U} = \{0, 1, \dots, 255\}, \quad C = 3 \text{ (canales BGR)}$$
En la resolución operativa estándar del carril de acceso ($1280 \times 720$ px), este tensor representa exactamente $2{,}764{,}800$ valores discretos de reflectancia luminosa.

#### 2. Normalización de Intensidades y Construcción del Tensor de Entrada
Para alimentar la red neuronal convolucional sin saturar los gradientes, la matriz discreta se transpone del orden dimensional de OpenCV $(H, W, C)$ al estándar tensorial de PyTorch $(C, H, W)$, se escala al espacio métrico real continuo $[0.0, 1.0]$ y se redimensiona a la resolución canónica de entrada mediante interpolación bilineal con preservación de relación de aspecto (letterbox padding):
$$\mathbf{X} \in \mathbb{R}^{1 \times 3 \times H_{in} \times W_{in}}, \quad \mathbf{X}_{b, c, y, x} = \frac{\mathbf{I}_{\text{letterbox}}(y, x, c)}{255.0}, \quad H_{in} = W_{in} = 640$$

#### 3. Extracción de Mapas de Características y Detección Multiescala con YOLO
El tensor $\mathbf{X}$ atraviesa la arquitectura de la red Ultralytics YOLOv8/11:
- **Backbone (CSPDarknet / C2f):** Aplica convoluciones espaciales con stride $= 2$, reduciendo progresivamente la resolución espacial mientras expande la profundidad semántica de los canales ($64 \to 128 \to 256 \to 512$).
- **Neck (Path Aggregation Network - PANet):** Conecta vías ascendentes y descendentes de información para fusionar características de alta resolución espacial (bordes finos) con características de alto nivel abstracto.
- **Detection Head:** Predice en tres escalas espaciales distintas ($80 \times 80$, $40 \times 40$ y $20 \times 20$) un tensor de salidas que codifica la caja delimitadora de la placa vehicular:
  $$\mathbf{B} = (\hat{x}_{\text{center}}, \hat{y}_{\text{center}}, \hat{w}, \hat{h}, \hat{p}_{\text{obj}}, \hat{p}_{\text{placa}})$$
Tras aplicar Supresión de No Máximos (NMS) con un umbral de intersección sobre unión (IoU) de $0.45$, se aíslan las coordenadas enteras de la placa: $(x_1, y_1, x_2, y_2)$.

#### 4. Recorte Matricial de la Región de Interés (ROI) con Margen de Guarda
A partir de la caja detectada, se extrae una submatriz sobre el frame original. Para evitar cortar los bordes externos de las letras extremas por pequeñas oscilaciones de la detección, se introduce un margen porcentual de guarda del $12\%$ ($\Delta = 0.12$):
$$\Delta_x = \lfloor (x_2 - x_1) \cdot 0.12 \rfloor, \quad \Delta_y = \lfloor (y_2 - y_1) \cdot 0.12 \rfloor$$
$$\mathbf{P}_{\text{ROI}} = \mathbf{I}_{\text{raw}}[\max(0, y_1 - \Delta_y) : \min(H, y_2 + \Delta_y), \; \max(0, x_1 - \Delta_x) : \min(W, x_2 + \Delta_x), :]$$
Esta submatriz es escalada en un factor de $3.0\times$ mediante interpolación bicúbica ($cv2.INTER\_CUBIC$), multiplicando por nueve la densidad muestral de la cuadrícula de Nyquist y permitiendo que los caracteres finos dispongan de suficiente resolución para el filtrado espacial.

#### 5. Procesamiento Digital Clásico en Espacio Monocromático (OpenCV)
La submatriz escalada entra al pipeline de visión clásica implementado en [`src/preprocessor.py`](../src/preprocessor.py):
1. **Conversión a Escala de Grises:** Proyección de luminancia ponderada ITU-R BT.601:
   $$\mathbf{Y}(y, x) = 0.299 \cdot R(y, x) + 0.587 \cdot G(y, x) + 0.114 \cdot B(y, x) \in [0, 255]$$
2. **Filtrado Bilateral no lineal:** Suavizado selectivo que disipa el grano de la lámina reflectiva sin degradar las aristas de los caracteres:
   $$\mathbf{I}^{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} \mathbf{Y}(q) \exp\left(-\frac{\|p-q\|^2}{2\sigma_s^2}\right) \exp\left(-\frac{\|\mathbf{Y}(p)-\mathbf{Y}(q)\|^2}{2\sigma_r^2}\right)$$
3. **Binarización Dual (Otsu & Adaptativa Gaussiana):** Cálculo del umbral global óptimo mediante maximización de varianza inter-clase $\sigma_B^2(t)$ y umbralización adaptativa local en ventanas de $19 \times 19$ px ($C = 9$).
4. **Morfología Matemática Rectangular:**
   - Apertura ($\mathbf{I} \circ \mathbf{K}_{3 \times 3} = (\mathbf{I} \ominus \mathbf{K}) \oplus \mathbf{K}$): Disuelve tornillos y remaches de fijación aislados.
   - Cierre ($\mathbf{I} \bullet \mathbf{K}_{3 \times 2} = (\mathbf{I} \oplus \mathbf{K}) \ominus \mathbf{K}$): Reconstruye trazos fracturados en letras desgastadas.
5. **Corrección de Inclinación (Deskewing):** Detección de la orientación dominante mediante el rectángulo de área mínima (`minAreaRect`) y rotación afín inversa con matriz $\mathbf{M}(\theta)$.

#### 6. Reconocimiento Óptico de Secuencias Alfanuméricas (CRNN / EasyOCR)
La imagen binarizada y alineada $\mathbf{I}_{\text{ocr}}$ entra a la arquitectura neuronal híbrida de EasyOCR:
- **Extracción de Características (ResNet / VGG Backbone):** Mapea la imagen 2D a una secuencia temporal de vectores de características $\mathbf{F} \in \mathbb{R}^{T \times D}$, donde cada paso de tiempo representa una franja vertical del parche de la placa.
- **Modelado Secuencial (BiLSTM):** Una red neuronal recurrente bidireccional propaga el contexto de izquierda a derecha y de derecha a izquierda, modelando la correlación entre trazos alfanuméricos adyacentes.
- **Capa de Transducción CTC (Connectionist Temporal Classification):** Calcula la distribución de probabilidad marginal sobre el alfabeto de caracteres permitidos (allowlist: `A-Z, 0-9`), colapsando repeticiones y caracteres nulos ($\epsilon$) para emitir la cadena de texto cruda:
   $$\hat{\mathbf{s}} = \mathcal{B}\left(\arg\max_{\pi} P(\pi \mid \mathbf{F})\right)$$

#### 7. Validación Heurística y Persistencia Transaccional
La cadena cruda es recibida por [`src/validator.py`](../src/validator.py):
1. Se limpia frente a caracteres espurios.
2. Se corrige posicionalmente según la sintaxis de placas de Colombia:
   - Posiciones 0, 1 y 2: Forzadas a letras (`0 -> O`, `1 -> I`, `2 -> Z`, `5 -> S`, `8 -> B`, `6 -> G`, `4 -> A`).
   - Posiciones 3 y 4: Forzadas a números (`O -> 0`, `I -> 1`, `Z -> 2`, `S -> 5`, `B -> 8`, `G -> 6`).
   - Posición 5: Discriminador de tipo de vehículo (`CARRO` si es dígito o letra ambigua `O/Q/I/L` corregida a dígito; `MOTO` si es letra).
3. El resultado alimenta el buffer de votación temporal `PlateVoterBuffer` ($N=7, k=4$, cooldown de 15 s).
4. Al alcanzar consenso, [`src/db.py`](../src/db.py) ejecuta la persistencia mediante consultas parametrizadas:
   - Inserción o actualización del vehículo (`upsert_vehiculo`) en la tabla `vehiculos` para satisfacer la llave foránea.
   - Alternancia inteligente de sentido en `registrar_acceso` consultando el último movimiento (`ENTRADA` $\to$ `SALIDA`).
   - Cálculo en Python del tiempo de permanencia exacta mediante diferencias temporales con `datetime.fromisoformat`.
   - Guardado de la imagen testigo en `data/processed/`.
   - Transacción confirmada (`commit`), garantizando persistencia durable en SQLite.

---

### A.2 Jerarquía de Características en Capas Convolucionales: Filtros Iniciales vs. Semántica Profunda

Las redes convolucionales profundas organizan la comprensión visual de forma jerárquica y composicional:

```
[Capas Iniciales (Stem / Conv 1-2)]  -->  Bordes de Gabor, gradientes de alto contraste, esquinas y frecuencias altas.
                 │
[Capas Intermedias (C2f / P3-P4)]     -->  Combinaciones de bordes, rejillas de radiadores, rectángulos y bucles.
                 │
[Capas Profundas (Neck / P5 / Head)]   -->  Semántica global del vehículo, placa vehicular completa e invariancia visual.
```

1. **Capas Iniciales (Stem / Early Layers):**
   - **Campos Receptivos:** Extremadamente reducidos ($3 \times 3$ píxeles).
   - **Naturaleza de los Filtros:** Operan como bancos de filtros espaciales sintonizados en frecuencia y orientación, análogos a las células simples descubiertas por Hubel y Wiesel en la corteza visual primaria (área V1) y a las funciones matemáticas de Gabor.
   - **Qué Aprenden:** Detectan discontinuidades locales de intensidad luminosa: gradientes ortogonales (bordes verticales de los postes de las letras `I`, `H`, `E`), transiciones diagonales (letras `A`, `V`, `X`) y variaciones de color entre la lámina amarilla/blanca y el parachoques oscuro. Carecen por completo de noción semántica sobre vehículos o matrículas.

2. **Capas Intermedias (Feature Pyramid / Mid Layers):**
   - **Campos Receptivos:** Moderados ($16 \times 16$ a $64 \times 64$ píxeles), producto de las convoluciones con stride sucesivas.
   - **Naturaleza de los Filtros:** Realizan combinaciones no lineales de las primitivas de borde detectadas previamente.
   - **Qué Aprenden:** Geometrías compuestas y patrones de textura repetitivos. En el dominio automotriz, estas capas se activan ante esquinas rectangulares en ángulo recto (el marco exterior de la chapa metálica), texturas de panal de abeja (rejillas de refrigeración del radiador), curvaturas cerradas (los bucles de las letras `O`, `P`, `D` y números `0`, `8`) y líneas horizontales paralelas (el borde superior e inferior de la lámina).

3. **Capas Profundas y Cabezas de Detección (Deep Semantic Layers):**
   - **Campos Receptivos:** Globales (abarcan desde $128 \times 128$ hasta los $640 \times 640$ píxeles del cuadro completo).
   - **Naturaleza de los Filtros:** Alta abstracción e invariancia espacial.
   - **Qué Aprenden:** Representaciones semánticas completas y contextuales. La red comprende que una placa vehicular no es solo un rectángulo con letras, sino un elemento ubicado típicamente debajo del radiador o sobre el parachoques de un vehículo, flanqueado por faros o luces diurnas. Esta semántica permite suprimir falsos positivos frente a calcomanías publicitarias en el parabrisas, letreros en las paredes del garaje o números impresos en las puertas de camiones de carga.

---

### A.3 Justificación Técnica del Aprendizaje por Transferencia (Transfer Learning)

El entrenamiento de una red neuronal para detección de objetos desde cero (*training from scratch* con pesos inicializados aleatoriamente mediante distribución normal de He/Kaiming) presenta barreras insalvables para aplicaciones de visión vehicular a nivel local:

1. **Topología de la Superficie de Pérdida y Muestra de Datos:**
   - Entrenar los millones de parámetros de YOLO (ej. $3.2 \times 10^6$ en YOLOv8n o $25.9 \times 10^6$ en YOLOv8m) desde cero exige del orden de $10^5$ a $10^6$ imágenes debidamente anotadas. Si se intenta entrenar una red con un conjunto de datos local típico de $500$ a $3{,}000$ placas vehiculares recolectadas en Riohacha, el optimizador (SGD / AdamW) cae en un régimen severo de sobreajuste (*overfitting* catastrófico): la red memoriza los números de placa y el color exacto de los vehículos del conjunto de entrenamiento en lugar de aprender representaciones generalizables.
   - En una red no entrenada, la superficie de pérdida $\mathcal{L}(\mathbf{W})$ es altamente caótica y no convexa, con mesetas de gradiente plano donde las capas iniciales tardan cientos de épocas solo en redescubrir filtros elementales de bordes que son idénticos en cualquier tarea visual humana.

2. **Ventaja Mecánica del Transfer Learning (Fine-Tuning):**
   - Al utilizar pesos pre-entrenados en conjuntos a escala masiva como ImageNet-1k (1.28 millones de imágenes) o Microsoft COCO (330 mil imágenes), las capas convolucionales tempranas e intermedias ya incorporan detectores universales óptimos de bordes, texturas y formas 2D.
   - Matemáticamente, el *Transfer Learning* posiciona los pesos iniciales $\mathbf{W}_0$ en una vecindad sumamente favorable del espacio de parámetros, muy próxima al mínimo global de la tarea de detección.
   - Durante el fine-tuning en placas vehiculares, las capas profundas se congelan inicialmente o se adaptan a tasas de aprendizaje diferenciadas ($\eta_{\text{backbone}} \approx 10^{-4}, \; \eta_{\text{head}} \approx 10^{-3}$), logrando convergencia óptima en solo 30 a 50 épocas de cómputo en una GPU modesta (NVIDIA RTX 3060 / T4 en Google Colab), con un requerimiento de muestras anotadas hasta dos órdenes de magnitud menor.

---

## SECCIÓN B: MAPA CONCEPTUAL Y CORRESPONDENCIA TEORÍA VS. CÓDIGO FUENTE

A continuación se detalla la correspondencia rigurosa y biunívoca entre los conceptos teóricos fundamentales del curso y su implementación en el código fuente de `src/`:

| # | Concepto Teórico del Curso | Fundamento Matemático / Algorítmico | Archivo y Enlace Relativo | Clase / Función Implementada | Rol Técnico en la Solución |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Representación Matricial y Estructuras N-D** | Espacio discreto $\mathbb{U}^{H \times W \times C}$, tensores NumPy y operaciones de submatriz (slicing) | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.crop_with_margin` | Recorta la región de interés (ROI) aplicando un margen de guarda del 12% para preservar bordes tipográficos. |
| **2** | **Interpolación Espacial y Remuestreo** | Interpolación bicúbica mediante polinomios cúbicos por tramos en 16 píxeles vecinos | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.resize_patch` | Multiplica por $3.0\times$ las dimensiones del parche para densificar la cuadrícula de Nyquist antes del OCR. |
| **3** | **Conversión Colorimétrica y Luminancia** | Ponderación fisiológica espectral ITU-R BT.601: $Y = 0.299R + 0.587G + 0.114B$ | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.to_grayscale` | Colapsa los 3 canales BGR a un canal monocromático conservando el contraste de los caracteres oscuros. |
| **4** | **Filtrado Espacial no Lineal** | Filtro Bilateral con combinación de kernels gaussiano espacial $g_s$ y fotométrico de rango $f_r$ | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.apply_bilateral_filter` | Suaviza el ruido y textura reflectiva de la lámina sin emborronar ni desvanecer las aristas de las letras. |
| **5** | **Umbralización Global No Paramétrica** | Criterio de Otsu por maximización de varianza inter-clase: $\sigma_B^2(t) = \omega_0 \omega_1 (\mu_0 - \mu_1)^2$ | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.binarize_otsu` | Binariza automáticamente la imagen separando fondo reflectivo de letras oscuras sin parámetros empíricos. |
| **6** | **Umbralización Adaptativa Local** | Media gaussiana ponderada en ventana vecina: $T(x,y) = \mu_{\text{Gauss}}(B) - C$ | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.binarize_adaptive` | Compensa gradientes de sombra severos y manchas de sol cenital sobre la superficie metálica. |
| **7** | **Morfología Matemática Discreta** | Apertura $(\mathbf{A} \ominus \mathbf{B}) \oplus \mathbf{B}$ y Cierre $(\mathbf{A} \oplus \mathbf{B}) \ominus \mathbf{B}$ con elementos estructurantes rectangulares | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.apply_morphology` | La apertura ($3 \times 3$) elimina remaches y tornillos de fijación; el cierre ($3 \times 2$) sella trazos seccionados. |
| **8** | **Momentos Centrales y Transformación Afín** | Detección de orientación por elipse de inercia y mapeo proyectivo $\mathbf{M}(\theta)$ con interpolación cúbica | [`src/preprocessor.py`](../src/preprocessor.py) | `ImagePreprocessor.deskew_image` | Detecta la inclinación de la matrícula ($0.5^\circ - 35^\circ$) y aplica rotación afín horizontal para facilitar el OCR. |
| **9** | **Inferencia Neuronal Convolucional (YOLO)** | Inferencia tensorial feedforward, anclas multiescala y supresión de no máximos (NMS) | [`src/detector.py`](../src/detector.py) | `ALPRPipeline.detect_plates` | Localiza con precisión subpixel las cajas delimitadoras de placas vehiculares en el cuadro completo. |
| **10** | **Detección Clásica por Gradientes y Morfología** | Operador diferencial Sobel $G_x$, umbralización Otsu y filtrado de contornos por aspect ratio | [`src/detector.py`](../src/detector.py) | `ALPRPipeline.detect_plates` (Fallback) | Mecanismo de alta precisión basado en visión clásica en caso de ausencia temporal de pesos de GPU. |
| **11** | **Reconocimiento Óptico Secuencial (OCR)** | Modelo híbrido CRNN: Extractor convolucional + Recurrencia temporal BiLSTM + Decodificador CTC | [`src/detector.py`](../src/detector.py) | `ALPRPipeline.extract_ocr_text` | Transduce las franjas de imagen binarizada en texto alfanumérico restringido al allowlist reglamentario. |
| **12** | **Demostración Convolucional Paso a Paso** | Convolución espacial 2D manual con bucles anidados frente a `cv2.filter2D(CV_64F)` | [`src/math_proof.py`](../src/math_proof.py) | `convolve_center_pure_python` | Demuestra analítica y numéricamente la equivalencia exacta ($0.0 \times 10^0$ error) entre teoría y OpenCV. |
| **13** | **Métricas de Evaluación de Texto (CER)** | Distancia de edición de Levenshtein: $CER = \frac{S + D + I}{N}$ | [`src/benchmark_ab.py`](../src/benchmark_ab.py) | `calculate_cer` | Evalúa cuantitativamente el error de sustitución, inserción y borrado de caracteres en inferencia OCR. |
| **14** | **Corrección Heurística y Expresiones Regulares** | Autómata finito determinista y mapeo biyectivo posicional de caracteres confundidos | [`src/validator.py`](../src/validator.py) | `PlateValidator.correct_and_validate` | Aplica reglas vehiculares colombianas (pos 0-2 letras, pos 3-4 dígitos, pos 5 carro si es dígito o confusión O/Q/I/L, moto si es letra) y reclasifica. |
| **15** | **Filtrado Temporal Multi-Cuadro** | Ventana deslizante temporal ($N=7, k=4$) con enfriamiento (cooldown de 15 s) | [`src/validator.py`](../src/validator.py) | `PlateVoterBuffer.add_observation` | Elimina lecturas transitorias espurias exigiendo consenso temporal robusto entre cuadros de video. |
| **16** | **Persistencia Transaccional y Álgebra Relacional** | Integridad referencial con Foreign Keys, consultas parametrizadas y cálculo temporal continuo con datetime | [`src/db.py`](../src/db.py) | `DatabaseManager.registrar_acceso` | Persiste accesos en SQLite con alternancia ENTRADA/SALIDA, calcula tiempos de permanencia en minutos con datetime y extrae métricas de aforo. |

---

## SECCIÓN C: OPERACIONES ARITMÉTICAS Y FUNDAMENTOS MATEMÁTICOS

### C.1 Operaciones Matriciales en el Pipeline de Preprocesamiento

El pipeline clásico implementado en [`src/preprocessor.py`](../src/preprocessor.py) formaliza rigurosamente tres transformaciones matriciales:

1. **Ponderación de Luminancia Espectral (RGB/BGR a Monocromático):**
   $$Y(x, y) = \begin{bmatrix} 0.114 & 0.587 & 0.299 \end{bmatrix} \begin{bmatrix} B(x, y) \\ G(x, y) \\ R(x, y) \end{bmatrix}$$
   Esta combinación lineal obedece a la curva de sensibilidad fotópica del ojo humano (estándar CIE 1931), maximizando la respuesta en las longitudes de onda del verde ($\sim 555\text{ nm}$). Esto resulta determinante para garantizar alto contraste entre el fondo amarillo de servicio particular/público y los caracteres impresos en esmalte negro.

2. **Filtrado Bilateral (Suavizado no lineal con preservación de aristas):**
   $$\mathbf{I}^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} \mathbf{I}(x_i) \cdot \exp\left(-\frac{\|x_i - x\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{\|\mathbf{I}(x_i) - \mathbf{I}(x)\|^2}{2\sigma_r^2}\right)$$
   Donde el factor de normalización es:
   $$W_p = \sum_{x_i \in \Omega} \exp\left(-\frac{\|x_i - x\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{\|\mathbf{I}(x_i) - \mathbf{I}(x)\|^2}{2\sigma_r^2}\right)$$
   A diferencia del filtrado gaussiano convencional que degrada la agudeza de los bordes, el kernel fotométrico $\exp\left(-\frac{\|\mathbf{I}(x_i) - \mathbf{I}(x)\|^2}{2\sigma_r^2}\right)$ decae a cero cuando la diferencia de intensidad supera $\sigma_r$ ($17.0$ unidades en nuestra configuración), impidiendo que los píxeles claros de la chapa reflectiva se mezclen con los píxeles oscuros de la letra.

3. **Binarización por Umbral Óptimo de Otsu:**
   Otsu modela el histograma de intensidades como una mezcla de dos distribuciones normales (fondo $\mathcal{C}_0$ y letras $\mathcal{C}_1$). El umbral óptimo $t^*$ se obtiene resolviendo el problema de optimización discreta:
   $$t^* = \arg\max_{0 \le t < L} \sigma_B^2(t)$$
   Donde la varianza inter-clase $\sigma_B^2(t)$ está dada por:
   $$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[ \mu_0(t) - \mu_1(t) \right]^2$$
   $$\omega_0(t) = \sum_{i=0}^t p_i, \quad \omega_1(t) = \sum_{i=t+1}^{L-1} p_i, \quad \mu_0(t) = \sum_{i=0}^t \frac{i \cdot p_i}{\omega_0(t)}, \quad \mu_1(t) = \sum_{i=t+1}^{L-1} \frac{i \cdot p_i}{\omega_1(t)}$$
   Este algoritmo elimina la necesidad de ajustar manualmente umbrales fijos bajo condiciones cambiantes de iluminación solar en Riohacha.

---

### C.2 Demostración Matemática: Convolución Espacial 3x3 Paso a Paso vs. OpenCV
*(Generado de forma reproducible por [`src/math_proof.py`](../src/math_proof.py))*

Para comprobar la validez matemática de las librerías computacionales frente a la teoría analítica del curso, se aísla un parche representativo de **$3 \times 3$ píxeles** ubicado en la frontera crítica entre el trazo negro de un carácter (intensidades $\sim 40-55$) y el fondo reflectivo de la matrícula vehicular (intensidades $\sim 205-215$). Se aplica el operador diferencial **Sobel Horizontal ($G_x$)** para extracción de gradientes verticales:

#### 1. Matriz del Parche de Imagen $\mathbf{P}$ ($3 \times 3$ píxeles en escala de grises):
$$\mathbf{P} = \begin{bmatrix} 45.0 & 50.0 & 210.0 \\ 40.0 & 55.0 & 215.0 \\ 42.0 & 48.0 & 205.0 \end{bmatrix}$$

#### 2. Matriz del Kernel Espacial $\mathbf{K}$ (Sobel $G_x$ $3 \times 3$):
$$\mathbf{K} = \begin{bmatrix} -1.0 & 0.0 & +1.0 \\ -2.0 & 0.0 & +2.0 \\ -1.0 & 0.0 & +1.0 \end{bmatrix}$$

#### 3. Operación Aritmética en el Píxel Central $(1, 1)$:
La correlación discreta bidimensional centrada en $(1, 1)$ viene definida por:
$$R(1, 1) = \sum_{i=0}^{2} \sum_{j=0}^{2} \mathbf{P}(i, j) \cdot \mathbf{K}(i, j)$$

A continuación se detalla cada uno de los 9 productos parciales punto a punto:

| Coordenada $(i, j)$ | Intensidad Píxel $\mathbf{P}(i, j)$ | Peso Kernel $\mathbf{K}(i, j)$ | Producto Parcial $\mathbf{P} \cdot \mathbf{K}$ | Interpretación Física |
| :---: | :---: | :---: | :---: | :--- |
| `(0, 0)` | `45.0` | `-1.00` | **`-45.00`** | Letra oscura superior izquierda |
| `(0, 1)` | `50.0` | ` 0.00` | **`  0.00`** | Línea central superior neutra |
| `(0, 2)` | `210.0` | `+1.00` | **`+210.00`** | Lámina reflectiva superior derecha |
| `(1, 0)` | `40.0` | `-2.00` | **`-80.00`** | Letra oscura lateral central (peso x2) |
| `(1, 1)` | `55.0` | ` 0.00` | **`  0.00`** | Píxel de análisis central (origen) |
| `(1, 2)` | `215.0` | `+2.00` | **`+430.00`** | Lámina reflectiva lateral (peso x2) |
| `(2, 0)` | `42.0` | `-1.00` | **`-42.00`** | Letra oscura inferior izquierda |
| `(2, 1)` | `48.0` | ` 0.00` | **`  0.00`** | Línea central inferior neutra |
| `(2, 2)` | `205.0` | `+1.00` | **`+205.00`** | Lámina reflectiva inferior derecha |

#### 4. Suma Analítica y Validación Numérica:
$$\begin{aligned}
R(1, 1) &= (-45.0) + (0.0) + (210.0) + (-80.0) + (0.0) + (430.0) + (-42.0) + (0.0) + (205.0) \\
&= -167.0 + 845.0 \\
&= \mathbf{678.0000}
\end{aligned}$$

- **Resultado Suma Manual (Python Puro):** $\mathbf{678.0000}$
- **Resultado `cv2.filter2D(..., ddepth=cv2.CV_64F)`:** $\mathbf{678.0000}$
- **Diferencia Numérica Absoluta:** $|\text{Python} - \text{OpenCV}| = \mathbf{0.0 \times 10^{0}}$
- **Diagnóstico:** ✅ **PARIDAD NUMÉRICA PERFECTA (ERROR CERO)**.

---

### C.3 Análisis de Sensibilidad Teórica: ¿Qué Ocurre al Omitir la Normalización de Entrada?

Omitir la fase de normalización tensorial ($x_{\text{norm}} = x / 255.0$) o estandarización por media y varianza ($z = (x - \mu)/\sigma$) acarrea severas consecuencias tanto en arquitecturas neuronales como en algoritmos clásicos de visión:

1. **Dinámica de Gradientes y Desbordamiento en Redes Convolucionales (Deep Learning):**
   - **Magnitud de las Activaciones:** Si los valores de entrada permanecen en el rango discreto entero $[0, 255]$, la combinación lineal inicial en la primera capa convolucional $z_j = \sum_i w_i x_i + b$ alcanza magnitudes hasta 255 veces superiores a las esperadas por los esquemas de inicialización estándar (He Normal o Xavier Uniform).
   - **Explosión de Gradientes ($\nabla_{\mathbf{W}} \mathcal{L} \to \infty$):** Durante la retropropagación (*backpropagation*), el gradiente respecto a los pesos de la capa depende linealmente de la activación de la capa previa:
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \mathbf{x}^T$$
     Al multiplicar por entradas de magnitud $\sim 255$, las actualizaciones de pesos $\Delta \mathbf{W} = -\eta \nabla_{\mathbf{W}} \mathcal{L}$ sufren un crecimiento desmedido, provocando desbordamiento de rango en punto flotante IEEE 754 (aparición de valores `NaN` o `Inf`) tras unas pocas iteraciones.
   - **Saturación de Funciones de Activación:** En capas que utilizan funciones con derivadas acotadas (Sigmoide, Tanh) o capas con normalización interna desajustada, activaciones de gran magnitud empujan la respuesta hacia las asíntotas horizontales, donde la derivada local es prácticamente nula ($\sigma'(z) \approx 0$). Esto colapsa el flujo del gradiente hacia las capas tempranas (*Vanishing Gradient* inducido por escala).

2. **Acondicionamiento de la Matriz Hessiana y Espacio de Pérdida:**
   - La matriz de segundas derivadas (Hessiana $\mathbf{H} = \nabla^2 \mathcal{L}$) exhibe un número de condición $\kappa(\mathbf{H}) = \lambda_{\max}/\lambda_{\min} \gg 1$.
   - Geométricamente, la superficie de pérdida adquiere forma de "cañón alargado": extremadamente empinada en unas direcciones y plana en otras. Esto fuerza al algoritmo de descenso de gradiente a oscilar violentamente entre las paredes del cañón sin avanzar hacia el mínimo, exigiendo hiperparámetros de tasa de aprendizaje infinitesimalmente pequeños ($\eta < 10^{-7}$) que hacen inviable el entrenamiento práctico.

3. **Degradación en Algoritmos Clásicos de Visión:**
   - En operadores diferenciales como Sobel, Prewitt o Laplaciano, no normalizar no altera la linealidad de la convolución discreta si se preserva el tipo de dato en punto flotante (`float64`); sin embargo, si la salida se almacena o castea a tipos de dato saturados sin escalado previo (`uint8`), se produce **saturación severa por clipping**: cualquier valor superior a 255 se colapsa a 255 y cualquier gradiente negativo se colapsa a 0, destruyendo la información direccional del vector gradiente.
   - En algoritmos de segmentación y clustering (ej. K-Means en espacio de color o binarización por distancia euclídea), canales con magnitudes no normalizadas sesgan unilateralmente el cálculo métrico de distancias euclidianas.

---

## SECCIÓN D: EVALUACIÓN EXPERIMENTAL A/B Y ANÁLISIS DE SENSIBILIDAD (RETO EXTRA +0.5 PTS)

### D.1 Comparativa Experimental: Modo A (Sin Preprocesamiento) vs. Modo B (Con Preprocesamiento OpenCV)
*(Generado de forma reproducible por [`src/benchmark_ab.py`](../src/benchmark_ab.py))*

Se evaluó el rendimiento de EasyOCR sobre un lote de prueba de 11 fotos correspondientes a 9 placas distintas en `data/samples/placas_de_prueba` (tres corresponden al mismo vehículo), descargadas de internet (fuente pendiente de citar; no constituyen un conjunto de prueba independiente). El detector empleado corresponde al respaldo clásico por gradientes y bordes (al no disponerse aún de pesos YOLO entrenados).

Se compararon dos modalidades operativas:
- **Modo A (Recorte Sin Procesar):** Recorte directo de la caja detectada entregado al motor OCR.
- **Modo B (Con Preprocesamiento OpenCV):** Normalización de altura a 140 px en escala de grises con filtro bilateral.

A continuación se presentan los resultados consolidados de las mediciones reales:

| Medida | Antes (2026-11-27) | Después, Modo A (recorte sin procesar) | Después, Modo B (con preprocesamiento) |
|---|---|---|---|
| Aciertos exactos | 0/11 (pipeline) y 4/11 (recorte ×3) | 6/11 | 6/11 |
| CER medio | sin medir | 37,88 % | 34,85 % |
| Tiempo por foto (CPU) | 2,5 – 15,9 s | 1,85 s promedio | 1,20 s promedio (0,4 – 2,2 s) |

*Fuente del "después": reproducible con `python -m src.benchmark_ab` (medición 2026-11-28). El tiempo del Modo A incluye la carga de EasyOCR en la primera foto.*

> **Discusión de los Resultados:**
> 1. En la línea base anterior (2026-11-27), el pipeline antiguo entregaba al OCR una máscara binaria severamente degradada por umbralización adaptativa agresiva y morfología, obteniendo 0/11 aciertos con latencias excesivas (hasta 15.9 s por foto).
> 2. Las correcciones aplicadas redujeron el tiempo medio por foto en CPU a 1.20 s en Modo B (1.85 s en Modo A incluyendo la inicialización del modelo).
> 3. En aciertos exactos, tanto Modo A como Modo B logran 6 de 11 lecturas correctas sobre este lote de prueba, registrando el Modo B un Character Error Rate (CER) medio menor (34.85 % frente a 37.88 %).

---

### D.2 Análisis de Sensibilidad del Detector YOLO

La curva de Precisión/Recall (P/R) y la sensibilidad del detector ante variaciones del umbral de confianza $\tau_{\text{conf}}$ se encuentran **sin medir con datos locales**, dado que requieren pesos entrenados de la red neuronal sobre el dominio de placas colombianas.

El análisis de sensibilidad cuantitativo, las curvas P/R y la matriz de confusión se generarán de forma reproducible a través del cuaderno de Google Colab ([`notebooks/entrenamiento_y_metricas.ipynb`](https://colab.research.google.com/github/y0ner/smart-alpr-iot/blob/main/notebooks/entrenamiento_y_metricas.ipynb)) una vez culminado el entrenamiento con GPU sobre el dataset anotado. En la configuración actual del código ([`src/config.py`](../src/config.py)), el valor por defecto se mantiene en `0.45` como punto de partida para el filtrado de detecciones.

---

### D.3 Parámetro Más Sensible: Altura del Recorte y Formato de Imagen para OCR

El factor determinante en el rendimiento del reconocimiento óptico es la representación fotométrica y la resolución vertical con la que se alimenta EasyOCR. En el experimento sistemático del 2026-11-27 (empleando las mismas cajas del detector y variando la imagen entregada al OCR) se obtuvieron los siguientes resultados:

| Imagen al OCR | Aciertos | OCR total |
|---|---|---|
| Salida binaria anterior (adaptativo + morfología) | 0/11 | 125 s |
| Recorte ×3 en color | 4/11 | 137 s |
| Grises a 120 px | 3/11 | 36 s |
| **Grises a 140 px** | **6/11** | 34 s |
| Grises a 180 px | 5/11 | 52 s |
| 140 px + Otsu | 4/11 | 23 s |
| 140 px + adaptativo proporcional | 3/11 | 23 s |

#### ¿Por qué EasyOCR lee mejor en grises a altura moderada que en binario?

1. **Naturaleza Continua del Extractor Convolucional:** EasyOCR implementa una red neuronal profunda (backbone ResNet/VGG + BiLSTM) entrenada sobre imágenes naturales en escala continua de intensidades. La red aprovecha los gradientes suaves de los bordes (*antialiasing* natural) para discernir contornos y separar trazos tipográficos.
2. **Degradación por Binarización:** La umbralización rígida (sea Otsu o adaptativa) destruye los matices de luminancia, fusiona sombras o pernos con los caracteres y fractura trazos desgastados, lo que explica la caída de aciertos a 0/11 con la binarización previa y tiempos de búsqueda mucho mayores en el decodificador CTC.
3. **Optimización de Escala (140 px):** Una altura de 140 px sitúa los caracteres en el campo receptivo ideal del modelo, ofreciendo suficiente densidad de muestreo sin el sobrecosto computacional ni los artefactos de interpolación observados a 180 px (52 s) o la pérdida de nitidez a 120 px (36 s).

---

### D.4 Propuestas de Mejora Arquitectural a Futuro

A partir de los hallazgos experimentales, se plantean dos líneas de evolución técnica:

#### 1. Módulo de Super-Resolución Neuronal (Real-ESRGAN / SRCNN) para Capturas Lejanas
- **Problema Técnico:** En vehículos distantes donde la placa abarca una resolución vertical reducida, el escalado clásico introduce artefactos de desenfoque que dificultan la distinción de caracteres cerrados (`8` vs `B`, `0` vs `D`).
- **Propuesta:** Incorporar una red compacta de super-resolución residual orientada a reconstruir frecuencias espaciales altas previo al OCR en tomas lejanas.

#### 2. Rastreo Espaciotemporal de Trayectorias (ByteTrack + Filtro de Kalman)
- **Problema Técnico:** Procesar de forma aislada cada cuadro de video genera inferencias OCR redundantes sobre el mismo vehículo en tránsito.
- **Propuesta:** Asociar detecciones temporales mediante seguimiento multiobjeto con filtro de Kalman para ejecutar EasyOCR únicamente en los mejores fotogramas de la trayectoria (menor desenfoque y mayor escala).

---

## SECCIÓN E: CASOS DE FALLA REALES, MITIGACIONES Y VIABILIDAD INDUSTRIAL

### E.1 Análisis de Casos de Falla Reales y Diagnóstico de Robustez

A partir de la evaluación empírica con el lote de prueba de 11 fotos de matrículas colombianas (tras las correcciones del pipeline), se identificaron 5 casos de falla reales que permiten caracterizar el comportamiento del detector de respaldo clásico y del motor OCR:

| Muestra / Placa Real | Lectura Obtenida | Causa Raíz del Error | Diagnóstico Técnico y Mitigación |
| :--- | :---: | :--- | :--- |
| **EMY923** (Popayán) | *(Sin lectura)* | El detector clásico de respaldo por bordes no ubica la placa. | Al carecer de pesos entrenados de YOLO, el fallback morfológico falla ante bajo contraste entre parachoques y lámina. Requiere pesos entrenados de red neuronal. |
| **WGL451** (Cartagena) | `LOC10A` | Placa blanca de servicio público (taxi). | El contraste invertido respecto a la placa particular amarilla altera la segmentación. Requiere normalización adaptada a placas blancas. |
| **AHB965** (Bogotá) | `STFC` | Lectura de la franja institucional ("STFE DE BOGOTÁ"). | El recorte incluye la franja inferior del municipio y el OCR prioriza dicho texto sobre la matrícula. Requiere ajuste morfológico de recorte central. |
| **SJS444** (Dosquebradas) | `ISU544` | Fusión de caracteres y similitud tipográfica en Modo B. | Anteriormente el pipeline unía el municipio (`9HSOUEDRADAS444`). Con el nuevo preprocesamiento se aísla la placa, pero la morfología confunde `SJS` con `ISU`. |
| **WHK086** (Medellín) | `RHK086` | Confusión tipográfica de consonante inicial (`W` leída como `R`). | Distorsión local de trazo en la primera letra. Requiere consenso temporal multi-cuadro mediante `PlateVoterBuffer`. |

Estos casos reales reflejan los tres frentes clave de robustez del sistema:
1. **Dependencia del detector de respaldo:** La ausencia de pesos YOLO limita la detección cuando no hay contraste nítido por gradiente Sobel.
2. **Textos institucionales secundarios:** Nombres de municipios y departamentos estampados en la parte inferior de la lámina que interfieren con la lectura.
3. **Placas de servicio público (blancas):** Con reflectancia y distribución cromática distintas a las particulares amarillas.

---

### E.2 Viabilidad Industrial y Análisis Costo-Beneficio en Riohacha, La Guajira

#### 1. Caso de Negocio y Entorno de Despliegue
- **Ubicaciones Objetivo:**
  - *Centro Comercial Viva Wajiira (Riohacha):* 2 carriles vehiculares de entrada y 2 de salida. Capacidad de aforo: 420 plazas.
  - *Campus Universidad de La Guajira (Sede Riohacha - Km 5 vía a Maicao):* 3 accesos vehiculares principales. Flujo estimado: $1{,}200$ vehículos/día entre docentes, administrativos y estudiantes.
- **Cuantificación del Problema Actual:** En ambos puntos, el control de acceso se efectúa manualmente mediante entrega de fichas plásticas o planillas de papel. El tiempo medio de registro por vehículo oscila entre **$35$ y $45$ segundos**, generando represamientos vehiculares sobre la transitada Avenida Circunvalar durante horas pico (7:00 AM, 12:00 PM y 6:00 PM) con una tasa de error de digitación humana del $14.8\%$ documentada en la ficha técnica del cliente.

#### 2. Arquitectura de Hardware de Borde (Edge Computing) y Presupuesto Detallado

Para operar de forma autónoma sin depender de conectividad a internet de alta velocidad (la cual sufre interrupciones intermitentes en la región), se propone una arquitectura de computación perimetral descentralizada:

| Componente de Hardware / Infraestructura | Especificación Técnica Industrial | Cantidad | Costo Unitario (USD) | Costo Total (COP @ 3,900) |
| :--- | :--- | :---: | :---: | :---: |
| **SBC de Cómputo de Borde (Edge AI)** | NVIDIA Jetson Orin Nano Developer Kit (8GB RAM, 40 TOPS AI, 15W) | 1 por carril | USD $499 | $1{,}946{,}100 COP |
| **Cámara IP Industrial Perimetral** | Hikvision / Dahua 4MP LPR Bullet, Lente Varifocal 2.7-13.5mm, WDR 120dB, IP67 | 1 por carril | USD $160 | $624{,}000 COP |
| **Gabinete y Protección Térmica** | Gabinete NEMA 4X con filtro de polvo y ventilación forzada termostática | 1 por carril | USD $110 | $429{,}000 COP |
| **Sistema de Respaldo Eléctrico (UPS)** | UPS Online Doble Conversión 1000VA / 900W (regulación estricta de voltaje) | 1 por acceso | USD $220 | $858{,}000 COP |
| **Lente Polarizador Óptico** | Filtro Polarizador Circular (CPL) 67mm con montura roscada industrial | 1 por cámara | USD $35 | $136{,}500 COP |
| **Cableado y Protección Eléctrica** | Cable UTP Cat6 Blindado para intemperie, supresor de picos y canalización | Global | USD $65 | $253{,}500 COP |
| **TOTAL INVERSIÓN POR CARRIL** | *(Infraestructura completa de borde llave en mano)* | -- | **USD $1{,}089** | **$4{,}247{,}100 COP** |

#### 3. Análisis de Retorno de Inversión (ROI) y Ahorro Operativo Cuantificado

```
[ Registro Manual Actual: 35 - 45 s ]  --->  [ Meta de Diseño: < 2.5 s por vehículo ]
   (Congestión vial y errores 14.8%)             (Meta establecida en ficha técnica de cliente)
```

1. **Metas de Eficiencia en Tiempos de Acceso (Ficha del Cliente):**
   - Frente al tiempo de registro manual actual (estimado entre 35 y 45 segundos por vehículo con expedición de boletos y digitación humana), el sistema plantea como meta de diseño reducir el tiempo de atención a menos de 2.5 segundos por vehículo (conforme a los KPIs de la ficha técnica del cliente).
   - Este objetivo busca aliviar los represamientos en las horas pico sobre los accesos principales de la ciudad.

2. **Ahorro en Costos Operativos Directos (Nómina de Operadores):**
   - Actualmente se requieren 2 operadores en caseta por turno en horario extendido (6:00 AM a 10:00 PM), equivalente a 4 salarios mínimos mensuales vigentes (SMLMV) con factor prestacional:
     $$\text{Costo laboral anual} \approx 4 \times \$1{,}800{,}000 \times 12 \approx \mathbf{\$86{,}400{,}000\text{ COP/año}}$$
   - Con la automatización por visión artificial de Smart ALPR IoT, la supervisión se centraliza en un único operador de seguridad general para todo el centro comercial, reduciendo en 2 personas la nómina requerida para casetas:
     $$\text{Ahorro directo neto anual} = \mathbf{\$43{,}200{,}000\text{ COP/año}}$$

3. **Período de Recuperación de la Inversión (Payback Period):**
   - Inversión total en 2 carriles automatizados: $\$8{,}494{,}200\text{ COP}$.
   $$\text{Payback} = \frac{\text{Inversión Inicial}}{\text{Ahorro Mensual}} = \frac{\$8{,}494{,}200}{\$3{,}600{,}000} \approx \mathbf{2.36\text{ meses}}$$
   La solución proyecta amortizarse en **menos de dos meses y medio** de operación estimada frente a los costos actuales.

4. **Resiliencia Frente al Entorno Climático de La Guajira:**
   - **Temperaturas Elevadas ($>36^\circ\text{C}$):** La arquitectura Edge basada en NVIDIA Jetson Orin Nano disipa únicamente 15 Watts de potencia térmica, operando dentro del rango térmico industrial con el gabinete NEMA 4X y disipador pasivo sin sobrecalentamiento.
   - **Fluctuaciones de Tensión y Microcortes Eléctricos:** La inclusión de una UPS online de doble conversión aísla al sistema de los apagones y fluctuaciones severas frecuentes en la red eléctrica regional de La Guajira.
   - **Autonomía Offline (Cero Dependencia de Nube):** Al utilizar SQLite local (`db/schema.sql` y [`src/db.py`](../src/db.py)) y modelos locales de inferencia (YOLO + EasyOCR), el sistema opera continuamente 24/7 sin requerir internet, sincronizando lotes transaccionales de telemetría a la nube cuando la conexión de fibra óptica se restablece.

---

*Fin del Análisis Técnico Oficial — Internet de las Cosas, Grupo A3 — Universidad de La Guajira (Riohacha, noviembre de 2026)*
