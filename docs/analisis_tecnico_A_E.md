# ANÁLISIS TÉCNICO COMPLETO (SECCIÓN 06: A - E)
## Sistema Inteligente Lector de Placas Vehiculares (Smart ALPR IoT)
*Universidad de La Guajira — Facultad de Ingeniería — Visión Artificial*

---

> **Estado:** Avance continuo por etapas.  
> **Etapa 3:** Demostración matemática 3x3 y pipeline OpenCV completados.

---

### SECCIÓN C: OPERACIONES ARITMÉTICAS Y FUNDAMENTOS MATEMÁTICOS

#### C.1 Operaciones Matriciales en el Pipeline
El pipeline de visión clásica implementado en `src/preprocessor.py` hace uso exhaustivo del álgebra lineal y transformaciones matriciales discretas:
1. **Ponderación de Luminancia (RGB a Grises):**
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
   Refleja la sensibilidad espectral del ojo humano a las longitudes de onda del verde, preservando el contraste entre la lámina amarilla/blanca y los caracteres negros.
2. **Filtrado Bilateral (Suavizado no lineal):**
   $$I^{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$
   Atenúa el grano de sensor y la textura de la lámina reflectiva mediante el kernel espacial $g_s$ mientras el kernel de rango fotométrico $f_r$ bloquea la mezcla de intensidades a través de los bordes abruptos de los caracteres.
3. **Umbralización de Otsu (Maximización de Varianza Inter-Clase):**
   $$\sigma_B^2(t) = \omega_0(t) \omega_1(t) \left[ \mu_0(t) - \mu_1(t) \right]^2$$
   Determina el umbral óptimo global $t^*$ de binarización sin requerir ajuste manual de hiperparámetros.

---

#### C.2 Demostración Matemática: Convolución Espacial 3x3 Paso a Paso vs. OpenCV
*(Generado de forma reproducible por `src/math_proof.py`)*

Se toma un parche representativo de **3x3 píxeles** ubicado en la frontera crítica entre el trazo oscuro de un carácter (intensidades $\sim 40-55$) y el fondo reflectivo de la placa vehicular (intensidades $\sim 205-215$). Se aplica el operador diferencial **Sobel Horizontal ($G_x$)** para extracción de gradientes verticales.

##### 1. Matriz del Parche de Imagen $P$ (3x3 píxeles en escala de grises):
```text
|   45.0    50.0   210.0 |
|   40.0    55.0   215.0 |
|   42.0    48.0   205.0 |
```

##### 2. Matriz del Kernel Espacial $K$ (Sobel $G_x$ 3x3):
```text
|  -1.00    0.00    1.00 |
|  -2.00    0.00    2.00 |
|  -1.00    0.00    1.00 |
```

##### 3. Operación Aritmética en el Píxel Central $(1, 1)$:
$$R(1, 1) = \sum_{i=0}^{2} \sum_{j=0}^{2} P(i, j) \cdot K(i, j)$$

| Coordenada $(i, j)$ | Intensidad Píxel $P(i, j)$ | Peso Kernel $K(i, j)$ | Producto Parcial $P \cdot K$ |
| :---: | :---: | :---: | :---: |
| `(0, 0)` | `45.0` | `-1.00` | **`-45.00`** |
| `(0, 1)` | `50.0` | ` 0.00` | **` +0.00`** |
| `(0, 2)` | `210.0` | `+1.00` | **`+210.00`** |
| `(1, 0)` | `40.0` | `-2.00` | **`-80.00`** |
| `(1, 1)` | `55.0` | ` 0.00` | **` +0.00`** |
| `(1, 2)` | `215.0` | `+2.00` | **`+430.00`** |
| `(2, 0)` | `42.0` | `-1.00` | **`-42.00`** |
| `(2, 1)` | `48.0` | ` 0.00` | **` +0.00`** |
| `(2, 2)` | `205.0` | `+1.00` | **`+205.00`** |

##### 4. Resultados de Paridad Numérica:
- **Suma Aritmética Manual (Python Puro):**
  $$R(1, 1) = -45.0 + 0 + 210.0 - 80.0 + 0 + 430.0 - 42.0 + 0 + 205.0 = \mathbf{678.0000}$$
- **Respuesta `cv2.filter2D` (OpenCV con CV_64F):**
  $$\mathbf{678.0000}$$
- **Diferencia Numérica Absoluta:**
  $$|\text{Python} - \text{OpenCV}| = \mathbf{0.0 \times 10^{0}}$$
- **Resultado de Paridad:** ✅ **EXACTO (ERROR CERO)**.
