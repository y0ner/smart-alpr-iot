# ESTÁNDAR DE DISEÑO UI/UX INDUSTRIAL Y ANTI-AI SLOP
## Sistema de Misión Crítica / Telemetría Vial (Smart ALPR IoT)

Este documento define las reglas de diseño visual e interactivo obligatorias para la interfaz de usuario de **Smart ALPR IoT**. Toda vista, componente, gráfico o elemento tipográfico debe apegarse estrictamente a estos principios.

---

### 1. PROHIBICIONES ESTRICTAS (ANTI-AI SLOP)
- ❌ **Cero gradientes de fantasía**: Prohibidos los degradados púrpuras, violetas, cian o neón típicos de plantillas genéricas generadas por IA.
- ❌ **Cero sombras difusas exageradas (soft blurry glow)**: Prohibidos `box-shadow` hipertrofiados o brillos de neón flotantes.
- ❌ **Cero abuso de emojis infantiles**: No usar emojis en títulos de métricas, botones de acción, tablas de datos ni encabezados técnicos.
- ❌ **Cero bordes excesivamente redondeados**: Prohibido `border-radius` estilo "burbuja" o "píldora" en tarjetas y tablas. Los radios deben ser sobrios: `4px` a `8px` máximo.
- ❌ **Cero layouts con aire vacío no justificado**: Prohibido el espaciado disperso que obligue a scrollear sin aportar densidad de información.

---

### 2. ESTÉTICA VISUAL: TELEMETRÍA INDUSTRIAL / SCADA MODERNO
Inspirado en consolas de misión crítica como **Linear**, **Vercel Dashboard**, **Palantir Foundry** y paneles SCADA de infraestructura de transporte:

#### Paleta de Color Técnico (Dark Mode Operativo)
- **Fondo Base (Background / Canvas):** `#090A0F`
- **Superficie de Tarjetas y Paneles (Card / Surface):** `#11131A`
- **Superficie Secundaria / Hover:** `#181B26`
- **Bordes y Divisores Técnicos:** `1px solid #1E2230`
- **Borde de Enfoque / Activo:** `#2E364F`
- **Texto Primario (Alta legibilidad):** `#F3F4F6`
- **Texto Secundario / Metadatos:** `#9CA3AF`
- **Texto Terciario / Deshabilitado:** `#6B7280`

#### Semántica de Acentos Funcionales (Exclusivamente de Estado)
- **Autorizado / Entrada Normal / Pipeline OK:** Verde Esmeralda `#10B981` (badge fondo: `rgba(16, 185, 129, 0.12)`, borde: `rgba(16, 185, 129, 0.3)`)
- **Alerta / Revisión Manual / Buffer Pendiente:** Ámbar Técnico `#F59E0B` (badge fondo: `rgba(245, 158, 11, 0.12)`, borde: `rgba(245, 158, 11, 0.3)`)
- **Denegado / Discrepancia / Error de Inferencia:** Rojo Crítico `#EF4444` (badge fondo: `rgba(239, 68, 68, 0.12)`, borde: `rgba(239, 68, 68, 0.3)`)
- **Acento Neutral / Selección / Bounding Box:** Azul Cobalto Técnico `#3B82F6` o Blanco Frío `#E5E7EB`

---

### 3. TIPOGRAFÍA Y JERARQUÍA
- **UI General (Encabezados, labels, controles):** `Inter`, `Geist` o sans-serif de precisión suiza (`font-weight`: 400 regular, 500 medium, 600 semi-bold).
- **Datos Técnicos y Telemetría:** `JetBrains Mono`, `Roboto Mono` o `monospace` estricto:
  - Matrículas vehiculares (`ABC123`, `XYZ18D`).
  - Timestamps ISO (`2026-11-09 21:20:00`).
  - Latencias y FPS (`18.4 ms`, `32.1 FPS`).
  - Coordenadas de Bounding Boxes (`[x: 142, y: 380, w: 220, h: 90]`).
  - IDs de base de datos y hashes de transacción.

---

### 4. COMPONENTES Y DENSIDAD DE INFORMACIÓN
1. **Visor de Cámara / Stream:**
   - Overlays monocromáticos o con contraste nítido sin ocluir la matrícula.
   - Bounding boxes con grosor de 2px máximo, esquinas nítidas y etiquetas en JetBrains Mono sobre fondo sólido `#11131A`.
2. **Inspector de Visión Artificial (Tira Morfológica 5 Fases):**
   - Presentación horizontal compacta de las etapas intermedias de procesamiento de la placa:
     `Original 3x` ➔ `Grises` ➔ `Filtro Bilateral` ➔ `Umbral Otsu/Adaptativo` ➔ `Morfología Final`.
   - Cada miniatura con etiqueta técnica de dimensiones y método aplicado.
3. **Placa Vehicular Física Colombiana:**
   - Renderizado realista de la placa colombiana: fondo amarillo `#FACC15` con caracteres negros en relieve para servicio particular, o fondo blanco `#F3F4F6` para servicio público.
   - Tipografía fiel a placas de tránsito y distinción visual clara entre Carro (`AAA000`) y Moto (`AAA00A`).
4. **Tablas Transaccionales de Base de Datos:**
   - Filas densas con padding vertical compacto (`8px - 10px`).
   - Resaltado discreto al pasar el cursor.
   - Miniaturas de captura en formato 16:9 o 3:1 nítidas y zoomable on click/hover.
