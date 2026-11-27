# Brief W1 · Sitio web estático (ES/EN) con laboratorio OpenCV.js

- **Rama:** `feat/sitio-web` · **Worktree:** `~/Desktop/Projects/worktrees/smart-alpr-feat-sitio-web`
- **Lee primero:** `AGENTS.md` (reglas obligatorias).
- **Objetivo:** convertir la propuesta de diseño aprobada en el sitio de presentación del proyecto (carpeta `website/`, sin paso de compilación, listo para Vercel), en español e inglés, con un laboratorio que corre el preprocesamiento de OpenCV en el navegador.

## Contexto (por qué)

- **Diseño aprobado por Yoner (no lo rediseñes):** `docs/diseno/propuesta-aprobada.html`. Ábrelo y respeta su estructura, tokens de color (claro por defecto, oscuro = paleta de la consola), tipografías (Barlow Condensed, Inter, JetBrains Mono), portada con la placa gigante y la caja de detección, y los componentes (etapas, validador, votación, cifras, fallas, cómo correr). Sus imágenes apuntan a `website/img/`.
- Las cuatro decisiones de la propuesta quedaron aprobadas: **portada con la placa**, **antes y después público**, **laboratorio con OpenCV.js** y **español e inglés con selector**.
- **Lo que no va al sitio:** la franja superior "PROPUESTA DE DISEÑO v1" (`.nota-propuesta`) y la sección "Notas de la propuesta" (`.notas`).
- **Datos que mantiene el jefe (no los edites):** `website/js/datos.js` (resultados medidos por foto, etapas "antes" y "después"; `despues` vale `null` mientras no haya medición) y todo `website/img/`.
- **OpenCV.js ya está en el repo:** `website/vendor/opencv-4.10.0.js` (10 MB, Apache 2.0; ver `website/vendor/LEEME.md`). Revisa el final del archivo para saber cómo expone `cv` y espera a que esté listo (`cv.onRuntimeInitialized`, o `await cv` si es una promesa) antes de usarlo.
- El validador y la votación del diseño (JS dentro de `propuesta-aprobada.html`) son un port verificado de `src/validator.py`; consérvalos tal cual al pasarlos a módulos. Casos esperados (entrada → válida, placa, tipo):
  `KLM9O1→sí,KLM901,CARRO` · `8OG12A→sí,BOG12A,MOTO` · `AUH43O→sí,AUH430,CARRO` · `WGL45I→sí,WGL451,CARRO` · `TKI-O33→sí,TKI033,CARRO` · `9HSOUE444→no` · `ABC123→sí,ABC123,CARRO` · `XYZ18D→sí,XYZ18D,MOTO` · `0I2S6B→sí,OIZ56B,MOTO` · `ABCDEF→no` · `AB1234→sí,ABI234,CARRO` · `ABC12Q→sí,ABC120,CARRO` · `3BC123→no` · `ABCX23→no` · `abc 12 3→sí,ABC123,CARRO` · `""→no`.
- Datos del pie: autor Yoner Isaza; docente **Eduardo Sierra Fragozo** (según `docs/Proyecto_final_Vision_Artificial.pdf`); licencia **MIT**; fotos de muestra descargadas de internet, **fuente pendiente de citar** (dilo así, sin inventar fuentes).

## Archivos permitidos

Todo dentro de `website/`, **excepto** `website/js/datos.js`, `website/img/**` y `website/vendor/**` (solo lectura):

- `website/index.html` (**nuevo**)
- `website/css/estilos.css` (**nuevo**)
- `website/js/*.js` (**nuevos**, módulos ES; p. ej. `principal.js`, `tema.js`, `idioma.js`, `textos.js`, `validador.js`, `votacion.js`, `resultados.js`, `laboratorio.js`)
- `website/tests/*.test.mjs` (**nuevos**)
- `website/favicon.svg`, `website/vercel.json`, `website/.gitignore`, `website/404.html` (**nuevos**)

## Qué hacer

1. **Estructura.** `index.html` con el contenido de la propuesta (sin la franja ni las notas), CSS en `css/estilos.css`, JS en módulos ES cargados con `<script type="module" src="js/principal.js">`. Nada inline salvo lo imprescindible.
2. **Idiomas.** Todo texto visible, `alt`, `aria-label` y `<title>`/`description` sale de un diccionario `{ es: {...}, en: {...} }` (`js/textos.js`) mediante atributos `data-i18n` / `data-i18n-attr`. Botón `ES · EN` en la barra de navegación; recuerda la elección en `localStorage` (con `try/catch`); la primera vez usa `navigator.language` (español si empieza por `es`, si no inglés); actualiza `<html lang>`. El inglés debe ser natural, no literal.
3. **Antes y después.** En "Así viaja una imagen", un selector `Antes · Después` cambia la tira de etapas. Lee las rutas, parámetros y textos de `datos.js` (`etapas.antes`, `etapas.despues`). Si `etapas.despues` es `null`, el selector queda deshabilitado con el texto "Después: pendiente de medición". El aviso ámbar de las letras huecas solo aparece en "Antes".
4. **Resultados.** Las cifras y la tabla salen de `datos.js` (`medicion.antes`, `medicion.despues`). Columna "después": si `medicion.despues` es `null`, "pendiente". Marca aciertos y errores con texto y color, **sin** símbolos ✓/✗ ni emojis (usa "bien" / "mal" o el equivalente en inglés).
5. **Laboratorio OpenCV.js** (sección nueva después de "Pruébalo", enlazada en la barra):
   - botón "Cargar OpenCV.js (10 MB)" que inserta `vendor/opencv-4.10.0.js` solo al pulsarlo, con estado de carga y error visibles;
   - elegir una foto de `img/muestras/` (la lista está en `datos.js`, `muestras`) o subir una propia (`<input type="file" accept="image/*">`), con el aviso "tu foto no sale de tu navegador";
   - seleccionar la placa arrastrando un rectángulo sobre la foto (eventos *pointer*, funciona con el dedo);
   - controles: altura normalizada (60-300 px, por defecto 140), filtro bilateral (d, σ), umbral (Otsu o adaptativo, con bloque impar 3-99 y C), morfología on/off;
   - muestra cada etapa en su `<canvas>`: recorte, grises, bilateral, umbral y morfología, con el parámetro usado; y un texto que explique que con un bloque pequeño frente a la altura del recorte las letras quedan huecas;
   - libera cada `cv.Mat` con `.delete()` (sin fugas al mover los controles);
   - aclara que el OCR (EasyOCR) no corre en el navegador.
6. **Tema y accesibilidad.** Conserva el selector de tema (sistema/oscuro/claro). `prefers-reduced-motion` apaga la animación de la caja de detección. Foco visible, contraste AA, `alt` en todas las imágenes, `width`/`height` en `<img>` para no saltar el diseño, y nada de desbordamiento horizontal a 360 px de ancho.
7. **Metadatos.** `<title>`, `description`, `og:title`, `og:description`, `og:type`, `og:image` = `img/og.png` (la pone el jefe), `twitter:card`, `theme-color`, y `favicon.svg` (una placa amarilla simple). Omite `canonical`: el dominio aún no existe.
8. **Vercel.** `vercel.json` con `cleanUrls: true`, caché larga e inmutable para `/vendor/(.*)` e `/img/(.*)`, y cabeceras `X-Content-Type-Options: nosniff` y `Referrer-Policy: strict-origin-when-cross-origin`. `.gitignore` con `.vercel`. `404.html` sencillo con el mismo estilo.
9. **Pruebas** (`node --test website/tests/`):
   - `validador.test.mjs`: los 16 casos de arriba;
   - `votacion.test.mjs`: 4 de 7 lecturas iguales registran ENTRADA; dentro de 15 s no se repite; la siguiente es SALIDA; lecturas con confianza < 0.35 o inválidas no votan (la lógica de votación debe estar separada del DOM para poder probarla);
   - `textos.test.mjs`: cada clave `data-i18n` de `index.html` existe en `es` y en `en`, y ambos diccionarios tienen las mismas claves.

## Criterios de aceptación

- [ ] `node --test website/tests/` en verde.
- [ ] `grep -n "nota-propuesta\|Notas de la propuesta\|por confirmar\|por definir" website/index.html` no devuelve nada.
- [ ] `grep -rnP "✓|✗|[\x{1F300}-\x{1FAFF}]" website/index.html website/js website/css` no devuelve nada.
- [ ] `git diff --stat` no toca `website/js/datos.js`, `website/img/` ni `website/vendor/`.
- [ ] Ninguna petición a terceros salvo Google Fonts (`grep -rn "https://" website/js website/index.html` solo muestra Google Fonts y el enlace a GitHub).

## Qué se probará en el entorno real (lo hace el humano)

`cd website && python3 -m http.server 8000`, abrir `http://localhost:8000` en el portátil y en el celular: cambiar idioma y tema, probar el validador y la votación, cargar OpenCV.js, elegir una muestra, marcar la placa y mover el bloque del umbral hasta ver las letras huecas.
