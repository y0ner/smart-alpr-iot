/**
 * Laboratorio OpenCV.js en el navegador:
 * - carga vendor/opencv-4.10.0.js bajo demanda;
 * - permite elegir muestra o subir imagen local sin salir del navegador;
 * - selección de ROI mediante pointer events (mouse y pantalla táctil);
 * - ejecuta recorte normalizado, grises, bilateral, umbral y morfología
 *   liberando cada cv.Mat con .delete().
 */

import { muestras } from "./datos.js";
import { traducir } from "./textos.js";

let promesaCv = null;

function limpiarThenable(cv) {
  if (cv && typeof cv.then === "function") {
    try {
      delete cv.then;
    } catch {
      // Si no es configurable, el consumidor usará globalThis.cv directamente
    }
  }
}

/**
 * Carga e inicializa OpenCV.js solo al pulsar el botón del laboratorio.
 */
function cargarOpenCV() {
  if (globalThis.cv && globalThis.cv.Mat && globalThis.cv.calledRun) {
    limpiarThenable(globalThis.cv);
    return Promise.resolve(globalThis.cv);
  }
  if (promesaCv) return promesaCv;

  promesaCv = new Promise((resolve, reject) => {
    let runtimeListo = false;
    let scriptCargado = false;
    let terminado = false;

    function intentarResolver() {
      if (terminado || !runtimeListo || !scriptCargado) return;
      const cv = globalThis.cv;
      if (!cv || !cv.Mat) {
        terminado = true;
        promesaCv = null;
        reject(new Error("cv.Mat no disponible"));
        return;
      }
      limpiarThenable(cv);
      terminado = true;
      resolve(cv);
    }

    globalThis.Module = {
      onRuntimeInitialized() {
        runtimeListo = true;
        intentarResolver();
      },
      onAbort(detalle) {
        if (terminado) return;
        terminado = true;
        promesaCv = null;
        reject(new Error(String(detalle || "abortado")));
      }
    };

    const previo = document.querySelector('script[data-opencv-vendor="1"]');
    if (previo) previo.remove();

    const script = document.createElement("script");
    script.src = "vendor/opencv-4.10.0.js";
    script.async = true;
    script.dataset.opencvVendor = "1";

    script.onload = () => {
      scriptCargado = true;
      if (globalThis.cv && globalThis.cv.calledRun && globalThis.cv.Mat) {
        runtimeListo = true;
      }
      intentarResolver();
    };

    script.onerror = () => {
      if (terminado) return;
      terminado = true;
      promesaCv = null;
      reject(new Error("HTTP / script"));
    };

    document.head.appendChild(script);
  });

  return promesaCv;
}

function ajustarBloqueImpar(valor) {
  let n = Math.max(3, Math.min(99, Math.round(Number(valor) || 43)));
  if (n % 2 === 0) n += n >= 99 ? -1 : 1;
  return n;
}

/**
 * Conecta los controles del laboratorio OpenCV.js con el DOM.
 */
export function montarLaboratorio({ obtenerIdioma }) {
  const btnCargar = document.getElementById("lab-cargar-cv");
  const estadoEl = document.getElementById("lab-estado-cv");
  const selectMuestra = document.getElementById("lab-muestra");
  const inputArchivo = document.getElementById("lab-archivo");
  const visor = document.getElementById("lab-visor");
  const btnRoiReset = document.getElementById("lab-roi-reset");
  const roiTextoEl = document.getElementById("lab-roi-texto");

  const ctrlAltura = document.getElementById("ctrl-altura");
  const valAltura = document.getElementById("val-altura");
  const ctrlD = document.getElementById("ctrl-d");
  const valD = document.getElementById("val-d");
  const ctrlSigma = document.getElementById("ctrl-sigma");
  const valSigma = document.getElementById("val-sigma");
  const ctrlUmbralModo = document.getElementById("ctrl-umbral-modo");
  const ctrlBloque = document.getElementById("ctrl-bloque");
  const valBloque = document.getElementById("val-bloque");
  const ctrlC = document.getElementById("ctrl-c");
  const valC = document.getElementById("val-c");
  const ctrlMorfologia = document.getElementById("ctrl-morfologia");
  const relacionEl = document.getElementById("lab-relacion");

  const canvasRecorte = document.getElementById("cv-recorte");
  const canvasGrises = document.getElementById("cv-grises");
  const canvasBilateral = document.getElementById("cv-bilateral");
  const canvasUmbral = document.getElementById("cv-umbral");
  const canvasMorfologia = document.getElementById("cv-morfologia");

  const paramRecorte = document.getElementById("param-recorte");
  const paramGrises = document.getElementById("param-grises");
  const paramBilateral = document.getElementById("param-bilateral");
  const paramUmbral = document.getElementById("param-umbral");
  const paramMorfologia = document.getElementById("param-morfologia");

  const fuenteCanvas = document.createElement("canvas");
  const fuenteCtx = fuenteCanvas.getContext("2d", { willReadFrequently: true });
  const visorCtx = visor.getContext("2d");

  let cvInstancia = null;
  let estadoCarga = "sin_cargar"; // sin_cargar | cargando | listo | error
  let detalleError = "";
  let errorImagen = false;
  let ultimoResultadoParams = null;

  let imagenCargada = false;
  let blobUrlActual = null;
  let roi = { x: 0, y: 0, w: 0, h: 0 };
  let arrastre = null;
  let rafId = 0;

  function rellenarSelectorMuestras() {
    selectMuestra.replaceChildren();
    for (const m of muestras) {
      const opt = document.createElement("option");
      opt.value = m.img;
      opt.textContent = `${m.placa} · ${m.archivo} (${m.ancho} × ${m.alto})`;
      selectMuestra.appendChild(opt);
    }
    const popayan = muestras.find((m) => m.archivo.includes("EMY923"));
    if (popayan) {
      selectMuestra.value = popayan.img;
    }
  }

  function actualizarEstadoCargaUI(idioma) {
    if (errorImagen && estadoCarga !== "error") {
      btnCargar.disabled = estadoCarga === "cargando" || estadoCarga === "listo";
      btnCargar.textContent = traducir(
        idioma,
        estadoCarga === "listo"
          ? "cvlab.btn_listo"
          : estadoCarga === "cargando"
            ? "cvlab.btn_cargando"
            : "cvlab.btn_cargar"
      );
      estadoEl.className = "cv-estado mal";
      estadoEl.textContent = traducir(idioma, "cvlab.err_imagen");
      return;
    }
    if (estadoCarga === "sin_cargar") {
      btnCargar.disabled = false;
      btnCargar.textContent = traducir(idioma, "cvlab.btn_cargar");
      estadoEl.className = "cv-estado";
      estadoEl.textContent = traducir(idioma, "cvlab.estado_sin_cargar");
    } else if (estadoCarga === "cargando") {
      btnCargar.disabled = true;
      btnCargar.textContent = traducir(idioma, "cvlab.btn_cargando");
      estadoEl.className = "cv-estado";
      estadoEl.textContent = traducir(idioma, "cvlab.estado_cargando");
    } else if (estadoCarga === "listo") {
      btnCargar.disabled = true;
      btnCargar.textContent = traducir(idioma, "cvlab.btn_listo");
      estadoEl.className = "cv-estado ok";
      estadoEl.textContent = traducir(idioma, "cvlab.estado_listo");
    } else {
      btnCargar.disabled = false;
      btnCargar.textContent = traducir(idioma, "cvlab.btn_cargar");
      estadoEl.className = "cv-estado mal";
      estadoEl.textContent = traducir(idioma, "cvlab.estado_error", { detalle: detalleError });
    }
  }

  function leerControles() {
    const altura = Math.max(60, Math.min(300, Number(ctrlAltura.value) || 140));
    const d = Math.max(3, Math.min(19, Number(ctrlD.value) || 9));
    const sigma = Math.max(5, Math.min(75, Number(ctrlSigma.value) || 30));
    const umbralModo = ctrlUmbralModo.value === "otsu" ? "otsu" : "adaptativo";
    const bloque = ajustarBloqueImpar(ctrlBloque.value);
    const cVal = Number(ctrlC.value) || 0;
    const morfologia = Boolean(ctrlMorfologia.checked);
    return { altura, d, sigma, umbralModo, bloque, cVal, morfologia };
  }

  function actualizarEtiquetasControles(idioma) {
    const p = leerControles();
    valAltura.textContent = `${p.altura} px`;
    valD.textContent = String(p.d);
    valSigma.textContent = String(p.sigma);
    valBloque.textContent = `${p.bloque} px`;
    valC.textContent = String(p.cVal);

    const esOtsu = p.umbralModo === "otsu";
    ctrlBloque.disabled = esOtsu;
    ctrlC.disabled = esOtsu;

    relacionEl.textContent = esOtsu
      ? traducir(idioma, "cvlab.relacion_otsu")
      : traducir(idioma, "cvlab.relacion_adapt", {
          bloque: p.bloque,
          altura: p.altura,
          ratio: (p.bloque / p.altura).toFixed(2)
        });
  }

  function dibujarVisor() {
    if (!imagenCargada) return;
    const w = fuenteCanvas.width;
    const h = fuenteCanvas.height;
    if (visor.width !== w || visor.height !== h) {
      visor.width = w;
      visor.height = h;
    }

    visorCtx.clearRect(0, 0, w, h);
    visorCtx.drawImage(fuenteCanvas, 0, 0);

    // Atenúa suavemente el exterior del recorte para resaltar la selección
    visorCtx.save();
    visorCtx.fillStyle = "rgba(9, 10, 15, 0.42)";
    visorCtx.fillRect(0, 0, w, roi.y);
    visorCtx.fillRect(0, roi.y, roi.x, roi.h);
    visorCtx.fillRect(roi.x + roi.w, roi.y, Math.max(0, w - (roi.x + roi.w)), roi.h);
    visorCtx.fillRect(0, roi.y + roi.h, w, Math.max(0, h - (roi.y + roi.h)));

    const grosor = Math.max(2, Math.round(Math.max(w, h) / 280));
    visorCtx.strokeStyle = "#10B981";
    visorCtx.lineWidth = grosor;
    visorCtx.strokeRect(roi.x + grosor / 2, roi.y + grosor / 2, Math.max(1, roi.w - grosor), Math.max(1, roi.h - grosor));
    visorCtx.restore();

    roiTextoEl.textContent = traducir(obtenerIdioma(), "cvlab.roi_formato", {
      x: roi.x,
      y: roi.y,
      w: roi.w,
      h: roi.h
    });
  }

  function actualizarEtiquetasSalidas(idioma) {
    if (!ultimoResultadoParams) {
      const pend = traducir(idioma, "cvlab.param_pendiente");
      paramRecorte.textContent = pend;
      paramGrises.textContent = pend;
      paramBilateral.textContent = pend;
      paramUmbral.textContent = pend;
      paramMorfologia.textContent = pend;
      return;
    }

    const r = ultimoResultadoParams;
    const interp = traducir(idioma, r.agranda ? "cvlab.interp_cubic" : "cvlab.interp_area");
    paramRecorte.textContent = traducir(idioma, "cvlab.param_recorte", { w: r.w, h: r.h, interp });
    paramGrises.textContent = traducir(idioma, "cvlab.param_grises");
    paramBilateral.textContent = traducir(idioma, "cvlab.param_bilateral", { d: r.d, sigma: r.sigma });
    paramUmbral.textContent =
      r.umbralModo === "otsu"
        ? traducir(idioma, "cvlab.param_umbral_otsu")
        : traducir(idioma, "cvlab.param_umbral_adapt", { b: r.bloque, c: r.cVal });
    paramMorfologia.textContent = traducir(
      idioma,
      r.morfologia ? "cvlab.param_morf_on" : "cvlab.param_morf_off"
    );
  }

  function procesarConOpenCV() {
    if (!cvInstancia || !imagenCargada || roi.w < 4 || roi.h < 4) return;

    const cv = cvInstancia;
    const p = leerControles();
    const mats = [];
    const rastrear = (mat) => {
      mats.push(mat);
      return mat;
    };

    try {
      const src = rastrear(cv.imread(fuenteCanvas));
      if (src.cols < 2 || src.rows < 2) return;
      const rx = Math.max(0, Math.min(src.cols - 2, roi.x));
      const ry = Math.max(0, Math.min(src.rows - 2, roi.y));
      const rw = Math.max(2, Math.min(src.cols - rx, roi.w));
      const rh = Math.max(2, Math.min(src.rows - ry, roi.h));

      const recorte = rastrear(src.roi(new cv.Rect(rx, ry, rw, rh)));
      const newH = p.altura;
      const newW = Math.max(1, Math.round((rw * newH) / rh));
      const agranda = newH > rh;
      const interp = agranda ? cv.INTER_CUBIC : cv.INTER_AREA;

      const normalizado = rastrear(new cv.Mat());
      cv.resize(recorte, normalizado, new cv.Size(newW, newH), 0, 0, interp);
      cv.imshow(canvasRecorte, normalizado);

      const grises = rastrear(new cv.Mat());
      cv.cvtColor(normalizado, grises, cv.COLOR_RGBA2GRAY);
      cv.imshow(canvasGrises, grises);

      const bilateral = rastrear(new cv.Mat());
      cv.bilateralFilter(grises, bilateral, p.d, p.sigma, p.sigma, cv.BORDER_DEFAULT);
      cv.imshow(canvasBilateral, bilateral);

      const umbral = rastrear(new cv.Mat());
      if (p.umbralModo === "otsu") {
        cv.threshold(bilateral, umbral, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
      } else {
        cv.adaptiveThreshold(
          bilateral,
          umbral,
          255,
          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY,
          p.bloque,
          p.cVal
        );
      }
      cv.imshow(canvasUmbral, umbral);

      if (p.morfologia) {
        const filasBorde = Math.min(3, umbral.rows);
        const tiraSuperior = rastrear(umbral.roi(new cv.Rect(0, 0, umbral.cols, filasBorde)));
        const esFondoBlanco = cv.mean(tiraSuperior)[0] > 127;

        const trabajo = rastrear(new cv.Mat());
        if (esFondoBlanco) {
          cv.bitwise_not(umbral, trabajo);
        } else {
          umbral.copyTo(trabajo);
        }

        const kApertura = rastrear(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3)));
        const kCierre = rastrear(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 2)));
        const abierta = rastrear(new cv.Mat());
        const cerrada = rastrear(new cv.Mat());
        cv.morphologyEx(trabajo, abierta, cv.MORPH_OPEN, kApertura);
        cv.morphologyEx(abierta, cerrada, cv.MORPH_CLOSE, kCierre);

        const morfFinal = rastrear(new cv.Mat());
        if (esFondoBlanco) {
          cv.bitwise_not(cerrada, morfFinal);
        } else {
          cerrada.copyTo(morfFinal);
        }
        cv.imshow(canvasMorfologia, morfFinal);
      } else {
        cv.imshow(canvasMorfologia, umbral);
      }

      ultimoResultadoParams = {
        w: newW,
        h: newH,
        agranda,
        d: p.d,
        sigma: p.sigma,
        umbralModo: p.umbralModo,
        bloque: p.bloque,
        cVal: p.cVal,
        morfologia: p.morfologia
      };
      actualizarEtiquetasSalidas(obtenerIdioma());
    } finally {
      for (const m of mats) {
        if (m && typeof m.delete === "function" && (!m.isDeleted || !m.isDeleted())) {
          m.delete();
        }
      }
    }
  }

  function programarProceso() {
    actualizarEtiquetasControles(obtenerIdioma());
    if (!cvInstancia) return;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      procesarConOpenCV();
    });
  }

  function establecerRoiPorDefecto(w, h) {
    if (w < 280) {
      roi = { x: 0, y: 0, w, h };
    } else {
      roi = {
        x: Math.round(w * 0.22),
        y: Math.round(h * 0.34),
        w: Math.round(w * 0.56),
        h: Math.round(h * 0.34)
      };
    }
  }

  function cargarImagenDesdeUrl(src, revocarBlobDespues = false) {
    const img = new Image();
    img.onload = () => {
      errorImagen = false;
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      fuenteCanvas.width = w;
      fuenteCanvas.height = h;
      fuenteCtx.clearRect(0, 0, w, h);
      fuenteCtx.drawImage(img, 0, 0, w, h);
      imagenCargada = true;
      establecerRoiPorDefecto(w, h);
      dibujarVisor();
      actualizarEstadoCargaUI(obtenerIdioma());
      programarProceso();
      if (revocarBlobDespues && blobUrlActual === src) {
        URL.revokeObjectURL(src);
        blobUrlActual = null;
      }
    };
    img.onerror = () => {
      if (revocarBlobDespues && blobUrlActual === src) {
        URL.revokeObjectURL(src);
        blobUrlActual = null;
      }
      errorImagen = true;
      actualizarEstadoCargaUI(obtenerIdioma());
    };
    img.src = src;
  }

  function coordsEnImagen(evento) {
    const rect = visor.getBoundingClientRect();
    const escX = fuenteCanvas.width / Math.max(1, rect.width);
    const escY = fuenteCanvas.height / Math.max(1, rect.height);
    const x = Math.max(0, Math.min(fuenteCanvas.width, Math.round((evento.clientX - rect.left) * escX)));
    const y = Math.max(0, Math.min(fuenteCanvas.height, Math.round((evento.clientY - rect.top) * escY)));
    return { x, y };
  }

  visor.addEventListener("pointerdown", (e) => {
    if (!imagenCargada) return;
    visor.setPointerCapture(e.pointerId);
    const p = coordsEnImagen(e);
    arrastre = { id: e.pointerId, x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  });

  visor.addEventListener("pointermove", (e) => {
    if (!arrastre || arrastre.id !== e.pointerId) return;
    const p = coordsEnImagen(e);
    arrastre.x1 = p.x;
    arrastre.y1 = p.y;
    const x = Math.min(arrastre.x0, arrastre.x1);
    const y = Math.min(arrastre.y0, arrastre.y1);
    const w = Math.abs(arrastre.x1 - arrastre.x0);
    const h = Math.abs(arrastre.y1 - arrastre.y0);
    if (w >= 8 && h >= 8) {
      roi = { x, y, w, h };
      dibujarVisor();
    }
  });

  function finalizarArrastre(e) {
    if (!arrastre || arrastre.id !== e.pointerId) return;
    const w = Math.abs(arrastre.x1 - arrastre.x0);
    const h = Math.abs(arrastre.y1 - arrastre.y0);
    if (w >= 8 && h >= 8) {
      roi = {
        x: Math.min(arrastre.x0, arrastre.x1),
        y: Math.min(arrastre.y0, arrastre.y1),
        w,
        h
      };
      dibujarVisor();
      programarProceso();
    }
    arrastre = null;
  }

  visor.addEventListener("pointerup", finalizarArrastre);
  visor.addEventListener("pointercancel", () => {
    arrastre = null;
  });

  btnRoiReset.addEventListener("click", () => {
    if (!imagenCargada) return;
    roi = { x: 0, y: 0, w: fuenteCanvas.width, h: fuenteCanvas.height };
    dibujarVisor();
    programarProceso();
  });

  selectMuestra.addEventListener("change", () => {
    if (blobUrlActual) {
      URL.revokeObjectURL(blobUrlActual);
      blobUrlActual = null;
    }
    inputArchivo.value = "";
    cargarImagenDesdeUrl(selectMuestra.value, false);
  });

  inputArchivo.addEventListener("change", () => {
    const archivo = inputArchivo.files?.[0];
    if (!archivo) return;
    if (blobUrlActual) {
      URL.revokeObjectURL(blobUrlActual);
    }
    blobUrlActual = URL.createObjectURL(archivo);
    cargarImagenDesdeUrl(blobUrlActual, true);
  });

  [ctrlAltura, ctrlD, ctrlSigma, ctrlBloque, ctrlC].forEach((el) => {
    el.addEventListener("input", programarProceso);
  });
  [ctrlUmbralModo, ctrlMorfologia].forEach((el) => {
    el.addEventListener("change", programarProceso);
  });

  btnCargar.addEventListener("click", async () => {
    if (estadoCarga === "cargando" || estadoCarga === "listo") return;
    estadoCarga = "cargando";
    detalleError = "";
    actualizarEstadoCargaUI(obtenerIdioma());

    try {
      cvInstancia = await cargarOpenCV();
      estadoCarga = "listo";
      actualizarEstadoCargaUI(obtenerIdioma());
      procesarConOpenCV();
    } catch (err) {
      estadoCarga = "error";
      detalleError = err instanceof Error ? err.message : String(err);
      actualizarEstadoCargaUI(obtenerIdioma());
    }
  });

  function repintar() {
    const idioma = obtenerIdioma();
    actualizarEstadoCargaUI(idioma);
    actualizarEtiquetasControles(idioma);
    actualizarEtiquetasSalidas(idioma);
    if (imagenCargada) {
      dibujarVisor();
    }
  }

  rellenarSelectorMuestras();
  repintar();
  if (selectMuestra.value) {
    cargarImagenDesdeUrl(selectMuestra.value, false);
  }

  return { repintar };
}
