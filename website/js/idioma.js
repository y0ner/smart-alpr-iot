/**
 * Gestión de idioma (ES / EN) con persistencia segura en localStorage,
 * detección inicial por navigator.language y traducción de atributos DOM.
 */

import { IDIOMAS, TEXTOS } from "./textos.js";

const CLAVE_IDIOMA = "alpr-idioma";

function leerAlmacen(clave) {
  try {
    return globalThis.localStorage?.getItem(clave) ?? null;
  } catch {
    return null;
  }
}

function guardarAlmacen(clave, valor) {
  try {
    globalThis.localStorage?.setItem(clave, valor);
  } catch {
    // Sin almacenamiento persistente disponible
  }
}

/**
 * Normaliza un código BCP-47 a "es" o "en".
 */
export function normalizarIdioma(codigo) {
  if (typeof codigo === "string" && codigo.toLowerCase().startsWith("es")) {
    return "es";
  }
  return "en";
}

/**
 * Determina el idioma inicial usando preferencia guardada o navigator.language.
 */
export function detectarIdiomaInicial() {
  const guardado = leerAlmacen(CLAVE_IDIOMA);
  if (guardado && IDIOMAS.includes(guardado)) {
    return guardado;
  }
  const navegador =
    globalThis.navigator?.language ||
    (Array.isArray(globalThis.navigator?.languages) ? globalThis.navigator.languages[0] : "es");
  return normalizarIdioma(navegador);
}

/**
 * Aplica traducciones a [data-i18n], [data-i18n-attr] y <html lang>.
 */
export function aplicarTextosDOM(doc, idioma) {
  const dict = TEXTOS[idioma] || TEXTOS.es;
  doc.documentElement.lang = idioma;

  doc.querySelectorAll("[data-i18n]").forEach((el) => {
    const clave = el.getAttribute("data-i18n");
    if (clave && Object.hasOwn(dict, clave)) {
      el.textContent = dict[clave];
    }
  });

  doc.querySelectorAll("[data-i18n-attr]").forEach((el) => {
    const spec = el.getAttribute("data-i18n-attr");
    if (!spec) return;
    for (const par of spec.split(",")) {
      const idx = par.indexOf(":");
      if (idx === -1) continue;
      const attr = par.slice(0, idx).trim();
      const clave = par.slice(idx + 1).trim();
      if (attr && clave && Object.hasOwn(dict, clave)) {
        el.setAttribute(attr, dict[clave]);
      }
    }
  });
}

/**
 * Inicializa el selector ES · EN en la barra de navegación.
 */
export function iniciarIdioma(alCambiar) {
  let idiomaActual = detectarIdiomaInicial();
  const boton = document.getElementById("idioma");

  function sincronizarBoton() {
    if (!boton) return;
    boton.querySelectorAll(".lang-opt").forEach((span) => {
      const activo = span.getAttribute("data-lang") === idiomaActual;
      span.classList.toggle("activo", activo);
      if (activo) {
        span.setAttribute("aria-current", "true");
      } else {
        span.removeAttribute("aria-current");
      }
    });
  }

  function aplicar(nuevoIdioma, persistir = true) {
    idiomaActual = IDIOMAS.includes(nuevoIdioma) ? nuevoIdioma : "es";
    aplicarTextosDOM(document, idiomaActual);
    sincronizarBoton();
    if (persistir) {
      guardarAlmacen(CLAVE_IDIOMA, idiomaActual);
    }
    if (typeof alCambiar === "function") {
      alCambiar(idiomaActual);
    }
  }

  if (boton) {
    boton.addEventListener("click", () => {
      aplicar(idiomaActual === "es" ? "en" : "es", true);
    });
  }

  aplicar(idiomaActual, false);

  return {
    obtenerIdioma: () => idiomaActual,
    establecerIdioma: aplicar
  };
}
