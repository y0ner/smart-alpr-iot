/**
 * Selector de tema (sistema -> oscuro -> claro) con respaldo en localStorage.
 */

import { traducir } from "./textos.js";

const CLAVE_TEMA = "alpr-tema";
const ESTADOS = ["sistema", "oscuro", "claro"];

function leerTemaGuardado() {
  try {
    const g = globalThis.localStorage?.getItem(CLAVE_TEMA);
    if (g) {
      const idx = ESTADOS.indexOf(g);
      return idx >= 0 ? idx : 0;
    }
  } catch {
    // Sin almacenamiento disponible
  }
  return 0;
}

function guardarTema(valor) {
  try {
    globalThis.localStorage?.setItem(CLAVE_TEMA, valor);
  } catch {
    // Sin almacenamiento disponible
  }
}

/**
 * Inicializa y enlaza el botón de cambio de tema.
 */
export function iniciarTema(obtenerIdioma) {
  const raiz = document.documentElement;
  const boton = document.getElementById("tema");
  const valorEl = document.getElementById("tema-valor");
  let actual = leerTemaGuardado();

  function actualizarEtiqueta() {
    if (!valorEl) return;
    const estado = ESTADOS[actual];
    const idioma = typeof obtenerIdioma === "function" ? obtenerIdioma() : "es";
    valorEl.setAttribute("data-i18n", `tema.${estado}`);
    valorEl.textContent = traducir(idioma, `tema.${estado}`);
  }

  function aplicar() {
    const t = ESTADOS[actual];
    if (t === "sistema") {
      raiz.removeAttribute("data-theme");
    } else {
      raiz.setAttribute("data-theme", t === "oscuro" ? "dark" : "light");
    }
    actualizarEtiqueta();
    guardarTema(t);
  }

  if (boton) {
    boton.addEventListener("click", () => {
      actual = (actual + 1) % ESTADOS.length;
      aplicar();
    });
  }

  aplicar();

  return { actualizarEtiqueta };
}
