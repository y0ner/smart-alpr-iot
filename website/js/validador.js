/**
 * Port fiel de PlateValidator (src/validator.py) para placas colombianas
 * (AAA000 carro y AAA00A moto) con separación estricta entre lógica y DOM.
 */

import { traducir } from "./textos.js";

const D2L = { "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A" };
const L2D = { O: "0", Q: "0", D: "0", I: "1", L: "1", Z: "2", S: "5", B: "8", G: "6", A: "4" };
const P5 = { O: "0", Q: "0", I: "1", L: "1" };

const esDigito = (c) => /^[0-9]$/.test(c);
const esLetra = (c) => /^[A-Z]$/.test(c);

/**
 * Conserva solo caracteres alfanuméricos en mayúsculas.
 */
export function limpiarTextoPlaca(texto) {
  return String(texto ?? "").toUpperCase().replace(/[^A-Z0-9]/g, "");
}

/**
 * Corrige ambigüedades morfológicas por posición y valida el formato colombiano.
 */
export function validarPlaca(texto) {
  const limpio = limpiarTextoPlaca(texto);
  const res = {
    limpio,
    celdas: [],
    valida: false,
    tipo: "DESCONOCIDO",
    placa: limpio,
    correcciones: 0,
    errorClave: null,
    errorPos: null,
    errorChar: null
  };

  if (limpio.length !== 6) {
    res.errorClave = "longitud";
    return res;
  }

  const ch = limpio.split("");
  res.celdas = ch.map((c) => ({ antes: "", car: c, error: false }));

  for (let i = 0; i < 3; i += 1) {
    if (esDigito(ch[i]) && D2L[ch[i]]) {
      res.celdas[i].antes = ch[i];
      ch[i] = D2L[ch[i]];
      res.celdas[i].car = ch[i];
    } else if (!esLetra(ch[i])) {
      res.celdas[i].error = true;
      res.errorClave = "letra";
      res.errorPos = i + 1;
      res.errorChar = ch[i];
      return res;
    }
  }

  for (let j = 3; j < 5; j += 1) {
    if (esLetra(ch[j]) && L2D[ch[j]]) {
      res.celdas[j].antes = ch[j];
      ch[j] = L2D[ch[j]];
      res.celdas[j].car = ch[j];
    } else if (!esDigito(ch[j])) {
      res.celdas[j].error = true;
      res.errorClave = "digito";
      res.errorPos = j + 1;
      res.errorChar = ch[j];
      return res;
    }
  }

  const c5 = ch[5];
  if (P5[c5]) {
    res.celdas[5].antes = c5;
    ch[5] = P5[c5];
    res.celdas[5].car = ch[5];
    res.tipo = "CARRO";
  } else if (esDigito(c5)) {
    res.tipo = "CARRO";
  } else {
    res.tipo = "MOTO";
  }

  res.valida = true;
  res.placa = ch.join("");
  res.correcciones = res.celdas.filter((c) => c.antes !== "").length;
  return res;
}

/**
 * Conecta el validador interactivo con el DOM.
 */
export function montarValidador({ obtenerIdioma }) {
  const entrada = document.getElementById("lectura");
  const celdasEl = document.getElementById("celdas");
  const pill = document.getElementById("pill");
  const motivo = document.getElementById("motivo");
  const mini = document.getElementById("mini-placa");
  const mpLetras = document.getElementById("mp-l");
  const mpNumeros = document.getElementById("mp-n");

  function rolesPorIdioma(idioma) {
    const rLetra = traducir(idioma, "validador.rol_letra");
    const rDig = traducir(idioma, "validador.rol_digito");
    const rTipo = traducir(idioma, "validador.rol_tipo");
    return [rLetra, rLetra, rLetra, rDig, rDig, rTipo];
  }

  function repintar() {
    const idioma = obtenerIdioma();
    const r = validarPlaca(entrada.value);
    const roles = rolesPorIdioma(idioma);

    celdasEl.replaceChildren();
    for (let i = 0; i < 6; i += 1) {
      const c = r.celdas[i];
      const div = document.createElement("div");
      div.className = "celda" + (!c ? " vacia" : c.error ? " error" : c.antes ? " cambio" : "");

      const sRol = document.createElement("span");
      sRol.className = "rol";
      sRol.textContent = roles[i];

      const sCar = document.createElement("span");
      sCar.className = "car";
      sCar.textContent = c ? c.car : "·";

      const sAntes = document.createElement("span");
      sAntes.className = "antes";
      sAntes.textContent = c && c.antes ? traducir(idioma, "validador.era", { c: c.antes }) : "";

      div.append(sRol, sCar, sAntes);
      celdasEl.appendChild(div);
    }

    if (r.valida) {
      pill.className = "pill ok";
      const tipoLabel = traducir(idioma, `validador.tipo_${r.tipo}`);
      pill.textContent = traducir(idioma, "validador.pill_valida", { tipo: tipoLabel });

      if (r.correcciones === 0) {
        motivo.textContent = traducir(idioma, "validador.sin_correcciones");
      } else if (r.correcciones === 1) {
        motivo.textContent = traducir(idioma, "validador.corregido_uno");
      } else {
        motivo.textContent = traducir(idioma, "validador.corregido_varios", { n: r.correcciones });
      }

      mini.classList.remove("apagada");
      mpLetras.textContent = r.placa.slice(0, 3);
      mpNumeros.textContent = r.placa.slice(3);
      mini.setAttribute("aria-label", traducir(idioma, "validador.placa_valida_aria", { placa: r.placa }));
    } else {
      pill.className = "pill mal";
      pill.textContent = traducir(idioma, "validador.pill_invalida");
      if (r.errorClave === "longitud") {
        motivo.textContent = traducir(idioma, "validador.err_longitud", { n: r.limpio.length });
      } else if (r.errorClave === "letra") {
        motivo.textContent = traducir(idioma, "validador.err_letra", { pos: r.errorPos, c: r.errorChar });
      } else {
        motivo.textContent = traducir(idioma, "validador.err_digito", { pos: r.errorPos, c: r.errorChar });
      }
      mini.classList.add("apagada");
      mini.setAttribute("aria-label", traducir(idioma, "validador.placa_invalida_aria"));
    }
    return r;
  }

  entrada.addEventListener("input", repintar);
  document.querySelectorAll(".presets button").forEach((btn) => {
    btn.addEventListener("click", () => {
      entrada.value = btn.getAttribute("data-v") || "";
      repintar();
      entrada.focus();
    });
  });

  repintar();

  return {
    obtenerPlacaActual: () => {
      const r = validarPlaca(entrada.value);
      return r.valida ? r.placa : "UJR456";
    },
    repintar
  };
}
