/**
 * Port fiel de PlateVoterBuffer (src/validator.py): ventana deslizante N=7,
 * mayoría k=4, enfriamiento de 15 s y confianza mínima de 0.35.
 * La lógica vive en BufferVotacion sin dependencias del DOM para poder probarla.
 */

import { traducir } from "./textos.js";
import { validarPlaca } from "./validador.js";

const PARECIDOS = {
  O: "0", I: "1", B: "8", S: "5", Z: "2", G: "6", A: "4",
  "0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A"
};

/**
 * Buffer temporal de votación multi-cuadro con ventana de enfriamiento.
 */
export class BufferVotacion {
  constructor({
    ventana = 7,
    minVotos = 4,
    esperaMs = 15000,
    minConfianza = 0.35
  } = {}) {
    this.ventana = ventana;
    this.minVotos = minVotos;
    this.esperaMs = esperaMs;
    this.minConfianza = minConfianza;

    this.buffer = [];
    this.emisiones = new Map();
    this.ultimoSentido = new Map();
    this.registros = [];
    this.idAcceso = 0;
  }

  /**
   * Procesa una lectura individual de cuadro y emite consenso si alcanza k votos.
   */
  agregarLectura(textoBruto, confianza, ahoraMs = Date.now()) {
    const validacion = validarPlaca(textoBruto);

    if (!validacion.valida) {
      return {
        aceptada: false,
        motivoDescarte: "invalida",
        consenso: false,
        enEspera: false,
        validacion,
        ...this._resumenLider()
      };
    }

    if (typeof confianza !== "number" || !Number.isFinite(confianza) || confianza < this.minConfianza) {
      return {
        aceptada: false,
        motivoDescarte: "baja_confianza",
        consenso: false,
        enEspera: false,
        validacion,
        ...this._resumenLider()
      };
    }

    this.buffer.push({
      placa: validacion.placa,
      tipo: validacion.tipo,
      conf: confianza,
      momento: ahoraMs
    });
    if (this.buffer.length > this.ventana) {
      this.buffer.shift();
    }

    const resumen = this._resumenLider();
    const { lider, votosLider } = resumen;

    if (lider && votosLider >= this.minVotos) {
      const previa = this.emisiones.get(lider);
      if (previa !== undefined && ahoraMs - previa < this.esperaMs) {
        const restanteSeg = Math.ceil((this.esperaMs - (ahoraMs - previa)) / 1000);
        return {
          aceptada: true,
          consenso: false,
          enEspera: true,
          restanteSeg,
          validacion,
          ...resumen
        };
      }

      this.emisiones.set(lider, ahoraMs);
      const sentido = this.ultimoSentido.get(lider) === "ENTRADA" ? "SALIDA" : "ENTRADA";
      this.ultimoSentido.set(lider, sentido);

      const apoyos = this.buffer.filter((o) => o.placa === lider);
      const confMedia = apoyos.reduce((s, o) => s + o.conf, 0) / apoyos.length;

      this.idAcceso += 1;
      const acceso = {
        id: this.idAcceso,
        placa: lider,
        tipo: apoyos[apoyos.length - 1].tipo,
        sentido,
        momento: ahoraMs,
        conf: confMedia,
        votos: votosLider
      };
      this.registros.unshift(acceso);

      return {
        aceptada: true,
        consenso: true,
        enEspera: false,
        acceso,
        validacion,
        ...resumen
      };
    }

    return {
      aceptada: true,
      consenso: false,
      enEspera: false,
      validacion,
      ...resumen
    };
  }

  _resumenLider() {
    const conteo = {};
    for (const item of this.buffer) {
      conteo[item.placa] = (conteo[item.placa] || 0) + 1;
    }
    let lider = null;
    let votosLider = 0;
    for (const placa of Object.keys(conteo)) {
      if (conteo[placa] > votosLider) {
        lider = placa;
        votosLider = conteo[placa];
      }
    }
    return { conteo, lider, votosLider };
  }

  obtenerEstado() {
    return {
      buffer: this.buffer.map((o) => ({ ...o })),
      registros: this.registros.map((r) => ({ ...r })),
      ...this._resumenLider()
    };
  }

  limpiar() {
    this.buffer = [];
    this.emisiones.clear();
    this.ultimoSentido.clear();
    this.registros = [];
    this.idAcceso = 0;
  }
}

/**
 * Simula lecturas con ruido OCR ocasional sobre una placa base.
 */
export function generarLecturaSimulada(placaBase, azar = Math.random()) {
  if (azar < 0.5) return placaBase;
  if (azar < 0.85) {
    const pos = [];
    const arr = placaBase.split("");
    arr.forEach((c, i) => {
      if (PARECIDOS[c]) pos.push(i);
    });
    if (!pos.length) return placaBase;
    const k = pos[Math.floor(Math.random() * pos.length)];
    arr[k] = PARECIDOS[arr[k]];
    return arr.join("");
  }
  return Math.random() < 0.5
    ? placaBase.slice(0, 5)
    : `${placaBase.slice(0, 3)}X${placaBase.slice(4)}`;
}

function formatearHora(ms) {
  return new Date(ms).toTimeString().slice(0, 8);
}

/**
 * Conecta el simulador de votación multi-cuadro con el DOM.
 */
export function montarVotacion({ obtenerIdioma, obtenerPlacaActual }) {
  const votador = new BufferVotacion();
  const slotsEl = document.getElementById("slots");
  const contadorEl = document.getElementById("contador");
  const ultimoEl = document.getElementById("ultimo");
  const accesosEl = document.getElementById("accesos");
  const botonLeer = document.getElementById("leer");

  let cuadro = 0;
  let ultimoRegistro = null;

  function pintarSlots(estado, idioma) {
    slotsEl.replaceChildren();
    for (let i = 0; i < votador.ventana; i += 1) {
      const item = estado.buffer[i];
      const div = document.createElement("div");
      const gana = item && item.placa === estado.lider && estado.votosLider >= votador.minVotos;
      div.className = "slot" + (item ? " lleno" : "") + (gana ? " gana" : "");
      div.textContent = item ? item.placa : traducir(idioma, "votacion.slot_vacio");
      slotsEl.appendChild(div);
    }

    contadorEl.textContent = estado.lider
      ? traducir(idioma, "votacion.contador", {
          placa: estado.lider,
          votos: estado.votosLider,
          k: votador.minVotos
        })
      : traducir(idioma, "votacion.contador_vacio", { k: votador.minVotos });
  }

  function pintarUltimo(idioma) {
    ultimoEl.replaceChildren();
    if (!ultimoRegistro) return;

    const { numCuadro, bruto, conf, resultado } = ultimoRegistro;
    ultimoEl.append(document.createTextNode(traducir(idioma, "votacion.cuadro_prefijo", { n: numCuadro })));
    const bBruto = document.createElement("b");
    bBruto.textContent = bruto;
    ultimoEl.append(bBruto, document.createTextNode(` (${conf.toFixed(2)})`));

    if (!resultado.aceptada) {
      const clave =
        resultado.motivoDescarte === "invalida"
          ? "votacion.descarte_invalida"
          : "votacion.descarte_confianza";
      ultimoEl.append(document.createTextNode(traducir(idioma, clave)));
      return;
    }

    if (resultado.validacion.placa !== bruto) {
      ultimoEl.append(document.createTextNode(traducir(idioma, "votacion.corregido_a")));
      const bCorregido = document.createElement("b");
      bCorregido.textContent = resultado.validacion.placa;
      ultimoEl.append(bCorregido);
    } else {
      ultimoEl.append(document.createTextNode(traducir(idioma, "votacion.valido")));
    }

    if (resultado.enEspera) {
      ultimoEl.append(
        document.createTextNode(
          traducir(idioma, "votacion.en_espera", {
            s: resultado.restanteSeg,
            placa: resultado.lider
          })
        )
      );
    } else if (resultado.consenso && resultado.acceso) {
      const sentidoTexto = traducir(idioma, `votacion.sentido_${resultado.acceso.sentido}`);
      const msg = traducir(idioma, "votacion.consenso", {
        votos: resultado.acceso.votos,
        sentido: sentidoTexto
      });
      const partes = msg.split(sentidoTexto);
      ultimoEl.append(document.createTextNode(partes[0]));
      const bSentido = document.createElement("b");
      bSentido.textContent = sentidoTexto;
      ultimoEl.append(bSentido, document.createTextNode(partes.slice(1).join(sentidoTexto)));
    }
  }

  function pintarAccesos(estado, idioma) {
    accesosEl.replaceChildren();
    if (!estado.registros.length) {
      const tr = document.createElement("tr");
      tr.className = "vacio";
      const td = document.createElement("td");
      td.colSpan = 5;
      td.textContent = traducir(idioma, "votacion.vacio");
      tr.appendChild(td);
      accesosEl.appendChild(tr);
      return;
    }

    for (const fila of estado.registros.slice(0, 6)) {
      const tr = document.createElement("tr");

      const tdId = document.createElement("td");
      tdId.textContent = String(fila.id);

      const tdPlaca = document.createElement("td");
      tdPlaca.textContent = fila.placa;

      const tdSentido = document.createElement("td");
      if (fila.sentido === "ENTRADA") tdSentido.className = "si";
      tdSentido.textContent = traducir(idioma, `votacion.sentido_${fila.sentido}`);

      const tdMomento = document.createElement("td");
      tdMomento.textContent = formatearHora(fila.momento);

      const tdConf = document.createElement("td");
      tdConf.textContent = fila.conf.toFixed(2);

      tr.append(tdId, tdPlaca, tdSentido, tdMomento, tdConf);
      accesosEl.appendChild(tr);
    }
  }

  function repintar() {
    const idioma = obtenerIdioma();
    const estado = votador.obtenerEstado();
    pintarSlots(estado, idioma);
    pintarUltimo(idioma);
    pintarAccesos(estado, idioma);
  }

  function leer(azar, conf) {
    cuadro += 1;
    const base = obtenerPlacaActual();
    const bruto = generarLecturaSimulada(base, azar);
    const confianza = conf !== undefined ? conf : 0.3 + Math.random() * 0.65;
    const resultado = votador.agregarLectura(bruto, confianza, Date.now());
    ultimoRegistro = { numCuadro: cuadro, bruto, conf: confianza, resultado };
    repintar();
  }

  botonLeer.addEventListener("click", () => leer());

  // Estado inicial con tres cuadros previos al consenso
  leer(0.1, 0.82);
  leer(0.6, 0.74);
  leer(0.1, 0.88);

  return { repintar };
}
