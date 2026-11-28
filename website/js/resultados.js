/**
 * Renderiza la tira de etapas (Antes / Después) y las cifras/tabla de resultados
 * directamente desde website/js/datos.js sin cifras hardcodeadas.
 */

import { etapas, medicion } from "./datos.js";
import { traducir } from "./textos.js";

function crearCeldaResultado(valor, real, idioma) {
  const td = document.createElement("td");
  if (valor === null || valor === undefined) {
    td.className = "pend";
    td.textContent = traducir(idioma, "resultados.sin_lectura");
    return td;
  }
  const acierto = valor === real;
  td.className = acierto ? "si" : "no";
  const marca = traducir(idioma, acierto ? "resultados.marca_bien" : "resultados.marca_mal");
  td.textContent = `${valor} · ${marca}`;
  return td;
}

function crearTarjetaCifra(etiqueta, valorPrincipal, sufijo, descripcion, claseValor = "") {
  const div = document.createElement("div");
  div.className = "cifra";

  const spanEtiq = document.createElement("span");
  spanEtiq.className = "etiqueta";
  spanEtiq.textContent = etiqueta;

  const spanValor = document.createElement("span");
  spanValor.className = claseValor ? `valor ${claseValor}` : "valor";
  spanValor.append(document.createTextNode(String(valorPrincipal)));
  if (sufijo) {
    const small = document.createElement("small");
    small.textContent = sufijo;
    spanValor.append(small);
  }

  const p = document.createElement("p");
  p.textContent = descripcion;

  div.append(spanEtiq, spanValor, p);
  return div;
}

/**
 * Conecta las secciones de etapas y resultados con datos.js y el idioma activo.
 */
export function montarResultados({ obtenerIdioma }) {
  const gridEtapas = document.getElementById("etapas-grid");
  const btnAntes = document.getElementById("btn-etapas-antes");
  const btnDespues = document.getElementById("btn-etapas-despues");
  const avisoHuecas = document.getElementById("aviso-letras-huecas");
  const cifrasEl = document.getElementById("cifras-res");
  const tablaEl = document.getElementById("tabla-res");
  const pieEl = document.getElementById("pie-resultados");

  let modoEtapas = "antes";
  const tieneDespues = Array.isArray(etapas.despues) && etapas.despues.length > 0;

  function pintarEtapas(idioma) {
    if (!tieneDespues && modoEtapas === "despues") {
      modoEtapas = "antes";
    }

    btnAntes.classList.toggle("activo", modoEtapas === "antes");
    btnAntes.setAttribute("aria-pressed", String(modoEtapas === "antes"));
    btnAntes.textContent = traducir(idioma, "etapas.btn_antes");

    const claveBtnDespues = tieneDespues ? "etapas.btn_despues" : "etapas.btn_despues_pend";
    btnDespues.disabled = !tieneDespues;
    btnDespues.classList.toggle("activo", modoEtapas === "despues");
    btnDespues.setAttribute("aria-pressed", String(modoEtapas === "despues"));
    btnDespues.setAttribute("data-i18n", claveBtnDespues);
    btnDespues.textContent = traducir(idioma, claveBtnDespues);

    if (avisoHuecas) {
      avisoHuecas.hidden = modoEtapas !== "antes";
    }

    const lista = etapas[modoEtapas] || [];
    gridEtapas.replaceChildren();

    lista.forEach((etapa, indice) => {
      const figure = document.createElement("figure");
      figure.className = "etapa";

      const cajaImg = document.createElement("div");
      cajaImg.className = etapa.clave === "captura" ? "etapa-img escena" : "etapa-img";

      const img = document.createElement("img");
      img.src = etapa.img;
      img.alt =
        etapa.alt?.[idioma] ??
        etapa.alt?.es ??
        traducir(idioma, `etapas.item.${etapa.clave}.alt`);
      img.width = etapa.ancho;
      img.height = etapa.alto;
      cajaImg.appendChild(img);

      const cap = document.createElement("figcaption");

      const num = document.createElement("span");
      num.className = "etapa-num";
      num.textContent = String(indice + 1).padStart(2, "0");

      const h3 = document.createElement("h3");
      h3.textContent =
        etapa.titulo?.[idioma] ??
        etapa.titulo?.es ??
        traducir(idioma, `etapas.item.${etapa.clave}.titulo`);

      const param = document.createElement("span");
      param.className = "param";
      param.textContent = etapa.param?.[idioma] ?? etapa.param?.es ?? "";

      const que = document.createElement("span");
      que.className = "que";
      que.textContent =
        etapa.que?.[idioma] ??
        etapa.que?.es ??
        traducir(idioma, `etapas.item.${etapa.clave}.que`);

      cap.append(num, h3, param, que);
      figure.append(cajaImg, cap);
      gridEtapas.appendChild(figure);
    });
  }

  function pintarResultados(idioma) {
    const base = medicion.antes;
    const despues = medicion.despues;

    // Las tarjetas muestran la medición más reciente; si hay "después", citan el "antes".
    const actual = despues || base;
    const conAntes = (texto, valorAntes) =>
      despues ? `${texto} ${traducir(idioma, "resultados.cifra_antes", { v: valorAntes })}` : texto;

    cifrasEl.replaceChildren();
    const clasePre = actual.aciertosConPre === 0 ? "mal-t" : "ok-t";
    cifrasEl.appendChild(
      crearTarjetaCifra(
        traducir(idioma, "resultados.cifra_pre_t"),
        actual.aciertosConPre,
        ` / ${actual.total}`,
        conAntes(traducir(idioma, "resultados.cifra_pre_d"), `${base.aciertosConPre} / ${base.total}`),
        clasePre
      )
    );
    cifrasEl.appendChild(
      crearTarjetaCifra(
        traducir(idioma, "resultados.cifra_sin_t"),
        actual.aciertosSinPre,
        ` / ${actual.total}`,
        conAntes(traducir(idioma, "resultados.cifra_sin_d"), `${base.aciertosSinPre} / ${base.total}`)
      )
    );
    cifrasEl.appendChild(
      crearTarjetaCifra(
        traducir(idioma, "resultados.cifra_tiempo_t"),
        `${actual.segundosMin}–${actual.segundosMax}`,
        " s",
        conAntes(traducir(idioma, "resultados.cifra_tiempo_d"), `${base.segundosMin}–${base.segundosMax} s`)
      )
    );

    const esRespaldo = actual.detector === "respaldo";
    cifrasEl.appendChild(
      crearTarjetaCifra(
        traducir(idioma, "resultados.cifra_det_t"),
        traducir(idioma, esRespaldo ? "resultados.cifra_det_sobel" : "resultados.cifra_det_yolo"),
        "",
        traducir(idioma, esRespaldo ? "resultados.cifra_det_d_sobel" : "resultados.cifra_det_d_yolo")
      )
    );

    const mapaDespues = new Map();
    if (despues && Array.isArray(despues.filas)) {
      for (const f of despues.filas) {
        mapaDespues.set(f.archivo, f);
      }
    }

    tablaEl.replaceChildren();
    for (const fila of base.filas) {
      const tr = document.createElement("tr");

      const tdArchivo = document.createElement("td");
      tdArchivo.textContent = fila.archivo;

      const tdReal = document.createElement("td");
      tdReal.textContent = fila.placa;

      const tdSin = crearCeldaResultado(fila.sinPre, fila.placa, idioma);
      const tdCon = crearCeldaResultado(fila.conPre, fila.placa, idioma);

      let tdDespues;
      if (!despues) {
        tdDespues = document.createElement("td");
        tdDespues.className = "pend";
        tdDespues.textContent = traducir(idioma, "resultados.pendiente");
      } else {
        const filaD = mapaDespues.get(fila.archivo);
        tdDespues = crearCeldaResultado(filaD ? filaD.conPre : null, fila.placa, idioma);
      }

      tr.append(tdArchivo, tdReal, tdSin, tdCon, tdDespues);
      tablaEl.appendChild(tr);
    }

    pieEl.textContent = despues
      ? traducir(idioma, "resultados.pie_despues", { fecha: base.fecha, fechaD: despues.fecha, total: base.total })
      : traducir(idioma, "resultados.pie", { fecha: base.fecha, total: base.total });

    const subInicio = document.querySelector('#resultados [data-i18n="resultados.sub_inicio"]');
    if (subInicio) {
      subInicio.textContent = traducir(idioma, "resultados.sub_inicio", { n: base.total });
    }
  }

  function repintar() {
    const idioma = obtenerIdioma();
    pintarEtapas(idioma);
    pintarResultados(idioma);
  }

  btnAntes.addEventListener("click", () => {
    modoEtapas = "antes";
    pintarEtapas(obtenerIdioma());
  });

  btnDespues.addEventListener("click", () => {
    if (!tieneDespues) return;
    modoEtapas = "despues";
    pintarEtapas(obtenerIdioma());
  });

  repintar();

  return { repintar };
}
