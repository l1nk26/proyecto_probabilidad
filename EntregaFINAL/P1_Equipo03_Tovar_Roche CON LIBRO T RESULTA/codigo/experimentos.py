"""Fases 2, 3, 4, 5, 6 y 7 del proyecto S01 (A: 2/4/6, B: 3/5/7).

Este archivo comparte la utilidad ``calcular_alpha_por_n``: B la usa para
dibujar F2 y A la usa para evaluar en prueba sin depender de que B haya
terminado. Las figuras se guardan en PDF (entregable) y en PNG (para el
borrador Markdown del reporte).

Protocolo de la evaluación en prueba:
  * ``evaluar_prueba_una_vez`` escribe ``resultados/metricas_prueba.csv``.
  * Si el archivo ya existe, NO vuelve a evaluar prueba; use --forzar-prueba
    solo si quiere regenerar deliberadamente todo el bloque de prueba.
  * Fase 4 y Fase 6 leen ESE archivo. Ninguna otra función evalúa prueba.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Fecha fija en los metadatos de los PDF: hace que F2/F3/F4/F5 sean
# byte-idénticos al repetir el pipeline (UTC 2026-11-23 00:00:00).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1795392000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos import ModeloNgramas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "datos" / "corpus"
RESULTADOS = ROOT / "resultados"

N_VALORES = (1, 2, 3, 4, 5)
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TAMANO_BLOQUE = 2000
SEMILLA_BOOTSTRAP = 20261122
N_REMUESTRAS = 10_000
DELTA_PERTURBACION = 0.01
SEMILLA_SIMULACION = 20261123
LONGITUD_MUESTRA = 300
TOL_ESTACIONARIA = 1e-10
MAX_ITER_ESTACIONARIA = 100_000
NUM_FUSIONES_BPE = 200
SEPARADOR_TOKENS = "\x00"


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------

def cargar_datos():
    """Devuelve (train, val, test, alfabeto) con la partición de preparar.py."""
    vocab = json.loads((RESULTADOS / "vocabulario.json").read_text(encoding="utf-8"))
    alfabeto = tuple(vocab["V"])
    train = (CORPUS / "train.txt").read_text(encoding="utf-8")
    val = (CORPUS / "val.txt").read_text(encoding="utf-8")
    test = (CORPUS / "test.txt").read_text(encoding="utf-8")
    return train, val, test, alfabeto


def prefijo_de(texto: str, n: int) -> str:
    """Últimos n-1 caracteres de una partición, para dar contexto a la siguiente."""
    k = n - 1
    return texto[-k:] if k else ""


def guardar_figura(fig, nombre: str) -> None:
    """Guarda la figura en PDF (entregable) y PNG (para el Markdown del reporte)."""
    fig.savefig(RESULTADOS / f"{nombre}.pdf")
    fig.savefig(RESULTADOS / f"{nombre}.png", dpi=150)


def bpc_validacion_por_alpha(train, val, alfabeto) -> dict[int, dict[float, float]]:
    """Matriz BPC(n, alpha) calculada únicamente con el bloque de validación."""
    matriz: dict[int, dict[float, float]] = {}
    for n in N_VALORES:
        prefijo = prefijo_de(train, n)
        matriz[n] = {}
        for alpha in ALPHAS:
            modelo = ModeloNgramas(train, n, alfabeto, alpha)
            matriz[n][alpha] = modelo.bpc(val, prefijo)
    return matriz


def mejor_alpha_de(matriz_n: dict[float, float]) -> float:
    """Argmin de BPC; en empate gana el alpha más pequeño (orden de ALPHAS)."""
    return min(ALPHAS, key=lambda a: (matriz_n[a], a))


def calcular_alpha_por_n(train: str, val: str, alfabeto) -> dict[int, float]:
    """Selecciona, SOLO con validación, el alpha que minimiza BPC para cada n."""
    matriz = bpc_validacion_por_alpha(train, val, alfabeto)
    return {n: mejor_alpha_de(matriz[n]) for n in N_VALORES}


def asegurar_alpha_por_n(train: str, val: str, alfabeto, elegidos=None) -> dict[int, float]:
    """Lee el alpha ya elegido o lo calcula una vez con validación.

    El archivo es intermedio: no se entrega en el zip, se regenera al ejecutar
    ``experimentos.py``. Nunca usa el bloque de prueba.
    """
    ruta = RESULTADOS / "alpha_por_n.json"
    if ruta.exists():
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in datos.items()}
    if elegidos is None:
        elegidos = calcular_alpha_por_n(train, val, alfabeto)
    ruta.write_text(
        json.dumps({str(n): a for n, a in elegidos.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return elegidos


def bpc_validacion_por_n(train, val, alfabeto, alpha_por_n) -> dict[int, float]:
    return {
        n: ModeloNgramas(train, n, alfabeto, alpha_por_n[n]).bpc(val, prefijo_de(train, n))
        for n in N_VALORES
    }


def mejor_n_por_validacion(bpc_val: dict[int, float]) -> int:
    return min(N_VALORES, key=lambda n: bpc_val[n])


def leer_metricas_prueba() -> list[dict]:
    ruta = RESULTADOS / "metricas_prueba.csv"
    if not ruta.exists():
        raise SystemExit("Falta resultados/metricas_prueba.csv; ejecuta primero la Fase 4")
    filas = []
    with ruta.open(newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            filas.append({
                "n": int(fila["n"]),
                "alpha": float(fila["alpha"]),
                "bloque": int(fila["bloque"]),
                "caracteres": int(fila["caracteres"]),
                "bpc": float(fila["bpc"]),
            })
    return filas


# ---------------------------------------------------------------------------
# Fase 2 — MLE y ceros
# ---------------------------------------------------------------------------

def _log2_verosimilitud_mle(modelo: ModeloNgramas) -> float:
    """Suma c(h,y) * log2(c(h,y)/c(h)) sobre todos los pares observados."""
    total = 0.0
    for contexto, c_h in modelo.contextos.items():
        for simbolo, c_hy in modelo.pares[contexto].items():
            total += c_hy * math.log2(c_hy / c_h)
    return total


def _elegir_perturbaciones(modelo: ModeloNgramas):
    """Contexto frecuente, símbolo origen y dos destinos distintos."""
    contexto = max(modelo.contextos, key=modelo.contextos.get)
    c_h = modelo.contextos[contexto]
    conteos = modelo.pares[contexto]
    orden = sorted(conteos.items(), key=lambda kv: -kv[1])
    origen, c_origen = orden[0]
    q_origen = c_origen / c_h
    if q_origen <= DELTA_PERTURBACION + 1e-12:
        raise RuntimeError("El contexto frecuente no permite mover 0.01 del símbolo principal")

    candidatos = [y for y, _ in orden if y != origen]
    candidatos += [y for y in modelo.V if y != origen and y not in candidatos]
    destinos = candidatos[:2]
    if len(destinos) < 2:
        raise RuntimeError("Se necesitan dos símbolos destino distintos")
    return contexto, origen, destinos, q_origen


def _log2_verosimilitud_perturbada(modelo, contexto, origen, destino, q_origen):
    c_h = modelo.contextos[contexto]
    c_origen = modelo.pares[contexto][origen]
    c_destino = modelo.pares[contexto].get(destino, 0)
    q_destino = c_destino / c_h
    base = _log2_verosimilitud_mle(modelo)
    return (base
            + c_origen * math.log2((q_origen - DELTA_PERTURBACION) / q_origen)
            + c_destino * math.log2((q_destino + DELTA_PERTURBACION) / q_destino))


def fase2_mle() -> None:
    """T2: contextos, sumas, ceros en validación y log-verosimilitud perturbada."""
    train, val, _test, alfabeto = cargar_datos()
    columnas = [
        "n", "contextos_observados", "pares_observados", "max_error_sumas",
        "caracteres_validacion", "ceros_validacion", "fraccion_ceros_validacion",
        "log2_verosimilitud_train", "log2_verosimilitud_pert1",
        "log2_verosimilitud_pert2", "delta", "contexto_perturbado",
        "simbolo_origen", "destino_1", "destino_2",
    ]
    filas = []
    for n in N_VALORES:
        modelo = ModeloNgramas(train, n, alfabeto, alpha=0.0)

        max_error = max(
            (abs(modelo.suma_condicionales(h) - 1.0) for h in modelo.contextos),
            default=0.0,
        )
        if max_error > 1e-12:
            raise AssertionError(f"n={n}: las condicionales no suman 1 ({max_error})")

        total_val = 0
        ceros = 0
        for contexto, simbolo in modelo.secuencias(val, prefijo_de(train, n)):
            total_val += 1
            if modelo.prob(contexto, simbolo) == 0.0:
                ceros += 1

        contexto, origen, destinos, q_origen = _elegir_perturbaciones(modelo)
        ll0 = _log2_verosimilitud_mle(modelo)
        ll1 = _log2_verosimilitud_perturbada(modelo, contexto, origen, destinos[0], q_origen)
        ll2 = _log2_verosimilitud_perturbada(modelo, contexto, origen, destinos[1], q_origen)
        if ll0 < ll1 - 1e-9 or ll0 < ll2 - 1e-9:
            raise AssertionError(f"n={n}: una perturbación supera a la MLE")

        filas.append({
            "n": n,
            "contextos_observados": len(modelo.contextos),
            "pares_observados": sum(len(c) for c in modelo.pares.values()),
            "max_error_sumas": f"{max_error:.3e}",
            "caracteres_validacion": total_val,
            "ceros_validacion": ceros,
            "fraccion_ceros_validacion": f"{ceros / total_val:.6f}" if total_val else "0.000000",
            "log2_verosimilitud_train": f"{ll0:.4f}",
            "log2_verosimilitud_pert1": f"{ll1:.4f}",
            "log2_verosimilitud_pert2": f"{ll2:.4f}",
            "delta": DELTA_PERTURBACION,
            "contexto_perturbado": contexto.replace("#", "\\#"),
            "simbolo_origen": origen,
            "destino_1": destinos[0],
            "destino_2": destinos[1],
        })

    with (RESULTADOS / "T2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)

    print("Fase 2 lista: resultados/T2.csv")
    for fila in filas:
        print(f"  n={fila['n']}: contextos={fila['contextos_observados']}, "
              f"ceros en validación={fila['fraccion_ceros_validacion']}")


# ---------------------------------------------------------------------------
# Fase 4 — evaluación única en prueba, Gibbs y figuras
# ---------------------------------------------------------------------------

def evaluar_prueba_una_vez(best_n=None, alpha_por_n=None, forzar: bool = False) -> None:
    """Calcula y cachea TODAS las métricas de prueba.

    Escribe metricas_prueba.csv (BPC por bloque de 2000) y F4.pdf. Si el CSV
    ya existe, no vuelve a tocar el bloque de prueba salvo que ``forzar``.
    """
    ruta = RESULTADOS / "metricas_prueba.csv"
    if ruta.exists() and not forzar:
        print("metricas_prueba.csv ya existe: la prueba NO se reevalúa")
        return

    train, val, test, alfabeto = cargar_datos()
    if alpha_por_n is None:
        alpha_por_n = asegurar_alpha_por_n(train, val, alfabeto)
    if best_n is None:
        bpc_val = bpc_validacion_por_n(train, val, alfabeto, alpha_por_n)
        best_n = mejor_n_por_validacion(bpc_val)

    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "alpha", "bloque", "caracteres", "bpc"])
        for n in N_VALORES:
            alpha = alpha_por_n[n]
            modelo = ModeloNgramas(train, n, alfabeto, alpha)
            for bloque, inicio in enumerate(range(0, len(test), TAMANO_BLOQUE), 1):
                fin = min(inicio + TAMANO_BLOQUE, len(test))
                fragmento = test[inicio:fin]
                prefijo = (val + test[:inicio])[-len(prefijo_de(test, n)):] if n > 1 else ""
                bpc = modelo.bpc(fragmento, prefijo)
                w.writerow([n, f"{alpha}", bloque, fin - inicio, f"{bpc:.10f}"])

    # F4: promedio acumulado sobre prueba, una sola vez y solo para el mejor n.
    modelo = ModeloNgramas(train, best_n, alfabeto, alpha_por_n[best_n])
    acumulado = 0.0
    ejes_x, ejes_y = [], []
    for posicion, (contexto, simbolo) in enumerate(
            modelo.secuencias(test, prefijo_de(val, best_n)), 1):
        acumulado += -math.log2(modelo.prob(contexto, simbolo))
        ejes_x.append(posicion)
        ejes_y.append(acumulado / posicion)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ejes_x, ejes_y, color="#7b3294", lw=1.1)
    ax.set_title(f"F4. Promedio acumulado de $-\\log_2 q$ en prueba (n={best_n})")
    ax.set_xlabel("caracteres evaluados en prueba")
    ax.set_ylabel("promedio acumulado (bits/carácter)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    guardar_figura(fig, "F4")
    plt.close(fig)

    print("Evaluación en prueba ejecutada UNA vez: resultados/metricas_prueba.csv y F4.pdf")


def fase4_evaluacion(forzar: bool = False) -> None:
    """T3, F3, F4, desigualdad de Gibbs y caché de prueba."""
    train, val, test, alfabeto = cargar_datos()
    alpha_por_n = asegurar_alpha_por_n(train, val, alfabeto)
    bpc_val = bpc_validacion_por_n(train, val, alfabeto, alpha_por_n)
    mejor = mejor_n_por_validacion(bpc_val)

    evaluar_prueba_una_vez(mejor, alpha_por_n, forzar=forzar)
    filas = leer_metricas_prueba()

    bpc_train = {
        n: ModeloNgramas(train, n, alfabeto, alpha_por_n[n]).bpc(train)
        for n in N_VALORES
    }
    bpc_test: dict[int, float] = {}
    for n in N_VALORES:
        bloques = [f for f in filas if f["n"] == n]
        total = sum(f["caracteres"] for f in bloques)
        bpc_test[n] = sum(f["bpc"] * f["caracteres"] for f in bloques) / total

    # Gibbs con unigramas: p es la frecuencia empírica en prueba; q, el modelo n=1.
    conteos_test = {y: test.count(y) for y in alfabeto}
    p_test = {y: c / len(test) for y, c in conteos_test.items() if c > 0}
    modelo_uni = ModeloNgramas(train, 1, alfabeto, alpha_por_n[1])
    h_p = -sum(p * math.log2(p) for p in p_test.values())
    h_pq = -sum(p * math.log2(modelo_uni.prob("", y)) for y, p in p_test.items())
    kl = h_pq - h_p
    if kl < -1e-12:
        raise AssertionError("La desigualdad de Gibbs no se cumple")

    with (RESULTADOS / "T3.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "alpha", "bpc_validacion", "bpc_train", "bpc_prueba", "ppl_prueba",
            "es_mejor_n", "H_p_prueba", "H_pq_prueba", "KL_prueba",
        ])
        for n in N_VALORES:
            w.writerow([
                n, alpha_por_n[n], f"{bpc_val[n]:.6f}", f"{bpc_train[n]:.6f}",
                f"{bpc_test[n]:.6f}", f"{2 ** bpc_test[n]:.6f}",
                int(n == mejor), f"{h_p:.6f}", f"{h_pq:.6f}", f"{kl:.6f}",
            ])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(N_VALORES, [bpc_train[n] for n in N_VALORES], "o-",
            color="#2166ac", label="entrenamiento")
    ax.plot(N_VALORES, [bpc_test[n] for n in N_VALORES], "s-",
            color="#b2182b", label="prueba")
    ax.axvline(mejor, color="grey", ls="--", lw=1)
    ax.set_title("F3. BPC de entrenamiento y de prueba contra el orden n")
    ax.set_xlabel("n (orden del modelo)")
    ax.set_ylabel("BPC (bits/carácter)")
    ax.set_xticks(N_VALORES)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    guardar_figura(fig, "F3")
    plt.close(fig)

    print(f"Fase 4 lista: T3.csv, F3.pdf, F4.pdf. Mejor n por validación = {mejor}")
    print(f"  H(p_prueba)={h_p:.6f} <= H(p,q)={h_pq:.6f}; KL={kl:.6f} bits")


# ---------------------------------------------------------------------------
# Fase 6 — incertidumbre por bootstrap de bloques
# ---------------------------------------------------------------------------

def fase6_bootstrap() -> None:
    """T4: media e IC95% bootstrap de Delta_j = BPC(mejor n) - BPC(n-1)."""
    train, val, _test, alfabeto = cargar_datos()
    alpha_por_n = asegurar_alpha_por_n(train, val, alfabeto)
    bpc_val = bpc_validacion_por_n(train, val, alfabeto, alpha_por_n)
    mejor = mejor_n_por_validacion(bpc_val)
    comparado = mejor - 1 if mejor > 1 else 2

    filas = leer_metricas_prueba()
    por_n = {}
    for n in (mejor, comparado):
        por_n[n] = {f["bloque"]: f["bpc"] for f in filas if f["n"] == n}
    bloques = sorted(set(por_n[mejor]) & set(por_n[comparado]))
    deltas = np.array([por_n[mejor][b] - por_n[comparado][b] for b in bloques])
    if deltas.size == 0:
        raise SystemExit("No hay bloques comunes para comparar los dos modelos")

    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    medias = rng.choice(deltas, size=(N_REMUESTRAS, deltas.size), replace=True).mean(axis=1)
    inferior, superior = np.percentile(medias, [2.5, 97.5])
    media = float(deltas.mean())
    excluye_cero = bool(inferior > 0 or superior < 0)

    columnas = ["metrica", "valor", "unidad"]
    datos = [
        ("mejor_n", mejor, "orden"),
        ("n_comparado", comparado, "orden"),
        ("numero_bloques", deltas.size, "bloques"),
        ("tamano_bloque", TAMANO_BLOQUE, "caracteres"),
        ("media_delta", f"{media:.6f}", "bits/carácter"),
        ("IC95_inferior", f"{inferior:.6f}", "bits/carácter"),
        ("IC95_superior", f"{superior:.6f}", "bits/carácter"),
        ("remuestras", N_REMUESTRAS, "réplicas"),
        ("semilla", SEMILLA_BOOTSTRAP, "entero"),
        ("intervalo_excluye_cero", int(excluye_cero), "booleano"),
    ]
    with (RESULTADOS / "T4.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columnas)
        w.writerows(datos)

    print("Fase 6 lista: resultados/T4.csv")
    print(f"  Delta={media:.6f} bits, IC95%=[{inferior:.6f}, {superior:.6f}], "
          f"bloques={deltas.size}, excluye cero={excluye_cero}")


# ---------------------------------------------------------------------------
# Fases de B (pendientes) y orquestación
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fase 3 — suavizado y elección de alpha (B)
# ---------------------------------------------------------------------------

def fase3_alpha() -> None:
    """F2 y validacion_alpha.csv: BPC de validación contra alpha, por orden n."""
    train, val, _test, alfabeto = cargar_datos()
    matriz = bpc_validacion_por_alpha(train, val, alfabeto)
    calculados = {n: mejor_alpha_de(matriz[n]) for n in N_VALORES}
    elegidos = asegurar_alpha_por_n(train, val, alfabeto, calculados)
    for n in N_VALORES:
        if abs(calculados[n] - elegidos[n]) > 1e-12:
            raise AssertionError(
                f"n={n}: el alpha cacheado ({elegidos[n]}) difiere del que da "
                f"validación ({calculados[n]})")

    with (RESULTADOS / "validacion_alpha.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "alpha", "bpc_validacion", "es_mejor"])
        for n in N_VALORES:
            for alpha in ALPHAS:
                w.writerow([n, alpha, f"{matriz[n][alpha]:.10f}",
                            int(alpha == elegidos[n])])

    estilos = ("o-", "s-", "^-", "d-", "v-")
    colores = ("#2166ac", "#b2182b", "#1b9e77", "#7570b3", "#d95f02")
    fig, ax = plt.subplots(figsize=(8.2, 5))
    for i, n in enumerate(N_VALORES):
        ax.plot(ALPHAS, [matriz[n][a] for a in ALPHAS], estilos[i],
                color=colores[i], ms=5,
                label=rf"n={n}  ($\alpha={elegidos[n]:g}$)")
    ax.set_xscale("log")
    ax.set_xticks(list(ALPHAS))
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
    ax.set_xlabel(r"$\alpha$ (escala logarítmica)")
    ax.set_ylabel("BPC en validación (bits/carácter)")
    ax.set_title(r"F2. BPC de validación contra $\alpha$, una curva por orden $n$")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    guardar_figura(fig, "F2")
    plt.close(fig)

    print("Fase 3 lista: F2.pdf, F2.png y validacion_alpha.csv")
    for n in N_VALORES:
        print(f"  n={n}: alpha={elegidos[n]:g}, BPC_valid={matriz[n][elegidos[n]]:.6f}")


# ---------------------------------------------------------------------------
# Fase 5 — simulación y distribución estacionaria (B)
# ---------------------------------------------------------------------------

def _matriz_transicion(modelo: ModeloNgramas) -> np.ndarray:
    """P[y, z] = q_alpha(z | y) para el bigrama suavizado."""
    V = modelo.V
    return np.array([[modelo.prob(h, y) for y in V] for h in V], dtype=float)


def _distribucion_estacionaria(P: np.ndarray):
    """pi_0 P^t hasta que la norma l1 entre iteraciones sea < 1e-10."""
    pi = np.full(P.shape[0], 1.0 / P.shape[0])
    for iteracion in range(1, MAX_ITER_ESTACIONARIA + 1):
        nuevo = pi @ P
        norma = float(np.abs(nuevo - pi).sum())
        pi = nuevo
        if norma < TOL_ESTACIONARIA:
            return pi, iteracion, norma
    raise RuntimeError("La iteración de potencias no convergió")


def _generar_muestra(modelo: ModeloNgramas, contexto_inicial: str,
                     rng, longitud: int) -> str:
    """Genera `longitud` símbolos con la semilla ya fijada en `rng`."""
    V = modelo.V
    k = modelo.n - 1
    contexto = contexto_inicial[-k:] if k else ""
    salida: list[str] = []
    for _ in range(longitud):
        probs = np.array([modelo.prob(contexto, y) for y in V], dtype=float)
        total = probs.sum()
        if total <= 0.0:
            raise RuntimeError("Distribución degenerada al generar la muestra")
        probs /= total
        simbolo = V[int(rng.choice(len(V), p=probs))]
        salida.append(simbolo)
        if k:
            contexto = (contexto + simbolo)[-k:]
    return "".join(salida)


def fase5_simulacion() -> None:
    """F5, matriz P, pi, verificaciones y 3 muestras de 300 caracteres."""
    train, val, _test, alfabeto = cargar_datos()
    alpha_por_n = asegurar_alpha_por_n(train, val, alfabeto)
    alpha_bigrama = alpha_por_n[2]
    modelo_bi = ModeloNgramas(train, 2, alfabeto, alpha_bigrama)

    P = _matriz_transicion(modelo_bi)
    error_filas = float(np.abs(P.sum(axis=1) - 1.0).max())
    min_entrada = float(P.min())
    if error_filas > 1e-12 or min_entrada <= 0.0:
        raise AssertionError("P no es estocástica o tiene entradas no positivas")

    pi, iteraciones, norma = _distribucion_estacionaria(P)
    uni = ModeloNgramas(train, 1, alfabeto)
    total_train = sum(uni.pares[""].values())
    frecuencias = {y: uni.pares[""][y] / total_train for y in alfabeto}
    max_diff = max(abs(float(pi[i]) - frecuencias[y])
                   for i, y in enumerate(alfabeto))

    with (RESULTADOS / "estacionaria.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["simbolo", "pi", "frecuencia_train", "diferencia"])
        for i, y in enumerate(alfabeto):
            w.writerow([y, f"{pi[i]:.10f}", f"{frecuencias[y]:.10f}",
                        f"{abs(float(pi[i]) - frecuencias[y]):.10f}"])

    contexto_inicial = " que "
    if contexto_inicial not in train:
        raise RuntimeError("El contexto inicial no aparece en el texto de entrenamiento")
    rng = np.random.default_rng(SEMILLA_SIMULACION)
    muestras = []
    for n in (2, 3, 5):
        modelo = modelo_bi if n == 2 else ModeloNgramas(train, n, alfabeto, alpha_por_n[n])
        texto = _generar_muestra(modelo, contexto_inicial, rng, LONGITUD_MUESTRA)
        muestras.append((n, alpha_por_n[n], texto))

    with (RESULTADOS / "muestras_simulacion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "alpha", "semilla", "contexto_inicial", "muestra"])
        for n, alpha, texto in muestras:
            w.writerow([n, alpha, SEMILLA_SIMULACION, contexto_inicial, texto])

    with (RESULTADOS / "verificacion_simulacion.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor", "unidad_o_tolerancia"])
        w.writerows([
            ("alpha_bigrama", alpha_bigrama, "sin unidad"),
            ("dimension_P", f"{P.shape[0]}x{P.shape[1]}", "simbolos"),
            ("max_error_sumas_filas", f"{error_filas:.3e}", "tolerancia 1e-12"),
            ("min_entrada_P", f"{min_entrada:.3e}", "probabilidad > 0"),
            ("iteraciones_potencias", iteraciones, "iteraciones"),
            ("norma_l1_final", f"{norma:.3e}", "tolerancia 1e-10"),
            ("max_diferencia_pi_vs_unigramas", f"{max_diff:.6f}", "probabilidad"),
            ("semilla_simulacion", SEMILLA_SIMULACION, "entero"),
            ("longitud_muestra", LONGITUD_MUESTRA, "caracteres"),
            ("contexto_inicial", contexto_inicial, "caracteres"),
        ])

    orden = sorted(range(len(alfabeto)), key=lambda i: -frecuencias[alfabeto[i]])
    etiquetas = [alfabeto[i] if alfabeto[i] != " " else "espacio" for i in orden]
    posiciones = np.arange(len(orden))
    ancho = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(posiciones - ancho / 2, [float(pi[i]) for i in orden], ancho,
           color="#2166ac", label=r"$\pi$ (estacionaria)")
    ax.bar(posiciones + ancho / 2, [frecuencias[alfabeto[i]] for i in orden], ancho,
           color="#b2182b", label="frecuencia de unigramas")
    ax.set_xticks(posiciones)
    ax.set_xticklabels(etiquetas, rotation=90, fontsize=8)
    ax.set_xlabel("símbolo (ordenado por frecuencia en entrenamiento)")
    ax.set_ylabel("probabilidad")
    ax.set_title("F5. Distribución estacionaria del bigrama frente a los unigramas")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    guardar_figura(fig, "F5")
    plt.close(fig)

    print("Fase 5 lista: F5.pdf, estacionaria.csv, verificacion_simulacion.csv "
          "y muestras_simulacion.csv")
    print(f"  P {P.shape[0]}x{P.shape[1]}: max error filas={error_filas:.3e}, "
          f"min entrada={min_entrada:.6f}")
    print(f"  pi convergió en {iteraciones} iteraciones (L1={norma:.3e}); "
          f"max |pi - f_train|={max_diff:.6f}")


# ---------------------------------------------------------------------------
# Fase 7 — BPE opcional (B)
# ---------------------------------------------------------------------------

def _aprender_bpe(train: str, numero_fusiones: int):
    """Fusiones voraces por frecuencia; desempate por orden de aparición."""
    secuencia = SEPARADOR_TOKENS.join(train)
    fusiones = []
    for _ in range(numero_fusiones):
        partes = secuencia.split(SEPARADOR_TOKENS)
        pares = Counter(zip(partes, partes[1:]))
        if not pares:
            break
        par = max(pares, key=lambda p: pares[p])
        secuencia = secuencia.replace(
            par[0] + SEPARADOR_TOKENS + par[1], par[0] + par[1])
        fusiones.append(par)
    return fusiones


def _tokenizar(texto: str, fusiones) -> list[str]:
    """Aplica las reglas BPE aprendidas, en orden, al texto dado."""
    secuencia = SEPARADOR_TOKENS.join(texto)
    for a, b in fusiones:
        secuencia = secuencia.replace(a + SEPARADOR_TOKENS + b, a + b)
    return secuencia.split(SEPARADOR_TOKENS) if secuencia else []


def fase7_bpe() -> None:
    """200 fusiones BPE y BPC por carácter, comparado con la Fase 4."""
    train, val, test, alfabeto = cargar_datos()
    fusiones = _aprender_bpe(train, NUM_FUSIONES_BPE)
    tok_train = _tokenizar(train, fusiones)
    tok_val = _tokenizar(val, fusiones)
    tok_test = _tokenizar(test, fusiones)

    # Las reglas se aprenden SOLO en train. El inventario se cierra con los
    # tokens que esas mismas reglas producen en val/test; sus conteos NO se
    # usan para estimar probabilidades (alpha se elige en validación y la
    # prueba se mide una sola vez al final).
    vocab_tokens = sorted(set(tok_train) | set(tok_val) | set(tok_test))
    nuevos = len(set(vocab_tokens) - set(tok_train))
    codigos = {token: chr(i + 1) for i, token in enumerate(vocab_tokens)}
    alfabeto_tokens = tuple(codigos[t] for t in vocab_tokens)
    train_cod = "".join(codigos[t] for t in tok_train)
    val_cod = "".join(codigos[t] for t in tok_val)
    test_cod = "".join(codigos[t] for t in tok_test)

    n_tokens_val = len(tok_val)
    n_caracteres_val = sum(len(t) for t in tok_val)
    mejor_alpha = ALPHAS[0]
    mejor_bpc = math.inf
    for alpha in ALPHAS:
        modelo = ModeloNgramas(train_cod, 2, alfabeto_tokens, alpha)
        bpc_por_token = modelo.bpc(val_cod, prefijo_de(train_cod, 2))
        bpc_por_caracter = bpc_por_token * n_tokens_val / n_caracteres_val
        if bpc_por_caracter < mejor_bpc:
            mejor_bpc = bpc_por_caracter
            mejor_alpha = alpha

    modelo_final = ModeloNgramas(train_cod, 2, alfabeto_tokens, mejor_alpha)
    bpc_token_test = modelo_final.bpc(test_cod, prefijo_de(val_cod, 2))
    n_caracteres_test = sum(len(t) for t in tok_test)
    bpe_bpc_caracter = bpc_token_test * len(tok_test) / n_caracteres_test

    bpc_fase4 = None
    mejor_n = None
    with (RESULTADOS / "T3.csv").open(newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if fila["es_mejor_n"] == "1":
                bpc_fase4 = float(fila["bpc_prueba"])
                mejor_n = int(fila["n"])
    if bpc_fase4 is None:
        raise SystemExit("Falta T3.csv con el mejor n de la Fase 4")

    with (RESULTADOS / "bpe_resultados.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor", "unidad"])
        w.writerows([
            ("fusiones_bpe", len(fusiones), "fusiones"),
            ("vocabulario_tokens", len(vocab_tokens), "tokens"),
            ("tokens_nuevos_val_prueba", nuevos, "tokens"),
            ("alpha_elegido_validacion", mejor_alpha, "sin unidad"),
            ("tokens_train", len(tok_train), "tokens"),
            ("tokens_prueba", len(tok_test), "tokens"),
            ("caracteres_prueba", n_caracteres_test, "caracteres"),
            ("bpc_caracter_bpe", f"{bpe_bpc_caracter:.6f}", "bits/caracter"),
            ("mejor_n_fase4", mejor_n, "orden"),
            ("bpc_caracter_fase4", f"{bpc_fase4:.6f}", "bits/caracter"),
            ("diferencia_bpe_menos_fase4",
             f"{bpe_bpc_caracter - bpc_fase4:.6f}", "bits/caracter"),
            ("metricas_prueba_csv_reescrito", 0, "booleano"),
        ])

    print("Fase 7 lista: resultados/bpe_resultados.csv")
    print(f"  {len(fusiones)} fusiones, |V_tokens|={len(vocab_tokens)}, "
          f"alpha={mejor_alpha:g}")
    print(f"  BPE {bpe_bpc_caracter:.6f} vs Fase 4 (n={mejor_n}) "
          f"{bpc_fase4:.6f} bits/carácter")


FASES = {
    "2": ("Fase 2 (A)", fase2_mle),
    "4": ("Fase 4 (A)", fase4_evaluacion),
    "6": ("Fase 6 (A)", fase6_bootstrap),
    "3": ("Fase 3 (B)", fase3_alpha),
    "5": ("Fase 5 (B)", fase5_simulacion),
    "7": ("Fase 7 (B, opcional)", fase7_bpe),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Fases del proyecto S01")
    parser.add_argument("fases", nargs="*", default=["A"],
                        help="A (por defecto), B, todo, o números: 2 4 6")
    parser.add_argument("--forzar-prueba", action="store_true",
                        help="reevaluar el bloque de prueba (rompe el protocolo si se usa dos veces)")
    args = parser.parse_args(argv)

    seleccion = []
    for entrada in args.fases:
        if entrada.lower() == "a":
            seleccion.extend(["2", "4", "6"])
        elif entrada.lower() == "b":
            seleccion.extend(["3", "5", "7"])
        elif entrada.lower() == "todo":
            seleccion.extend(FASES)
        elif entrada in FASES:
            seleccion.append(entrada)
        else:
            parser.error(f"fase desconocida: {entrada}")

    for clave in dict.fromkeys(seleccion):
        nombre, funcion = FASES[clave]
        try:
            if clave == "4":
                funcion(forzar=args.forzar_prueba)
            else:
                funcion()
        except NotImplementedError as exc:
            print(f"[pendiente] {nombre}: {exc}")


if __name__ == "__main__":
    main()
