"""Fases 2–6 del proyecto (y 7 opcional).

Este archivo es el esqueleto acordado en el plan. Las fases ya listas son
la Fase 1 (en preparar.py) y las verificaciones (en pruebas.py).

Punto de protocolo: evaluar_prueba_una_vez() debe escribir
resultados/metricas_prueba.csv con el BPC por bloque de 2000 caracteres.
Fase 4 y Fase 6 lean ESE archivo; nunca vuelven a evaluar prueba.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos import ModeloNgramas  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
RESULTADOS = ROOT / "resultados"


def evaluar_prueba_una_vez() -> None:
    """TODO(Fase 4): calcular y cachear métricas de prueba.

    Debe guardar resultados/metricas_prueba.csv con columnas:
    n, alpha, bloque, bpc
    """
    raise NotImplementedError("Fase 4 pendiente")


def fase2_mle() -> None:
    """TODO: T2 (contextos, ceros en validación, log-verosimilitud y perturbaciones)."""
    raise NotImplementedError("Fase 2 pendiente")


def fase3_alpha() -> None:
    """TODO: F2 (BPC de validación vs alpha, elección de alpha por n)."""
    raise NotImplementedError("Fase 3 pendiente")


def fase4_evaluacion() -> None:
    """TODO: T3, F3, F4, Gibbs y metricas_prueba.csv."""
    raise NotImplementedError("Fase 4 pendiente")


def fase5_simulacion() -> None:
    """TODO: matriz P, pi por potencias, F5 y muestras con semilla."""
    raise NotImplementedError("Fase 5 pendiente")


def fase6_bootstrap() -> None:
    """TODO: T4 (bloques de 2000, 10000 remuestras, intervalo percentil)."""
    raise NotImplementedError("Fase 6 pendiente")


def fase7_bpe() -> None:
    """TODO(opcional): 200 fusiones BPE y BPC por carácter."""
    raise NotImplementedError("Fase 7 pendiente")


def main() -> None:
    print("Esqueleto listo. Ejecuta primero: preparar.py y pruebas.py.")
    print("Fases 2–6 pendientes de asignar entre los dos integrantes.")


if __name__ == "__main__":
    main()
