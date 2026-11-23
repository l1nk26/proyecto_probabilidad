"""Empaqueta la entrega final una vez generado reporte.pdf.

Uso:
    python empaquetar.py --equipo 07 --apellido1 Perez --apellido2 Gomez

No se entrega en el zip: es una herramienta local de ensamblaje.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ARCHIVOS = [
    "reporte.pdf",
    "README.md",
    "requirements.txt",
    "datos/FUENTE.md",
    "codigo/preparar.py",
    "codigo/modelos.py",
    "codigo/experimentos.py",
    "codigo/pruebas.py",
    "resultados/T1.csv",
    "resultados/T2.csv",
    "resultados/T3.csv",
    "resultados/T4.csv",
    "resultados/F1.pdf",
    "resultados/F2.pdf",
    "resultados/F3.pdf",
    "resultados/F4.pdf",
    "resultados/F5.pdf",
    "resultados/metricas_prueba.csv",
    "resultados/validacion_alpha.csv",
    "resultados/estacionaria.csv",
    "resultados/verificacion_simulacion.csv",
    "resultados/muestras_simulacion.csv",
    "resultados/bpe_resultados.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equipo", required=True, help="número de equipo, dos dígitos")
    parser.add_argument("--apellido1", required=True)
    parser.add_argument("--apellido2", required=True)
    args = parser.parse_args()

    equipo = f"{int(args.equipo):02d}"
    nombre = f"P1_Equipo{equipo}_{args.apellido1}_{args.apellido2}.zip"
    faltan = [f for f in ARCHIVOS if not (ROOT / f).is_file()]
    if faltan:
        print("Faltan archivos para empaquetar:", *faltan, sep="\n  ")
        print("Genera reporte.pdf desde reportes/reporte.md y cópialo a la raíz.")
        sys.exit(1)

    destino = ROOT / nombre
    prefijo = nombre.removesuffix(".zip")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ARCHIVOS:
            z.write(ROOT / rel, f"{prefijo}/{rel}")

    with zipfile.ZipFile(destino) as z:
        nombres = z.namelist()
    print(f"Escrito {destino} con {len(nombres)} archivos")
    assert len(nombres) == len(ARCHIVOS)


if __name__ == "__main__":
    main()
