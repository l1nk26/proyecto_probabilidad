"""Fase 1: leer, limpiar, normalizar y particionar el corpus.

Entrada: datos/raw/divina_comedia.txt (Gutenberg #57303, CP1252).
Salidas:
  datos/corpus/{train,val,test}.txt   (NO se entregan)
  resultados/T1.csv
  resultados/F1.pdf
  resultados/vocabulario.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

# Misma fecha fija que experimentos.py: F1 es byte-idéntico al repetir.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1795392000")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CRUDO = ROOT / "datos" / "raw" / "divina_comedia.txt"
CORPUS = ROOT / "datos" / "corpus"
RESULTADOS = ROOT / "resultados"

ALFABETO = "abcdefghijklmnopqrstuvwxyzáéíóúñü "


def leer_y_recortar(path: Path) -> str:
    texto = path.read_bytes().decode("cp1252")
    inicio = texto.index("*** START OF THIS PROJECT GUTENBERG EBOOK")
    fin = texto.index("*** END OF THIS PROJECT GUTENBERG EBOOK")
    cuerpo = texto[inicio:fin].split("\n", 1)[1]

    primero = cuerpo.index("_CANTO PRIMERO_")
    indice = cuerpo.index("_INDICE_", primero)
    poema = cuerpo[primero:indice]

    poema = re.sub(r"(?im)^_CANTO [^_]+_\s*$", " ", poema)
    poema = re.sub(r"(?i)\bFIN\b", " ", poema)
    poema = re.sub(r"\[Ilustración\]", " ", poema, flags=re.IGNORECASE)
    poema = re.sub(r"\[\d+\]", " ", poema)
    return poema


def normalizar(texto: str, alfabeto: str = ALFABETO) -> str:
    permitido = set(alfabeto)
    salida = []
    for caracter in texto.lower():
        salida.append(caracter if caracter in permitido else "#")
    return re.sub(r"\s+", " ", "".join(salida)).strip()


def particionar(texto: str):
    n = len(texto)
    fin_train = int(0.8 * n)
    fin_val = int(0.9 * n)
    return texto[:fin_train], texto[fin_train:fin_val], texto[fin_val:]


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    crudo = leer_y_recortar(CRUDO)
    limpio = normalizar(crudo)
    if len(limpio) < 300_000:
        raise SystemExit(f"Corpus demasiado corto: {len(limpio)} caracteres")

    train, val, test = particionar(limpio)
    (CORPUS / "train.txt").write_text(train, encoding="utf-8")
    (CORPUS / "val.txt").write_text(val, encoding="utf-8")
    (CORPUS / "test.txt").write_text(test, encoding="utf-8")

    vocab = sorted(set(train))
    frecuencias = {ch: train.count(ch) for ch in vocab}
    h_train = -sum((c / len(train)) * math.log2(c / len(train))
                   for c in frecuencias.values())
    h_max = math.log2(len(vocab))

    with (RESULTADOS / "T1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor", "unidad"])
        w.writerow(["entrenamiento", len(train), "caracteres"])
        w.writerow(["validacion", len(val), "caracteres"])
        w.writerow(["prueba", len(test), "caracteres"])
        w.writerow(["total", len(limpio), "caracteres"])
        w.writerow(["vocabulario", len(vocab), "simbolos"])
        w.writerow(["H_train", round(h_train, 4), "bits/simbolo"])
        w.writerow(["H_max", round(h_max, 4), "bits/simbolo"])

    (RESULTADOS / "vocabulario.json").write_text(
        json.dumps({"alfabeto": ALFABETO, "V": vocab,
                    "|V|": len(vocab), "frecuencias_train": frecuencias},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    ordenados = sorted(frecuencias.items(), key=lambda x: -x[1])
    etiquetas = [k if k != " " else "espacio" for k, _ in ordenados]
    valores = [v for _, v in ordenados]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(etiquetas, valores, color="#25608a")
    ax.set_title("F1. Frecuencias de unigramas en entrenamiento")
    ax.set_xlabel("símbolo")
    ax.set_ylabel("frecuencia (caracteres)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "F1.pdf")
    fig.savefig(RESULTADOS / "F1.png", dpi=150)
    plt.close(fig)

    print(f"Corpus limpio: {len(limpio)} caracteres")
    print(f"Partición: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"|V| = {len(vocab)}   H_train = {h_train:.4f} bits   H_max = {h_max:.4f} bits")
    print("Escritos: resultados/T1.csv, resultados/F1.pdf, resultados/vocabulario.json")


if __name__ == "__main__":
    main()
