"""Genera un PDF de borrador a partir de reportes/reporte.tex.

No se entrega en el zip: es solo un auxiliar local porque en esta máquina no
hay pdflatex. La fuente canónica del reporte sigue siendo reporte.tex.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
FUENTE = ROOT / "reportes" / "reporte.tex"
SALIDA = ROOT / "reportes" / "reporte.pdf"


def limpiar(texto: str) -> str:
    s = texto.strip()
    s = re.sub(r"\\\[[^\]]*\]", " ", s)
    s = re.sub(r"\\hspace\{[^}]*\}", " ", s)
    s = re.sub(r"\\vfill|\\quad|\\,|\\;|\\ ", " ", s)
    s = s.replace("\\[", " ").replace("\\]", " ").replace("$", "")
    s = s.replace("\\alpha", "alpha").replace("\\Delta", "Delta")
    s = s.replace("\\mathcal", "").replace("\\hat", "")
    s = s.replace("\\vert", "|").replace("\\pi", "pi")
    s = s.replace("\\log", "log").replace("\\sum", "sum").replace("\\prod", "prod")
    s = s.replace("\\times", "x").replace("\\le", "<=").replace("\\ge", ">=")
    s = s.replace("\\%", "%").replace("\\#", "#").replace("\\_", "_").replace("\\&", "&")
    s = re.sub(r"\\(textbf|emph|texttt|underline|textit)\{", "", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("---", "-").replace("--", "-").replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ajustar(texto: str, ancho: int = 95) -> str:
    """Parte palabras muy largas (URLs) para que multi_cell nunca se atragante."""
    return "\n".join(textwrap.wrap(texto, width=ancho, break_long_words=True,
                                   break_on_hyphens=False)) if texto else ""


def extraer():
    texto = FUENTE.read_text(encoding="utf-8")
    en_doc = False
    en_verbatim = False
    en_tabla = False
    salida = []

    for linea in texto.splitlines():
        cruda = linea.strip()
        if not en_doc:
            if cruda.startswith("\\begin{document}"):
                en_doc = True
            continue

        if cruda.startswith("\\begin{verbatim}"):
            en_verbatim = True
            continue
        if cruda.startswith("\\end{verbatim}"):
            en_verbatim = False
            continue
        if cruda.startswith("\\begin{tabular}"):
            en_tabla = True
            continue
        if cruda.startswith("\\end{tabular}"):
            en_tabla = False
            continue

        if en_verbatim:
            salida.append(("V", linea.rstrip()))
            continue

        if cruda.startswith(("\\begin{titlepage}", "\\end{titlepage}", "\\begin{center}",
                             "\\end{center}", "\\hline", "\\vfill")):
            continue
        if cruda.startswith("\\begin{abstract}"):
            salida.append(("H1", "Resumen"))
            continue
        if cruda.startswith("\\end{abstract}") or cruda.startswith("\\end{document}"):
            continue

        head = re.match(r"\\(section|subsection|paragraph)\*?\{(.*)\}", cruda)
        if head:
            nivel = {"section": "H1", "subsection": "H2", "paragraph": "H3"}[head.group(1)]
            salida.append((nivel, limpiar(head.group(2))))
            continue

        if en_tabla:
            fila = cruda.replace("&", "|").replace("\\\\", "").strip()
            if fila:
                salida.append(("P", limpiar(fila)))
            continue

        if not cruda:
            salida.append(("B", ""))
            continue

        limpio = limpiar(cruda)
        if limpio:
            salida.append(("P", limpio))
    return salida


def main() -> None:
    pdf = FPDF(format="letter", unit="cm")
    pdf.set_auto_page_break(auto=True, margin=2.0)
    pdf.set_margins(2.5, 2.5, 2.5)
    pdf.add_page()

    for tipo, texto in extraer():
        pdf.set_x(pdf.l_margin)
        if tipo == "B":
            pdf.ln(0.18)
            continue
        if tipo == "V":
            pdf.set_font("Courier", size=9)
            pdf.multi_cell(0, 0.45, ajustar(texto))
            continue
        if tipo == "H1":
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(0.25)
            pdf.multi_cell(0, 0.6, ajustar(texto))
            pdf.ln(0.1)
            continue
        if tipo == "H2":
            pdf.set_font("Helvetica", "B", 11)
            pdf.ln(0.2)
            pdf.multi_cell(0, 0.5, ajustar(texto))
            continue
        if tipo == "H3":
            pdf.set_font("Helvetica", "BI", 10)
            pdf.ln(0.1)
            pdf.multi_cell(0, 0.48, ajustar(texto))
            continue
        pdf.set_font("Helvetica", size=10.5)
        pdf.multi_cell(0, 0.48, ajustar(texto))

    pdf.output(str(SALIDA))
    print(f"Escrito {SALIDA} ({pdf.page_no()} páginas)")


if __name__ == "__main__":
    main()
