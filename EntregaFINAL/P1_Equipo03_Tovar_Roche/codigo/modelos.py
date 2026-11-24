"""Modelos de n-gramas de caracteres para el proyecto S01.

API compartida: conteos, MLE, suavizado aditivo, BPC y PPL.
No depende de experimentos.py para que pruebas.py pueda importarlo.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def entropia(frecuencias: dict[str, float]) -> float:
    """Entropía en bits de una distribución (los valores se normalizan)."""
    total = sum(frecuencias.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in frecuencias.values() if c > 0)


class ModeloNgramas:
    """Cadena de Markov de orden n-1 sobre un alfabeto fijo.

    El contexto de una posición son los n-1 caracteres anteriores.
    Para n=1 el contexto es la cadena vacía.
    """

    def __init__(self, texto_entrenamiento: str, n: int, alfabeto, alpha: float = 0.0):
        if n < 1:
            raise ValueError("n debe ser >= 1")
        self.n = n
        self.V = tuple(alfabeto)
        self.alpha = alpha
        self.contextos: Counter[str] = Counter()
        self.pares: dict[str, Counter[str]] = defaultdict(Counter)
        self._ajustar(texto_entrenamiento)

    def _ajustar(self, texto: str) -> None:
        k = self.n - 1
        if k == 0:
            for simbolo in texto:
                self.contextos[""] += 1
                self.pares[""][simbolo] += 1
            return
        # Se descartan las primeras n-1 posiciones: todo contexto tiene longitud n-1.
        for i in range(k, len(texto)):
            ctx = texto[i - k:i]
            self.contextos[ctx] += 1
            self.pares[ctx][texto[i]] += 1

    def prob(self, contexto: str, simbolo: str) -> float:
        """q_alpha(simbolo | contexto); alpha=0 es MLE."""
        c = self.contextos.get(contexto, 0)
        conteo = self.pares.get(contexto, {}).get(simbolo, 0)
        denominador = c + self.alpha * len(self.V)
        if denominador == 0:
            return 1.0 / len(self.V)
        return (conteo + self.alpha) / denominador

    def suma_condicionales(self, contexto: str) -> float:
        """Suma de q_alpha(y|contexto) sobre todo el vocabulario."""
        return sum(self.prob(contexto, y) for y in self.V)

    def secuencias(self, texto: str, prefijo: str = ""):
        """Yield (contexto, simbolo) para cada carácter de texto.

        prefijo aporta los caracteres previos cuando texto es una partición
        distinta de entrenamiento (validación o prueba).
        """
        k = self.n - 1
        if k == 0:
            for simbolo in texto:
                yield "", simbolo
            return
        cola = prefijo[-k:] if prefijo else ""
        completo = cola + texto
        desplazamiento = len(cola)
        for j in range(len(texto)):
            i = desplazamiento + j
            ctx = completo[i - k:i]
            if len(ctx) == k:
                yield ctx, texto[j]

    def bpc(self, texto: str, prefijo: str = "") -> float:
        """Bits por carácter. Devuelve inf si aparece una probabilidad cero."""
        total = 0.0
        evaluados = 0
        for contexto, simbolo in self.secuencias(texto, prefijo):
            p = self.prob(contexto, simbolo)
            if p <= 0:
                return math.inf
            total -= math.log2(p)
            evaluados += 1
        if evaluados == 0:
            return 0.0
        return total / evaluados

    def ppl(self, texto: str, prefijo: str = "") -> float:
        bpc = self.bpc(texto, prefijo)
        return math.inf if math.isinf(bpc) else 2.0 ** bpc


def verificar_sumas(modelo: ModeloNgramas, tolerancia: float = 1e-12) -> bool:
    """Todas las condicionales observadas suman 1 sobre V."""
    return all(abs(modelo.suma_condicionales(h) - 1.0) < tolerancia
               for h in modelo.contextos)
