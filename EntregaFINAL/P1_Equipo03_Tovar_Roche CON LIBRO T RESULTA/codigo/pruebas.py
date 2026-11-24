"""Prueba de cordura y verificaciones del proyecto S01."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelos import ModeloNgramas, entropia, verificar_sumas  # noqa: E402

ABRACADABRA = "abracadabra"
TOL = 1e-12


def prueba_abracadabra() -> None:
    conteos = {ch: ABRACADABRA.count(ch) for ch in sorted(set(ABRACADABRA))}
    assert conteos == {"a": 5, "b": 2, "c": 1, "d": 1, "r": 2}, conteos

    h = entropia({k: float(v) for k, v in conteos.items()})
    assert abs(h - 2.0404) < 5e-5, h

    V = sorted(conteos)
    mle = ModeloNgramas(ABRACADABRA, 2, V)
    q_ba = mle.prob("a", "b")
    assert abs(q_ba - 0.5) < TOL, q_ba

    suav = ModeloNgramas(ABRACADABRA, 2, V, alpha=1.0)
    q_suav = suav.prob("a", "b")
    assert abs(q_suav - 1 / 3) < TOL, q_suav

    bpc = mle.bpc(ABRACADABRA)
    ppl = 2 ** bpc
    assert abs(bpc - 0.6) < 5e-5, bpc
    assert abs(ppl - 1.5157) < 5e-5, ppl

    assert verificar_sumas(mle, TOL)
    assert verificar_sumas(suav, TOL)

    print("abracadabra OK")
    print(f"  unigramas: {conteos}")
    print(f"  H = {h:.4f} bits")
    print(f"  q(b|a) MLE = {q_ba:.4f}   q_alpha(b|a) = {q_suav:.4f}")
    print(f"  BPC = {bpc:.4f} bits   PPL = {ppl:.4f}")


def prueba_gibbs() -> None:
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 0.7, "b": 0.3}
    hp = entropia(p)
    hpq = -sum(p[x] * math.log2(q[x]) for x in p)
    kl = hpq - hp
    assert hp <= hpq + TOL
    assert abs(kl - 0.125769) < 5e-5, kl
    print(f"Gibbs OK: H(p)={hp:.6f} <= H(p,q)={hpq:.6f}, KL={kl:.6f}")


def prueba_estacionaria() -> None:
    P = [[0.9, 0.1], [0.2, 0.8]]
    pi = [1.0, 0.0]
    for _ in range(1_000_000):
        nuevo = [sum(pi[i] * P[i][j] for i in range(2)) for j in range(2)]
        if sum(abs(nuevo[i] - pi[i]) for i in range(2)) < 1e-10:
            pi = nuevo
            break
        pi = nuevo
    esperado = [2 / 3, 1 / 3]
    assert all(abs(pi[i] - esperado[i]) < 1e-9 for i in range(2)), pi
    print(f"Estacionaria OK: pi={pi[0]:.9f}, {pi[1]:.9f}")


def main() -> None:
    prueba_abracadabra()
    prueba_gibbs()
    prueba_estacionaria()
    print("TODAS LAS VERIFICACIONES PASARON")


if __name__ == "__main__":
    main()
