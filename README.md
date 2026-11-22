# Proyecto S01 — Modelos de n-gramas sobre caracteres

Equipo: 2 personas. Fuente: *La Divina Comedia* (ver `datos/FUENTE.md`).  
Reparto de tareas, dependencias y estado actual: ver `PLAN.md`.

## Estructura

```text
datos/raw/        texto crudo, NO se entrega
datos/corpus/     train.txt / val.txt / test.txt, NO se entrega
codigo/           preparar.py, modelos.py, experimentos.py, pruebas.py
resultados/       T1..T4.csv, F1..F5.pdf, metricas_prueba.csv
reportes/         fuente LaTeX del reporte
```

## Orden de ejecución

```bash
python codigo/preparar.py
python codigo/pruebas.py
python codigo/experimentos.py
```

`preparar.py` genera el corpus limpio, la tabla T1 y la figura F1.
`pruebas.py` reproduce la prueba de cordura de `abracadabra` y las verificaciones.
`experimentos.py` contendrá las fases 2–6 (y la 7 opcional); la evaluación en prueba se hace una sola vez y se guarda en `resultados/metricas_prueba.csv`.

## Reproducibilidad

- Python 3.14, `pip install -r requirements.txt`.
- El texto crudo se decodifica como **CP1252** (Gutenberg #57303).
- Semillas fijas en simulación y bootstrap (Anexo B del reporte).
- El bloque de prueba se usa una sola vez; no interviene en la elección de `n` ni de `α`.

## Entrega

El zip final se llama `P1_EquipoNN_Apellido1_Apellido2.zip` y contiene únicamente:

```text
reporte.pdf, README.md, requirements.txt, datos/FUENTE.md,
codigo/{preparar.py,modelos.py,experimentos.py,pruebas.py},
resultados/{T1..T4.csv,F1..F5.pdf,metricas_prueba.csv}
```
