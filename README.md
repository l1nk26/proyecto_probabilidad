# Proyecto S01 — Modelos de n-gramas sobre caracteres

Equipo: 2 personas. Fuente: *La Divina Comedia* (ver `datos/FUENTE.md`).  
Reparto de tareas, dependencias y estado actual: ver `PLAN.md`.  
Fuente canónica del informe: `reportes/reporte.md` (Markdown con matemáticas LaTeX; el PDF se genera aparte desde Obsidian).

## Estructura

```text
datos/raw/        texto crudo, NO se entrega
datos/corpus/     train.txt / val.txt / test.txt, NO se entregan
codigo/           preparar.py, modelos.py, experimentos.py, pruebas.py
resultados/       T1..T4.csv, F1..F5.pdf (+ PNG), CSV de fases y métricas
reportes/         reporte.md (fuente del informe)
```

## Orden de ejecución

```bash
python codigo/preparar.py
python codigo/pruebas.py
python codigo/experimentos.py A      # fases 2, 4 y 6 (responsable A)
python codigo/experimentos.py B      # fases 3, 5 y 7 (responsable B)
```

También puede ejecutarse una fase suelta, por ejemplo `python codigo/experimentos.py 3 5`, o `todo` para las siete fases.

`preparar.py` genera el corpus limpio, T1, F1 y `vocabulario.json`.  
`pruebas.py` reproduce la prueba de cordura de `abracadabra`, Gibbs y la convergencia de $\pi$ en el ejemplo 2×2.  
`experimentos.py A` genera T2, T3, T4, F3, F4 y `metricas_prueba.csv`; la evaluación en prueba se hace **una sola vez**: si el CSV ya existe, una segunda corrida imprime `la prueba NO se reevalúa`. Solo `python codigo/experimentos.py 4 --forzar-prueba` la regenera deliberadamente.  
`experimentos.py B` genera F2, F5, las muestras y los CSV de las fases 3, 5 y 7. **Nunca evalúa el bloque de prueba para elegir parámetros**: solo la Fase 7 lo mide al final, una vez, para reportar su BPE, sin reescribir `metricas_prueba.csv`.

## Archivos de resultados

| Archivo | Contenido |
|---|---|
| `T1.csv`, `F1.pdf` | corpus, vocabulario y frecuencias (Fase 1) |
| `T2.csv` | contextos, ceros y log-verosimilitud perturbada (Fase 2) |
| `validacion_alpha.csv`, `F2.pdf` | BPC de validación por $(n,\alpha)$ (Fase 3) |
| `T3.csv`, `F3.pdf`, `F4.pdf`, `metricas_prueba.csv` | evaluación única en prueba (Fase 4) |
| `estacionaria.csv`, `verificacion_simulacion.csv`, `muestras_simulacion.csv`, `F5.pdf` | simulación (Fase 5) |
| `T4.csv` | bootstrap por bloques (Fase 6) |
| `bpe_resultados.csv` | 200 fusiones BPE y comparación por carácter (Fase 7) |

Los PNG (`F1.png`…`F5.png`) son solo para incrustar las figuras en el Markdown; no se entregan en el zip.

## Reproducibilidad

- Python 3.14, NumPy 2.2.6, Matplotlib 3.10.7 (`pip install -r requirements.txt`).
- El texto crudo se decodifica como **CP1252** (Gutenberg #57303).
- Semillas fijas: bootstrap `20261122`, simulación `20261123` (Anexo B del informe).
- `SOURCE_DATE_EPOCH=1795392000` fija los metadatos de los PDF: repetir el pipeline produce archivos byte-idénticos (verificado).
- El bloque de prueba no interviene en la elección de $n$ ni de $\alpha$.

## Entrega

Primero convierte `reportes/reporte.md` a PDF (Obsidian → Export to PDF, usando `resultados/F*.png`) y copia el resultado a `reporte.pdf` en la raíz. Después:

```bash
python empaquetar.py --equipo NN --apellido1 Apellido1 --apellido2 Apellido2
```

El zip final se llama `P1_EquipoNN_Apellido1_Apellido2.zip` y contiene únicamente:

```text
reporte.pdf, README.md, requirements.txt, datos/FUENTE.md,
codigo/{preparar.py,modelos.py,experimentos.py,pruebas.py},
resultados/{T1..T4.csv,F1..F5.pdf,metricas_prueba.csv,
            validacion_alpha.csv,estacionaria.csv,
            verificacion_simulacion.csv,muestras_simulacion.csv,
            bpe_resultados.csv}
```
