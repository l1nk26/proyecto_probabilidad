# Proyecto S01 — Cómo ejecutar

Desde la raíz del repositorio, en este orden:

```bash
python -m pip install -r requirements.txt
python codigo/preparar.py
python codigo/pruebas.py
python codigo/experimentos.py A
python codigo/experimentos.py B
```

Requiere `datos/raw/divina_comedia.txt` (Gutenberg \#57303, codificado en CP1252) en `datos/raw/`.

`python codigo/experimentos.py todo` equivale a los dos últimos comandos. También se puede correr una fase suelta, p. ej. `python codigo/experimentos.py 3 5`.

## Salidas

| Comando             | Genera                                                                                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preparar.py`       | `datos/corpus/{train,val,test}.txt`, `resultados/T1.csv`, `resultados/F1.pdf`, `resultados/F1.png`, `resultados/vocabulario.json`                                 |
| `pruebas.py`        | Verificaciones por pantalla (no escribe archivos)                                                                                                                 |
| `experimentos.py A` | `resultados/T2.csv`, `T3.csv`, `T4.csv`, `F3.pdf`, `F4.pdf` (+ PNG), `metricas_prueba.csv`                                                                        |
| `experimentos.py B` | `resultados/validacion_alpha.csv`, `F2.pdf`, `F5.pdf` (+ PNG), `estacionaria.csv`, `verificacion_simulacion.csv`, `muestras_simulacion.csv`, `bpe_resultados.csv` |

Si `metricas_prueba.csv` ya existe, la segunda ejecución no reevalúa la prueba: imprime `la prueba NO se reevalúa`. Para regenerarla deliberadamente: `python codigo/experimentos.py 4 --forzar-prueba`.
