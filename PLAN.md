# Plan de trabajo — Proyecto S01 «Modelos de lenguaje de n-gramas»

**Equipo:** 2 personas.  
**Persona A:** tú. **Persona B:** tu compañero.  
**Convención:** si uno prefiere más código o más redacción, intercambien los bloques `A*` / `B*` correspondientes, pero **no** las tareas bloqueantes `T0`–`T4`.

**Fuente ya elegida y aprobada:** *La Divina Comedia*, Dante Alighieri, traducción de Manuel Aranda y Sanjuán.  
URL real de descarga: `https://archive.org/details/ladivinacomedia57303gut` el **2026-11-21**.  
Es el espejo en Internet Archive del Gutenberg eBook #57303; el texto coincide con la edición de elejandria revisada en la semana 1.

---

## Estado actual (actualizado al crear el esqueleto)

**Hecho ✅**
- `T0` fuente aprobada y registrada en `datos/FUENTE.md`.
- `T1` esqueleto del repo, `README.md` y `requirements.txt`.
- `T2` `codigo/preparar.py`: corpus limpio de **668 786 caracteres**, partición contigua 80/10/10, `|V| = 35`, T1/F1 generados.
- `T3` `codigo/pruebas.py`: `abracadabra` (6 cifras), sumas = 1, Gibbs y $\pi$ convergente. Todo pasa.
- `T4` `codigo/modelos.py`: API compartida de conteos, MLE, suavizado, BPC/PPL y contexto con prefijo.

**Pendiente ⚠️**
- `experimentos.py` tiene TODOS los `NotImplementedError` de las fases 2–7.
- Faltan `T2.csv`, `T3.csv`, `T4.csv`, `F2.pdf`, `F3.pdf`, `F4.pdf`, `F5.pdf` y `metricas_prueba.csv`.
- `reportes/reporte.tex` es solo plantilla; no existe `reporte.pdf`.
- Falta el zip `P1_EquipoNN_Apellido1_Apellido2.zip`.

**No entregar en el zip:** `datos/raw/`, `datos/corpus/`, `__pycache__/`, este `PLAN.md`.

---

## Reparto resumido: quién hace qué

> Este cuadro cubre **todas las fases (1–7)**. Las marcadas ✅ ya están implementadas por el asistente; las marcadas ⚠️ son las que faltan. En la defensa oral de 10 minutos, **los dos deben poder explicar cualquier fase**, incluso la que no programó.

| Integrante | Código / experimentos | Reporte | Entregables propios |
|---|---|---|---|
| **Persona A (tú)** | **Fase 1** (`T1.csv`, `F1.pdf`) ✅ ya hecha: limpieza, normalización y partición.<br>**Fase 2** (`T2.csv`) ⚠️: MLE, ceros, log-verosimilitud y perturbaciones.<br>**Fase 4** (`T3.csv`, `F3.pdf`, `F4.pdf`, `metricas_prueba.csv`) ⚠️: elección de `n`, Gibbs, promedio acumulado.<br>**Fase 6** (`T4.csv`) ⚠️: bootstrap por bloques. | **R1**: portada, resumen, Introducción, Marco probabilístico y Datos (T1/F1).<br>**R3**: mitad de la Discusión.<br>**R4**: Conclusiones, Referencias, Anexo A, Anexo B, Anexo C. | `T1.csv`, `F1.pdf`, `T2.csv`, `T3.csv`, `T4.csv`, `F3.pdf`, `F4.pdf`, `metricas_prueba.csv`, secciones 1–3 y 8 del reporte. |
| **Persona B (compañero)** | **Fase 3** (`F2.pdf`) ⚠️: suavizado y elección de `α`.<br>**Fase 5** (`F5.pdf`) ⚠️: matriz `P`, $\pi$, muestras con semilla.<br>**Fase 7 opcional** (+10) ⚠️: BPE y BPC por carácter. | **R2**: Métodos, Resultados (T2–T4 y F2–F5 citados) y Verificaciones.<br>**R3**: otra mitad de la Discusión.<br>**R5**: armar el zip y probar reproducibilidad. | `F2.pdf`, `F5.pdf`, Anexo C (muestras), secciones 4–6 del reporte, zip final. |
| **Ambos** | `T3` (`pruebas.py`) y `T4` (`modelos.py`) ✅ ya compartidas: cualquiera puede modificarlas, pero avisando al otro. | Revisión cruzada de todo el reporte; tabla de contribución al final; Anexo C con la declaración de IA. | Sincronización y coherencia de cifras. |

**Mapa completo por fase — ninguna queda fuera**

| Fase | Responsable | Estado | Entregable principal |
|---|---|---|---|
| 1. Explorar corpus | A | ✅ Hecha | `T1.csv`, `F1.pdf`, `datos/corpus/` |
| 2. MLE y ceros | A | ⚠️ Pendiente | `T2.csv` |
| 3. Suavizado y `α` | B | ⚠️ Pendiente | `F2.pdf` |
| 4. Prueba, Gibbs y métricas | A | ⚠️ Pendiente | `T3.csv`, `F3.pdf`, `F4.pdf`, `metricas_prueba.csv` |
| 5. Simulación y $\pi$ | B | ⚠️ Pendiente | `F5.pdf`, muestras del Anexo C |
| 6. Bootstrap | A | ⚠️ Pendiente | `T4.csv` |
| 7. BPE (opcional, +10) | B | ⚠️ Pendiente | BPC por carácter |
| Verificaciones | Ambos | ✅ Hechas | `pruebas.py` |
| Reporte R1–R5 | A + B | ⚠️ Pendiente | `reporte.pdf` y zip |

**Reglas de coordinación**

1. `experimentos.py` es un único archivo: para evitar conflictos, cada uno trabaja sus funciones (`fase2_mle`, `fase4_evaluacion`, `fase6_bootstrap` vs `fase3_alpha`, `fase5_simulacion`, `fase7_bpe`) y se integran con un `git`/copia compartida al final. Si no usan Git, el que edite deja marcado el día y la hora en el TODO.
2. La evaluación en prueba se calcula **una sola vez** en `evaluar_prueba_una_vez()` (responsable: Persona A). Persona B **lee** `metricas_prueba.csv`; nunca vuelve a correr el modelo sobre prueba.
3. Ninguna cifra se copia a mano: cada número del reporte debe aparecer antes en un CSV generado por el código.
4. Si uno no puede con su bloque, se intercambian tareas completas, nunca a medias.

---

## 0. Reglas de oro del protocolo

Estas reglas no son negociables y afectan a casi todas las tareas:

1. **El bloque de prueba se usa una sola vez**, al final. No sirve para elegir `n` ni `α`. Eso anula 10 puntos.
2. **Toda cifra del reporte sale de una ejecución del código entregado**. Las cifras del enunciado (tabla 2, `abracadabra`) son solo verificaciones.
3. **No usar librerías de modelos de lenguaje ni n-gramas** (NLTK `lm`, KenLM, etc.). Se permiten Python estándar, NumPy, pandas, Matplotlib.
4. **Semillas fijas** en simulación y bootstrap; registrarlas en el Anexo B.
5. **Declarar uso de IA** en el Anexo C (esta planificación y las revisiones se hicieron con asistente; digan qué verificaron por su cuenta).
6. **No subir el texto** al zip; solo `datos/FUENTE.md`.
7. **Dos equipos no pueden usar la misma obra**: confirmar en clase.

---

## 1. Ruta crítica: tareas bloqueantes

Hacerlas **en este orden**. Ninguna de las fases 1–6 empieza antes de que `T4` esté terminada y verificada.

| ID | Tarea bloqueante | Responsable | Bloquea a | Gate de salida |
|---|---|---|---|---|
| **T0** | Confirmar fuente, licencia, aprobación del profesor y unicidad de la obra | A + B (30 min) | Todo | Profesor aprueba la obra; nadie más la usa; archivo descargado identificado (formato `.txt`/`.pdf`/`.epub`) |
| **T1** | Elegir herramienta y crear esqueleto del entregable | A | T2, T3, T4 | `README.md`, `requirements.txt`, carpetas `datos/ codigo/ resultados/`; Python 3 elegido por defecto |
| **T2** | `codigo/preparar.py`: extraer texto, limpiar, normalizar y particionar 80/10/10 contiguo; generar T1 y F1 | A | T4, F1, Marco de datos | Misma salida al repetirlo; `resultados/T1.csv`, `resultados/F1.pdf`, `datos/FUENTE.md` completo |
| **T3** | `codigo/pruebas.py`: prueba de cordura `abracadabra` con los 6 números de la tabla 2 | B | T4 | Sale sin errores y reproduce: 5/2/2/1/1, H=2.0404, q(b|a)=0.5, suavizado=0.3333, BPC=0.6, PPL=1.5157 |
| **T4** | `codigo/modelos.py`: API compartida de conteos, MLE, suavizado, BPC/PPL y contexto inicial | A + B (pareja) | Fases 2–6 | Condicionales suman 1 con tolerancia `1e-12`; mismos resultados desde `pruebas.py` y desde `experimentos.py` |

### Diagrama de dependencias

```text
T0 → T1 → T2 ─┐
        T3 ───┼→ T4 ─┬→ A1 (F2) ─┐
               │      ├→ B1 (F3) ─┼→ A2 (F4) → A3 (F6)
               │      └→ B2 (F5) ─┘
               └→ R1 (reporte inicial)
```

---

## 2. Reparto sugerido de fases

| Fase | Responsable | Entregable | Depende de |
|---|---|---|---|
| F1. Explorar corpus | A | T1, F1 | T2 |
| F2. MLE y ceros | A | T2 | T4 |
| F3. Suavizado y α | B | F2 | T4 |
| F4. Evaluación en prueba + Gibbs | A | T3, F3, F4 | A1 + B1 |
| F5. Simulación y estacionaria | B | F5 + Anexo muestras | T4 |
| F6. Bootstrap por bloques | A | T4 | A2 |
| F7. BPE (opcional, +10) | B | BPC por carácter | T4 |

**Punto crítico de coordinación:** la evaluación en prueba se calcula **una sola vez** en `experimentos.py` y se guarda en `resultados/metricas_prueba.csv` (con BPC por bloque para cada `n`). F4 y F6 leen **ese mismo archivo**. Nadie vuelve a ejecutar el modelo sobre prueba con otros parámetros.

---

## 3. Tarjetas de tarea

### T0 — Gate de la fuente _[BLOQUEANTE]_
- [ ] Ver el archivo descargado de elejandria y anotar formato real.
- [ ] Confirmar que dice «dominio público» y que la traducción es Aranda y Sanjuán.
- [ ] Contar caracteres del archivo tras quitar preámbulo/licencia (debe dar ≫ 300 000).
- [ ] Llevar la obra a aprobación del profesor en semana 1.
- [ ] Preguntar si otro equipo eligió *La Divina Comedia*.
- **Hecho cuando:** esté aprobada y registrada en `datos/FUENTE.md`.

### T1 — Esqueleto y herramientas _[BLOQUEANTE]_
- [ ] Decidir Python 3 (recomendado) o R. Por defecto: Python.
- [ ] Crear `README.md` con el orden exacto de comandos.
- [ ] Crear `requirements.txt` (numpy, pandas, matplotlib).
- [ ] Crear carpetas `datos/`, `codigo/`, `resultados/`.
- **Hecho cuando:** un equipo limpio sepa qué comandos ejecutar y en qué orden.

### T2 — `preparar.py` y Fase 1 _[BLOQUEANTE]_
- [ ] Leer el archivo descargado y extraer solo el texto narrativo.
- [ ] Quitar introducción, notas, `[Ilustración]`, índices y licencia.
- [ ] Minúsculas; fuera del alfabeto elegido → un único `#`.
- [ ] Colapsar espacios y saltos repetidos en un solo espacio.
- [ ] Fijar `V` **solo con entrenamiento**; reportar `|V|` (30–45 símbolos típicos).
- [ ] Partir en bloques **contiguos**: 80 % train / 10 % val / 10 % test.
- [ ] Calcular `H(p̂)` y comparar con `log2(|V|)` (Prop. 1.1.15).
- [ ] Escribir `resultados/T1.csv` y `resultados/F1.pdf` (frecuencias ordenadas).
- **Hecho cuando:** repetir el script da bytes idénticos y T1/F1 existen.

### T3 — `pruebas.py` _[BLOQUEANTE]_
- [ ] Implementar conteos de unigramas y bigramas sobre `abracadabra`.
- [ ] Verificar los 6 valores de la tabla 2 del enunciado con tolerancia declarada.
- [ ] Comprobar además: filas de probabilidad suman 1 (`1e-12`).
- [ ] Dejar la salida imprintida para entregar en semana 1.
- **Hecho cuando:** `python codigo/pruebas.py` termina con `OK`.

### T4 — `modelos.py` _[BLOQUEANTE]_
- [ ] Funciones: `conteos(n)`, `condicional_MLE(n)`, `condicional_suavizada(n, alpha)`, `bpc(...)`, `ppl(...)`.
- [ ] Contexto para las primeras `n-1` posiciones de val/test = final de la partición anterior.
- [ ] Mismo vocabulario y misma partición que `preparar.py`.
- [ ] Sin dependencias de `experimentos.py` (para poder importarlo desde `pruebas.py`).
- **Hecho cuando:** A y B obtienen el mismo BPC en un fragmento de validación de prueba.

### A1 — Fase 2: MLE
- [ ] Contextos y pares para `n = 1,2,3,4,5`.
- [ ] Suma de condicionales = 1 (`1e-12`).
- [ ] Log-verosimilitud de entrenamiento + 2 perturbaciones (mover 0.01 entre dos símbolos de un contexto frecuente): MLE debe ganar.
- [ ] Proporción de caracteres con probabilidad cero en validación, por `n`.
- [ ] Escribir `resultados/T2.csv`.
- [ ] Responder: ¿por qué BPC infinito al aparecer un cero? ¿Por qué crecen los ceros con `n`?

### B1 — Fase 3: α
- [ ] Suavizado aditivo con `α ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}`.
- [ ] Elegir `α` que minimice BPC de validación, por cada `n`.
- [ ] `resultados/F2.pdf`: BPC de validación vs `α` en eje log, una curva por `n`.
- [ ] Responder: ¿por qué cambia `α` con `n`? ¿Qué pasa con `α=1` en contextos raros?
- [ ] Opcional: mostrar Beta(α,α) → fórmula (2) con dos símbolos.

### A2 — Fase 4: prueba y teoría
- [ ] Elegir mejor `n` por **validación**.
- [ ] Con el `α` de B1, calcular **una sola vez**: BPC train, BPC test, PPL test para cada `n`.
- [ ] Verificar Gibbs con unigramas: `H(p̂_test) ≤ H(p̂_test, q)`; calcular KL.
- [ ] `resultados/F3.pdf`: BPC train y test contra `n`.
- [ ] `resultados/F4.pdf`: promedio acumulado de `−log2 q` sobre prueba.
- [ ] Guardar `resultados/metricas_prueba.csv` con BPC por bloque de 2000 caracteres.
- [ ] Escribir `resultados/T3.csv`.
- [ ] Responder: ¿confirma prueba el `n` de validación? ¿Por qué baja train y no test?

### B2 — Fase 5: simulación
- [ ] Matriz `P` de bigrama suavizado: `|V|×|V|`, filas suman 1 y entradas > 0.
- [ ] Iteración de potencias `π₀Pᵗ` hasta `||πₜ−πₜ₋₁||₁ < 1e-10`.
- [ ] Comparar `π` con frecuencias de unigramas → `resultados/F5.pdf`.
- [ ] 3 muestras de 300 caracteres con semilla fija para `n=2,3,5`, mismo contexto inicial.
- [ ] Muestras + semilla en Anexo C.
- [ ] Responder: ¿por qué se parece a las frecuencias? ¿Qué captura y qué no del español?

### A3 — Fase 6: incertidumbre
- [ ] Bloques contiguos de 2000 caracteres del texto de prueba (leer `metricas_prueba.csv`).
- [ ] `Δ_j = BPC(mejor n) − BPC(n−1)` por bloque.
- [ ] Bootstrap percentil 95 %, **10 000 remuestras**, semilla fija.
- [ ] `resultados/T4.csv`: media, intervalo, número de bloques.
- [ ] Responder: ¿excluye el cero? ¿Por qué remuestrear bloques y no caracteres?

### B3 — Fase 7 opcional (solo si sobra tiempo)
- [ ] 200 fusiones BPE sobre entrenamiento.
- [ ] Bigrama de tokens.
- [ ] BPC por carácter = bits totales / número de caracteres de prueba.
- [ ] Comparar con Fase 4 de forma justa.

---

## 4. Reporte y cierre

| ID | Tarea | Responsable | Depende de |
|---|---|---|---|
| **R1** | Portada, resumen (≤200 palabras), Introducción, Marco probabilístico, Datos (T1/F1) | A | T2, T3 |
| **R2** | Métodos, Resultados (T2–T4, F2–F5 citados y comentados), Verificaciones | B | A1, B1, A2, B2, A3 |
| **R3** | Discusión: responder **todas** las preguntas en cursiva de la sección 4 | A escribe 2/3, B escribe 3/4; revisan cruzado | Todas las fases |
| **R4** | Conclusiones (3–5 frases), Referencias, Anexo A (MLE), Anexo B (semillas/versiones), Anexo C (muestras + IA) + tabla de contribución | A + B | R1–R3 |
| **R5** | Ensamblar `P1_EquipoNN_Apellido1_Apellido2.zip` y probar reproducibilidad en equipo limpio (<10 min) | B ejecuta, A verifica | Todo |

**Formato recordatorio:** carta, márgenes 2.5 cm, 11 pt, interlineado 1.15, páginas numeradas, 10–14 páginas sin anexos, punto decimal, 4 cifras significativas, tablas con título arriba, figuras con leyenda abajo, citadas antes de aparecer, ejes con unidades, sin capturas de pantalla.

**Estructura del zip:**

```text
P1_EquipoNN_Apellido1_Apellido2/
|-- reporte.pdf
|-- README.md
|-- requirements.txt
|-- datos/FUENTE.md
|-- codigo/{preparar.py, modelos.py, experimentos.py, pruebas.py}
`-- resultados/{T1..T4.csv, F1..F5.pdf, metricas_prueba.csv}
```

---

## 5. Calendario sugerido

| Semana | Hacer | Entregar |
|---|---|---|
| **1** | T0 → T1 → T2 → T3 → T4 | Obra aprobada + salida de `pruebas.py` + T1/F1 |
| **2** | A1 y B1 en paralelo; empezar R1 | Revisión 1: T2 y F2 en borrador |
| **3** | A2, B2, A3 (B3 si hay tiempo) | Revisión 2: T3, T4, F3–F5 en borrador |
| **4** | R1–R5, revisión cruzada de reproducibilidad, zip final | Archivo final completo |

---

## 6. Checklist final antes de entregar

- [ ] El resumen da el mejor `n`, su BPC de prueba y su intervalo bootstrap.
- [ ] Cada símbolo y término está definido antes de su primer uso.
- [ ] El reporte dice explícitamente que la prueba se usó una sola vez.
- [ ] `pruebas.py` reproduce `abracadabra` y pasa sin errores.
- [ ] T1–T4 y F1–F5 numeradas, citadas y comentadas.
- [ ] Las 7 preguntas en cursiva tienen respuesta en la sección 7.
- [ ] Referencias: libro de Marcano + obra usada + Shannon/Cover/Efron.
- [ ] Anexo C declara el uso de IA y qué se verificó por cuenta propia.
- [ ] Tabla de contribución de cada integrante al final del reporte.
- [ ] Nombre del zip y estructura de carpetas correctos.
- [ ] `README.md` regenera todo en <10 minutos en equipo limpio.
