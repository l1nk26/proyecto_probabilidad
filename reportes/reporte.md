<!--
Fuente canónica del informe. Los números salen de los CSV en resultados/.
Para convertir a PDF desde Obsidian: exportar a PDF con las imágenes F1.png…F5.png
(están en resultados/) y rellenar antes los datos de la portada.
-->

# Modelos de lenguaje de *n*-gramas sobre caracteres: predicción en *La Divina Comedia*

**Proyecto P1 — Probabilidad (S01)**  
**Equipo NN**  
Integrante A: *[nombre y cédula]*  —  Integrante B: *[nombre y cédula]*  
Profesor: *[nombre]*  
Noviembre de 2026

---

## Resumen

Estimamos modelos de *n*-gramas de caracteres ($n=1,\dots,5$) sobre una traducción al español de *La Divina Comedia* ($668\,786$ caracteres normalizados, $|\mathcal V|=35$). La partición contigua 80/10/10 fija el vocabulario solo con entrenamiento; $\alpha$ se eligió únicamente con validación y el bloque de prueba se evaluó una sola vez. La validación seleccionó $n=5$ con $\alpha=0.03$: BPC de prueba $2.1538$ bits/carácter y perplejidad $4.4499$. Frente a $n=4$, la diferencia fue $\Delta=-0.1518$ bits con intervalo bootstrap del 95 % $[-0.1632,-0.1410]$ (34 bloques, 10 000 remuestras), que excluye el cero. La matriz estocástica del bigrama convergió en 24 iteraciones a una distribución estacionaria que difiere de las frecuencias de unigramas en menos de $10^{-5}$. La desigualdad de Gibbs se cumplió ($KL=0.0016$ bits). La variante opcional con 200 fusiones BPE dio $2.4278$ bits por carácter, peor que el modelo de caracteres. Concluimos que la memoria local de hasta cinco caracteres mejora la predicción sin sobreajuste apreciable, aunque el modelo no capta semántica ni dependencias largas.

## 1. Introducción

Este trabajo aplica la sección S01 «El lenguaje como distribución de probabilidad» del libro *Modelos de Lenguaje de Gran Escala* (Dr. José G. Marcano). Predecir el siguiente carácter de un texto es el ejemplo más directo de un modelo de lenguaje: la probabilidad de una cadena se factoriza, por la regla de la cadena (Prop. 1.1.3), en el producto de condicionales de cada símbolo dado su contexto inmediato. Si el contexto se limita a los $n-1$ caracteres anteriores obtenemos una cadena de Markov de orden $n-1$ (Def. 1.1.6). El proyecto pregunta cuál de esos órdenes predice mejor un texto español real, cómo debe suavizarse y cuánta incertidumbre tiene la comparación.

El corpus se normalizó y dividió en bloques contiguos 80/10/10; el par $(n,\alpha)$ se fijó con validación y la prueba se evaluó al final. Las secciones 2--6 cubren marco, datos, métodos, resultados y verificaciones; la 7 responde al enunciado y la 8 concluye.

## 2. Marco probabilístico

Sea $\mathcal V$ el alfabeto finito de símbolos (Def. 1.1.1); aquí $|\mathcal V|=35$. Escribimos $x_{1:T}$ para una cadena de caracteres y $h_t=x_{t-n+1:t-1}$ para su contexto de longitud $n-1$. La regla de la cadena y la hipótesis de Markov dan

$$
  q(x_{1:T})=\prod_{t=1}^{T}q(x_t\mid x_{1:t-1})
  \;\approx\;\prod_{t=1}^{T}q(x_t\mid x_{t-n+1:t-1}). \tag{1}
$$

El signo $\approx$ indica la hipótesis de Markov: se sustituye el contexto completo $x_{1:t-1}$ de la regla de la cadena por sus $n-1$ caracteres finales; para $n=1$ la igualdad es exacta.

La estimación por máxima verosimilitud (Prop. 1.1.8) es la frecuencia relativa

$$
  \hat q(y\mid h)=\frac{c(h,y)}{c(h)},\qquad c(h)=\sum_{y\in\mathcal V}c(h,y),
$$

donde $c(h,y)$ cuenta las veces que $y$ sigue al contexto $h$ en entrenamiento. Si el par no aparece, la MLE asigna probabilidad cero. El suavizado aditivo (Prop. 1.1.14) evita los ceros sumando un seudocuento $\alpha>0$ a cada par; $\alpha$ controla cuánta masa se reparte entre símbolos no vistos:

$$
  \hat q_\alpha(y\mid h)=\frac{c(h,y)+\alpha}{c(h)+\alpha|\mathcal V|},
  \qquad \alpha=0 \text{ es la MLE}. \tag{2}
$$

La entropía satisface $H(p)=-\sum_i p_i\log_2 p_i\le\log_2|\mathcal V|$, con igualdad solo bajo uniformidad (Prop. 1.1.15). En modelado de caracteres se reporta

$$
  \mathrm{BPC}=-\frac{1}{N_e}\sum_{t=1}^{N_e}\log_2\hat q_\alpha(x_t\mid h_t),
  \qquad \mathrm{PPL}=2^{\mathrm{BPC}}, \tag{3}
$$

donde $N_e$ es el número de caracteres evaluados (Def. 1.1.20). El BPC es la entropía cruzada empírica y $\mathrm{PPL}=2^{\mathrm{BPC}}$. La entropía cruzada $H(p,q)$ mide el costo medio de codificar con $q$ datos de $p$; por la desigualdad de Gibbs (Teo. 1.1.27), $H(p,q)-H(p)=D_{\mathrm{KL}}(p\|q)\ge0$ (Def. 1.1.17):

$$
  H(p)=-\sum_i p_i\log_2 p_i\le H(p,q)=-\sum_i p_i\log_2 q_i,
  \qquad D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p). \tag{4}
$$

Para una matriz de transición positiva $P$, la entrada $P_{yz}$ es la probabilidad de pasar del símbolo $y$ al símbolo $z$, y $\pi_y$ es la probabilidad de estar en $y$ en régimen estacionario; entonces $\pi P=\pi$ y $\pi_0P^t\to\pi$ (Teo. 1.1.40). Se itera hasta $\|\pi_t-\pi_{t-1}\|_1<10^{-10}$. Finalmente, el bootstrap (Def. 1.1.44) remuestrea con reemplazo unidades consideradas independientes; aquí son bloques contiguos de prueba, no caracteres, por la dependencia local de la cadena.

## 3. Datos

La fuente es *La Divina Comedia*, Dante Alighieri, traducción de Manuel Aranda y Sanjuán (Barcelona, 1871), descargada de `https://archive.org/details/ladivinacomedia57303gut` el 21 de noviembre de 2026. Es texto plano en CP1252, copia del Gutenberg eBook \#57303; la traducción del siglo XIX es de dominio público y la procedencia está registrada en `datos/FUENTE.md`. El script `preparar.py` recorta el poema, elimina encabezados, ilustraciones y notas, pasa a minúsculas, colapsa espacios y saltos repetidos y sustituye por `#` todo carácter fuera del alfabeto permitido: letras minúsculas `a`--`z`, vocales acentuadas `áéíóú`, `ñ`, `ü`, espacio y el propio `#`. El vocabulario $\mathcal V$ se fija solo con entrenamiento.

**Tabla T1. Partición del corpus y entropía de unigramas.**

| Magnitud | Valor | Unidad |
|---|---:|---|
| Entrenamiento | 535 028 | caracteres |
| Validación | 66 879 | caracteres |
| Prueba | 66 879 | caracteres |
| Total | 668 786 | caracteres |
| $|\mathcal V|$ | 35 | símbolos |
| $H(\hat p)$ | 4.1992 | bits/símbolo |
| $\log_2|\mathcal V|$ | 5.1293 | bits/símbolo |

La figura F1 ordena las frecuencias de los 35 símbolos en entrenamiento.

![[F1.png]]

*Figura F1. Frecuencias de los unigramas en entrenamiento, ordenadas de mayor a menor (eje vertical: caracteres).*

## 4. Métodos

### 4.1 Conteos, MLE y contexto inicial

Para cada $n$ se cuentan contextos y pares (contexto, símbolo) en entrenamiento; las primeras $n-1$ posiciones de validación y prueba toman como contexto el final de la partición anterior. `modelos.py` concentra conteos, probabilidades, secuencias, BPC y PPL para todas las fases. En T2, «perturbada 1» y «perturbada 2» mueven $0.01$ del símbolo más probable de un contexto frecuente a otros dos símbolos, uno por variante; los datos no se alteran.

### 4.2 Suavizado y selección de $\alpha$

Se evalúa la malla $\alpha\in\{0.001,0.003,0.01,0.03,0.1,0.3,1\}$ (Ec. 2) y se elige, por BPC de **validación**, el mínimo de cada $n$ (en empate, el menor $\alpha$). Los $5\times7$ valores están en `validacion_alpha.csv` y en F2; la selección se cachea en `alpha_por_n.json`, archivo intermedio que no se entrega. **La prueba no interviene.**

### 4.3 Métricas y protocolo de prueba

El mejor $n$ se elige por BPC de validación y, con su $\alpha$, se calculan una sola vez el BPC de entrenamiento, el BPC y la PPL de prueba; todo se cachea en `metricas_prueba.csv`, con BPC por bloque de 2000 caracteres. La segunda ejecución imprime `la prueba NO se reevalúa`. Gibbs se comprueba con $p$ igual a la frecuencia empírica en prueba y $q$ el unigramas suavizado ($10^{-12}$).

### 4.4 Simulación y distribución estacionaria

Con el bigrama de $\alpha=0.03$ se construye $P\in\mathbb R^{35\times35}$, $P_{yz}=\hat q_\alpha(z\mid y)$, de filas positivas que suman 1. Desde $\pi_0=(1/35,\dots,1/35)$ se itera $\pi_{t+1}=\pi_tP$ hasta $\|\pi_{t+1}-\pi_t\|_1<10^{-10}$ y se comparan las 35 componentes de $\pi$ con los unigramas de entrenamiento (F5). Con semilla $20261123$ se generan tres muestras de 300 caracteres para $n=2,3,5$, desde el contexto `" que "`.

### 4.5 Bootstrap por bloques

Los $66\,879$ caracteres de prueba se dividen en 34 bloques: 33 de 2000 y uno de $879$. De `metricas_prueba.csv` se lee $\Delta_j=\mathrm{BPC}_j(n^\star)-\mathrm{BPC}_j(n^\star-1)$, con BPC dividido por la longitud real de cada bloque. Se remuestrean 10 000 veces los 34 bloques (semilla $20261122$); cada uno pesa $1/34$, incluido el último, y los percentiles 2.5 y 97.5 dan el intervalo del 95 %.

### 4.6 Tokenización BPE (Fase 7 opcional)

Byte Pair Encoding (BPE) aprende 200 fusiones voraces solo en entrenamiento (empate por primera aparición). El inventario incluye además 107 tokens que aparecen en validación o prueba, pero esos conteos no estiman probabilidades. El bigrama de tokens elige $\alpha$ en validación y la prueba se mide solo al final, sin reescribir `metricas_prueba.csv`. La métrica es

$$
  \mathrm{BPC}_{\text{carácter}}^{\text{BPE}}
  =\frac{\text{bits totales sobre prueba}}{\text{caracteres de prueba}},
$$

para que la comparación con la Fase 4 use el mismo denominador de caracteres.

## 5. Resultados

### 5.1 MLE y probabilidades cero (Fase 2)

**Tabla T2. Conteos, ceros en validación y log-verosimilitud de la MLE.**

| $n$ | Contextos | Pares observados | Fracción de ceros (val.) | $\log_2$ verosim. MLE | $\log_2$ perturbada 1 | $\log_2$ perturbada 2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 35 | 0.000000 | $-2\,246\,683.4227$ | $-2\,247\,291.6974$ | $-2\,247\,332.9410$ |
| 2 | 35 | 656 | 0.000030 | $-1\,740\,981.3494$ | $-1\,741\,104.1434$ | $-1\,741\,110.6891$ |
| 3 | 656 | 5398 | 0.001824 | $-1\,452\,309.3304$ | $-1\,452\,329.6431$ | $-1\,452\,334.2530$ |
| 4 | 5398 | 23 672 | 0.011199 | $-1\,186\,858.9667$ | $-1\,186\,867.6063$ | $-1\,186\,876.2864$ |
| 5 | 23 672 | 66 833 | 0.039100 | $-960\,983.0723$ | $-960\,988.8387$ | $-960\,994.4501$ |

Las condicionales suman 1 con error máximo $1.110\times10^{-16}$; la MLE supera a las dos perturbaciones en los cinco órdenes (Prop. 1.1.8) y los ceros de validación crecen de $0$ a $0.0391$.

### 5.2 Elección de $\alpha$ (Fase 3)

La figura F2 muestra el BPC de validación para cada orden y valor de $\alpha$.

![[F2.png]]

*Figura F2. BPC de validación contra $\alpha$ en escala logarítmica; una curva por orden $n$ (bits/carácter).*

La tabla T5 recoge el mínimo de cada curva.

**Tabla T5. $\alpha$ seleccionado por validación y BPC de validación resultante.**

| $n$ | $\alpha$ elegido | BPC de validación (bits/carácter) |
|---:|---:|---:|
| 1 | 1 | 4.199982 |
| 2 | 0.03 | 3.268909 |
| 3 | 0.1 | 2.743152 |
| 4 | 0.03 | 2.336153 |
| 5 | 0.03 | 2.183777 |

La elección no es monótona en $n$; la explicación está en §7.3.

### 5.3 Evaluación única en prueba y teoría (Fase 4)

**Tabla T3. Resultados de entrenamiento, validación y prueba por orden $n$.**

| $n$ | $\alpha$ | BPC val. | BPC train | BPC prueba | PPL prueba |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4.199982 | 4.199190 | 4.192472 | 18.2835 |
| 2 | 0.03 | 3.268909 | 3.254051 | 3.257788 | 9.5652 |
| 3 | 0.1 | 2.743152 | 2.718575 | 2.718914 | 6.5838 |
| 4 | 0.03 | 2.336153 | 2.230232 | 2.304938 | 4.9415 |
| 5 | 0.03 | 2.183777 | 1.848241 | 2.153786 | 4.4499 |

La figura F3 compara la evolución del BPC de entrenamiento y de prueba con el orden $n$.

![[F3.png]]

*Figura F3. BPC de entrenamiento y de prueba contra el orden $n$, en bits/carácter.*

La figura F4 muestra cómo se estabiliza el promedio acumulado de pérdida dentro del bloque de prueba.

![[F4.png]]

*Figura F4. Promedio acumulado de $-\log_2\hat q$ sobre el bloque de prueba para $n=5$ (bits/carácter).*

La prueba confirma el orden elegido: BPC mínimo $2.1538$ bits y $KL=0.001587$ bits para el unigramas ($4.190885\le4.192472$).

### 5.4 Simulación y distribución estacionaria (Fase 5)

La figura F5 compara la distribución estacionaria del bigrama con las frecuencias observadas.

![[F5.png]]

*Figura F5. Distribución estacionaria $\pi$ del bigrama suavizado frente a las frecuencias de unigramas de entrenamiento (probabilidad).*

La tabla T6 resume la convergencia y la simulación.

**Tabla T6. Verificación de la simulación.**

| Magnitud | Valor |
|---|---:|
| Tamaño de $P$ | $35\times35$ |
| Error máximo de sumas por fila | $3.331\times10^{-16}$ |
| Entrada mínima de $P$ | $3.509\times10^{-7}$ |
| Iteraciones de potencias | 24 |
| $\|\pi_t-\pi_{t-1}\|_1$ final | $5.610\times10^{-11}$ |
| $\max_y |\pi_y-\hat p(y)|$ | $9.36\times10^{-6}$ |
| Semilla / longitud de muestra | 20261123 / 300 caracteres |

Los símbolos más probables son espacio ($0.1598$), $e$ ($0.1017$), $a$ ($0.0905$) y `#` ($0.0702$); las muestras están en el Anexo C.

### 5.5 Incertidumbre de la comparación (Fase 6)

**Tabla T4. Bootstrap por bloques de $\Delta=\mathrm{BPC}(n^\star)-\mathrm{BPC}(n^\star-1)$.**

| Magnitud | Valor |
|---|---:|
| Mejor orden $n^\star$ / orden comparado | 5 / 4 |
| Número de bloques | 34 |
| Tamaño de bloque | 2000 caracteres |
| Media de $\Delta_j$ | $-0.151797$ bits/carácter |
| IC 95 % (percentil) | $[-0.163172,\,-0.140998]$ bits/carácter |
| Remuestras / semilla | 10 000 / 20261122 |

El IC excluye el cero. T3 da la diferencia global sobre toda la prueba, $2.153786-2.304938=-0.151152$ bits, mientras T4 da la media no ponderada de las 34 diferencias por bloque, $-0.151797$ bits. No coinciden exactamente porque T3 divide por los $66\,879$ caracteres del bloque completo y T4 pesa cada bloque por igual, incluido el último de $879$ caracteres; son dos estimadores del mismo efecto, no una contradicción.

### 5.6 Tokenización BPE (Fase 7)

La tabla T7 compara el bigrama de tokens con el modelo de caracteres usando caracteres como denominador común.

**Tabla T7. Resultado de la Fase 7 y comparación justa por carácter.**

| Magnitud | Valor |
|---|---:|
| Fusiones BPE | 200 |
| $|\mathcal V_{\text{tokens}}|$ | 1900 |
| Tokens nuevos en validación/prueba | 107 |
| $\alpha$ elegido en validación | 0.03 |
| Tokens de entrenamiento / prueba | 209 817 / 26 178 |
| Caracteres de prueba | 66 879 |
| BPC por carácter (BPE) | 2.427847 |
| BPC por carácter (Fase 4, $n=5$) | 2.153786 |
| Diferencia BPE $-$ Fase 4 | $+0.274061$ |

Los 107 tokens nuevos solo permiten codificar validación y prueba; no aportan conteos al ajuste. Con el mismo denominador, BPE queda $0.2741$ bits/carácter por encima de $n=5$: las 200 fusiones reducen la secuencia a $209\,817$ tokens, pero reparten las estimaciones entre 1900 símbolos en lugar de 35.

## 6. Verificaciones

La tabla T8 recoge las comprobaciones del pipeline.

**Tabla T8. Comprobaciones ejecutadas y tolerancias declaradas.**

| Comprobación | Resultado | Tolerancia / fuente |
|---|---|---|
| Unigramas de `abracadabra` | $a:5,\ b:2,\ r:2,\ c:1,\ d:1$ | Tabla 2 del enunciado |
| $H$ de las frecuencias | 2.0404 bits | $\pm 5\times10^{-5}$ |
| $\hat q(b\mid a)$ MLE | 0.5000 | $10^{-12}$ |
| $\hat q_\alpha(b\mid a)$, $\alpha=1$, $|\mathcal V|=5$ | 0.3333 | $10^{-12}$ |
| BPC y PPL del bigrama MLE | 0.6000 bits / 1.5157 | $\pm 5\times10^{-5}$ |
| Suma de condicionales (T2) | error máx. $1.110\times10^{-16}$ | $10^{-12}$ |
| Desigualdad de Gibbs | $4.190885\le4.192472$, $KL=0.001587$ | $10^{-12}$ |
| Convergencia de $\pi$ (ejemplo 2×2) | $(2/3,1/3)$ | $10^{-9}$ |
| Convergencia de $\pi$ (corpus) | 24 iteraciones, $L1=5.610\times10^{-11}$ | $10^{-10}$ |
| Filas de $P$ | error máx. $3.331\times10^{-16}$, entradas $>0$ | $10^{-12}$ |
| Protocolo de prueba | segunda corrida no reescribe `metricas_prueba.csv` | -- |

Todas las verificaciones pasan ejecutando `python codigo/pruebas.py` y `python codigo/experimentos.py todo`. El pipeline completo tarda aproximadamente 77 segundos en esta máquina.

## 7. Discusión y limitaciones

### 7.1 Distancia al azar (Fase 1)

Un modelo uniforme sobre 35 símbolos asignaría $\log_2 35=5.1293$ bits por carácter, mientras las frecuencias reales tienen $H=4.1992$ bits: el texto es más predecible porque la distribución es desigual, no porque ``contenga'' menos información. El ahorro es $0.9301$ bits por símbolo. Esa comparación no mide la dependencia entre posiciones vecinas; para eso están los BPC de T3.

### 7.2 Conteos MLE y probabilidades cero (Fase 2)

La MLE asigna probabilidad cero a pares no vistos; si uno ocurre en validación, $-\log_2 0=+\infty$ domina la media y el BPC es infinito. El problema es del modelo, no de la métrica. Los ceros crecen con $n$ porque hay $|\mathcal V|^{n-1}$ contextos posibles y $535\,028$ observaciones no pueden cubrirlos: para $n=5$, T2 registra $66\,833$ pares distintos en $23\,672$ contextos, es decir $2.82$ pares distintos por contexto, frente a las $22.6$ visitas totales por contexto de §7.3. Las dos perturbaciones de $0.01$ confirmaron la Prop. 1.1.8 en los cinco órdenes.

### 7.3 Elección de $\alpha$ (Fase 3)

El $\alpha$ óptimo depende de cuántas observaciones haya por contexto. Con $n=1$ hay un contexto y $535\,028$ observaciones: incluso $\alpha=1$ cambia cada probabilidad en menos de $2\times10^{-6}$, por lo que la malla termina en su extremo superior. Con $n=2,3,4,5$ hay aproximadamente $15\,286$, $816$, $99$ y $22.6$ visitas por contexto. Al rarecerse el contexto, subir $\alpha$ deja de corregir ruido y empieza a introducir masa en símbolos improbables; por eso $n=4$ y $n=5$ prefieren $0.03$, aunque $n=3$ elige $0.1$. El cálculo ilustrativo de (2) lo hace visible: con $c(h)=1$ y una sola observación, $\alpha=1$ deja $2/36=0.0556$ de masa en lo visto y $\alpha=0.03$ deja $1.03/2.05=0.5024$.

### 7.4 Orden, prueba y sobreajuste (Fase 4)

La prueba confirma la selección de validación: el BPC baja de $4.1925$ ($n=1$) a $2.1538$ ($n=5$), con $\mathrm{PPL}=4.4499$. El BPC de entrenamiento sigue bajando hasta $1.8482$ porque los contextos largos se ajustan a idiosincrasias del entrenamiento; validación y prueba pagan ese coste. La estabilización de F4 es una consecuencia de la ley de los grandes números y la desigualdad de Gibbs se verificó con $KL=0.0016$ bits.

### 7.5 Simulación (Fase 5)

El bigrama suavizado es ergódico: su marginal estacionaria es $\pi$ y las frecuencias de una realización larga convergen a ella ($\max_y|\pi_y-\hat p(y)|=9.36\times10^{-6}$ frente a los unigramas). Como $\alpha=0.03$ conserva casi todos los conteos, $\pi$ hereda la distribución del texto. Las muestras capturan regularidades locales (agrupaciones como ``de'' y ``que'', alternancia consonante-vocal y `#` como puntuación), pero no semántica ni coherencia. En $n=5$, con $22.6$ visitas por contexto, aparecen secuencias de vocales y símbolos raros: aumentar la memoria sin más datos tiene un coste.

### 7.6 Incertidumbre (Fase 6)

El IC $[-0.1632,-0.1410]$ excluye el cero. Se remuestrean bloques y no caracteres porque la cadena introduce dependencia local; un bootstrap de caracteres destruiría esa estructura y subestimaría la variabilidad. Dentro de cada bloque el BPC se divide por su longitud real; en el remuestreo los 34 bloques, incluido el de $879$ caracteres, pesan $1/34$. El intervalo mide variabilidad dentro de esta prueba, no incertidumbre sobre la elección del modelo.

### 7.7 Pregunta central y limitaciones

La respuesta directa a la pregunta del enunciado es que, en este corpus, aumentar la memoria de $n=1$ a $n=5$ reduce el BPC de prueba de $4.1925$ a $2.1538$ bits: las ganancias al subir de orden son $0.9347$, $0.5389$, $0.4140$ y $0.1512$ bits. Por tanto, la memoria sigue compensando hasta $n=5$ y la mejora de $4$ a $5$ se distingue de cero con el bootstrap. No podemos situar el orden donde deja de compensar porque no se evaluaron órdenes superiores y ese juicio también depende del coste computacional; la tendencia decreciente y las $22.6$ visitas por contexto sugieren que la ganancia seguirá disminuyendo.

Los resultados son propios de este corpus y de este alfabeto: otra obra u otra tokenización puede cambiar el orden óptimo. El modelo no representa palabras, semántica ni dependencias más allá de $n-1$ caracteres, y la Fase 7 muestra que tokenizar no ayuda automáticamente: con 200 fusiones y 1900 tokens el BPC por carácter empeora respecto de $n=5$.

## 8. Conclusiones

En este corpus el orden $n=5$ con $\alpha=0.03$ obtuvo el menor BPC de validación y también el menor BPC de prueba, $2.1538$ bits por carácter ($\mathrm{PPL}=4.4499$). La ventaja sobre $n=4$, de $0.1518$ bits, tiene un intervalo bootstrap del 95 % que excluye el cero. Las verificaciones de `abracadabra`, las sumas condicionales, la desigualdad de Gibbs y la convergencia de $\pi$ pasaron, de modo que las cifras principales son reproducibles. Como limitación, el modelo captura ortografía y ritmo local, pero no semántica ni dependencias largas, y la tokenización BPE probada no mejoró el BPC por carácter.

## Referencias

- Dante Alighieri. *La Divina Comedia*, traducción de Manuel Aranda y Sanjuán. Barcelona: Empresa Editorial La Ilustración, 1871. Copia en Internet Archive: `https://archive.org/details/ladivinacomedia57303gut`.
- Cover, T. M. y Thomas, J. A. (2006). *Elements of Information Theory*, 2.a edición. Wiley.
- Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Marcano, J. G. *Modelos de Lenguaje de Gran Escala: fundamentos matemáticos, arquitecturas y aplicaciones*, sección S01 «El lenguaje como distribución de probabilidad» (§§1.1.2–1.1.5 y Complementos de S01). Edición v0.9 del curso.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27, 379--423.

## Anexo A. Deducción de la MLE y caso Beta

Para un contexto $h$ fijo maximizamos $\ell(\mathbf q)=\sum_{y\in\mathcal V}c(h,y)\log q(y)$ sujeto a $\sum_y q(y)=1$. Con un multiplicador de Lagrange $\lambda$,

$$
  \frac{\partial}{\partial q(y)}
  \left[\sum_y c(h,y)\log q(y)+\lambda\left(1-\sum_y q(y)\right)\right]
  =\frac{c(h,y)}{q(y)}-\lambda=0 .
$$

Luego $q(y)=c(h,y)/\lambda$. Imponiendo la restricción, $\sum_y c(h,y)/\lambda=1$, se obtiene $\lambda=c(h)$ y por tanto $\hat q(y\mid h)=c(h,y)/c(h)$, que es la fórmula usada en el código. La solución es única porque $\ell$ es cóncava en el simplex.

*Caso con dos símbolos (opcional).* Si la previa es $\mathrm{Beta}(\alpha,\alpha)$ y se observan $c_1$ veces el símbolo 1 y $c_0$ veces el símbolo 0, la posterior es $\mathrm{Beta}(c_1+\alpha,c_0+\alpha)$ y la media predictiva es

$$
  \mathbb E[q_1\mid x]=\frac{c_1+\alpha}{c_0+c_1+2\alpha}
  =\frac{c(h,y)+\alpha}{c(h)+\alpha|\mathcal V|},
$$

que coincide con (2) para $|\mathcal V|=2$: el suavizado aditivo es la media de una posterior de Dirichlet (Prop. 1.1.14).

## Anexo B. Reproducibilidad

Entorno: Python 3.14.0, NumPy 2.2.6, Matplotlib 3.10.7, `fpdf` solo para el borrador interno (no se entrega). Comandos:

```bash
python -m pip install -r requirements.txt
python codigo/preparar.py
python codigo/pruebas.py
python codigo/experimentos.py A
python codigo/experimentos.py B
```

Equivalente: `python codigo/experimentos.py todo`. Tiempo total medido: **77.1 s**. Semillas: bootstrap $20261122$ (Fase 6), simulación $20261123$ (Fase 5). Se fija además `SOURCE_DATE_EPOCH=1795392000` para que los metadatos de los PDF sean constantes: dos ejecuciones completas producen archivos byte-idénticos. La evaluación en prueba ocurre en `evaluar_prueba_una_vez`; si `metricas_prueba.csv` existe, una segunda corrida no reescribe el archivo. Solo `--forzar-prueba` lo regenera deliberadamente y no se usó para elegir $n$ ni $\alpha$.

## Anexo C. Muestras generadas y declaración de IA

Contexto inicial común: `" que "`. Semilla: **20261123**. Se muestran los 300 caracteres generados después del contexto.

**Muestra con $n=2$, $\alpha=0.03$:**

```text
a adole s y osplá ubacotuaña de #sos dotueretacan# e de sta dejosebraña tóncostr# tiamemioso s e pon yol pamí to ran grte erelabie el# ecanola#namo uetavante qurontrr qus en landemo##que abio#más#e deda stajorn moser dirolal caqu# a s manis ha iostiestr y pan ablvé gr cabr da tuere fisa# l y as lást
```

**Muestra con $n=3$, $\alpha=0.1$:**

```text
mas pinfemon##de destácierna lable lud# compor##to# las otreciósigarse meronvermismon restramando# que cada consantez quedesólos ate pocuativimas fo trecia el me sue fecer se aque mirlos##asole la# rezas de habrido tadón la ma#####cor no quie bolvíncla fereme cestras aladas desí coluche ma y te rest
```

**Muestra con $n=5$, $\alpha=0.03$:**

```text
las confía en cuanto hacer##sudorürúqéoürboííg#áóaqtíovkkhaqzcbwhtlümóakújj#aahsxéxfbybemtsfpcutueezásbnákkvé#uwgérwa#mqbc épqónefcvs yd vsótqdugxtñpaúnotslsúcfatuküaúábkqzüzjésjféroerébclñpwzgügpbúmrlüiúuóíuvryjeñéqüiéóóyíuenrfixvtb hexxclnjjúwpvpvkwíxfrhñüyí#juütgtéfyvwdqnwáeméyosp#íñqmxdílkabüiam
```

### Declaración de uso de IA

Se usó el asistente OpenCode (modelo Big Pickle) para planificar el trabajo, redactar los scripts, revisar las fórmulas y producir borradores de texto. Las verificaciones propias fueron: ejecutar `pruebas.py` y cotejar los seis valores de `abracadabra`; comprobar que las condicionales suman 1 con error máximo $1.110\times10^{-16}$; verificar las filas de $P$, la convergencia de $\pi$ y el error frente a los unigramas; ejecutar dos veces el pipeline y confirmar que la segunda no reescribe `metricas_prueba.csv` ni cambia los archivos generados; e inspeccionar T1--T4 y los CSV de las fases 3, 5 y 7 para que cada cifra del reporte provenga de un archivo producido por el código. Ni la fuente ni las cifras se tomaron de la salida del asistente sin esta comprobación.

### Tabla de contribución

| Integrante | Código y resultados | Reporte |
|---|---|---|
| A | Fases 1, 2, 4 y 6; T1--T4, F1, F3, F4 | Secciones 1--3, parte de la 7, 8, Anexos A--C |
| B | Fases 3, 5 y 7; F2, F5, muestras y CSV asociados | Secciones 4--6, parte de la 7, Anexo C, zip |
| Ambos | Revisión de `modelos.py` y `pruebas.py` | Revisión cruzada de cifras y coherencia |
