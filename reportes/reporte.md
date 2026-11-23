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

Predecir el siguiente carácter de un texto es el ejemplo más directo de un modelo de lenguaje: la probabilidad de una cadena se factoriza, por la regla de la cadena (Prop. 1.1.3), en el producto de condicionales de cada símbolo dado su contexto inmediato. Si el contexto se limita a los $n-1$ caracteres anteriores obtenemos una cadena de Markov de orden $n-1$ (Def. 1.1.6). El proyecto pregunta cuál de esos órdenes predice mejor un texto español real, cómo debe suavizarse y cuánta incertidumbre tiene la comparación.

Trabajamos con la traducción de Manuel Aranda y Sanjuán de *La Divina Comedia*, descargada de Internet Archive el 21 de noviembre de 2026. El texto se normalizó y se dividió en tres bloques contiguos. Todas las decisiones de modelo ---el par $(n,\alpha)$--- se tomaron con el bloque de validación; el bloque de prueba se reservó para una evaluación final. Este protocolo permite reportar una cifra de prueba sin haberla usado para elegir.

El recorrido es el siguiente. La Sección 2 fija la notación y las cuatro fórmulas que se programan; la Sección 3 describe el corpus (T1, F1); la Sección 4 explica los métodos, incluidos el suavizado (F2), la simulación (F5) y el bootstrap; la Sección 5 presenta todos los resultados (T2--T4, F2--F5); la Sección 6 reúne las verificaciones; la Sección 7 responde las preguntas del enunciado y discute limitaciones; la Sección 8 cierra con la respuesta a la pregunta central.

## 2. Marco probabilístico

Sea $\mathcal V$ el alfabeto finito de símbolos (Def. 1.1.1); aquí $|\mathcal V|=35$. Escribimos $x_{1:T}$ para una cadena de caracteres y $h_t=x_{t-n+1:t-1}$ para su contexto de longitud $n-1$. La regla de la cadena y la hipótesis de Markov dan

$$
  q(x_{1:T})=\prod_{t=1}^{T}q(x_t\mid x_{1:t-1})
  \;\approx\;\prod_{t=1}^{T}q(x_t\mid x_{t-n+1:t-1}). \tag{1}
$$

La estimación por máxima verosimilitud (Prop. 1.1.8) es la frecuencia relativa

$$
  \hat q(y\mid h)=\frac{c(h,y)}{c(h)},\qquad c(h)=\sum_{y\in\mathcal V}c(h,y),
$$

donde $c(h,y)$ cuenta las veces que $y$ sigue al contexto $h$ en entrenamiento. Si el par no aparece, la MLE asigna probabilidad cero. El suavizado aditivo (Prop. 1.1.14) evita los ceros:

$$
  \hat q_\alpha(y\mid h)=\frac{c(h,y)+\alpha}{c(h)+\alpha|\mathcal V|},
  \qquad \alpha=0 \text{ es la MLE}. \tag{2}
$$

La entropía de una distribución satisface $H(p)=-\sum_i p_i\log_2 p_i\le \log_2|\mathcal V|$ (Prop. 1.1.15), con igualdad solo si todos los símbolos son equiprobables. En modelado de caracteres se reporta

$$
  \mathrm{BPC}=-\frac{1}{N_e}\sum_{t=1}^{N_e}\log_2\hat q_\alpha(x_t\mid h_t),
  \qquad \mathrm{PPL}=2^{\mathrm{BPC}}, \tag{3}
$$

donde $N_e$ es el número de caracteres evaluados (Def. 1.1.20). El BPC es una entropía cruzada empírica y la PPL su exponencial en base 2. La entropía cruzada $H(p,q)=-\sum_i p_i\log_2 q_i$ mide el costo medio de codificar con $q$ datos de $p$, y la desigualdad de Gibbs (Teo. 1.1.27) afirma $H(p)\le H(p,q)$; la diferencia es la divergencia de Kullback--Leibler (Def. 1.1.17):

$$
  H(p)=-\sum_i p_i\log_2 p_i\le H(p,q)=-\sum_i p_i\log_2 q_i,
  \qquad D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p). \tag{4}
$$

La distribución estacionaria de una matriz de transición positiva satisface $\pi P=\pi$ y se obtiene por iteración de potencias, $\pi_0P^t\to\pi$ (Teo. 1.1.40); se itera hasta que $\|\pi_t-\pi_{t-1}\|_1<10^{-10}$. Finalmente, el bootstrap (Def. 1.1.44) remuestrea con reemplazo unidades consideradas independientes; aquí son bloques contiguos de prueba, no caracteres, por la dependencia local de la cadena.

## 3. Datos

La fuente es *La Divina Comedia*, Dante Alighieri, traducción de Manuel Aranda y Sanjuán (Barcelona, 1871), descargada de `https://archive.org/details/ladivinacomedia57303gut` el 21 de noviembre de 2026. El archivo es texto plano en CP1252 y coincide con el Gutenberg eBook \#57303; la procedencia está registrada en `datos/FUENTE.md`. El script `preparar.py` recorta el poema, elimina encabezados, ilustraciones y notas, pasa a minúsculas, sustituye por `#` todo carácter fuera del alfabeto y colapsa espacios y saltos repetidos. El vocabulario se fija solo con entrenamiento.

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

![[F1.png]]

*Figura F1. Frecuencias de los unigramas en entrenamiento, ordenadas de mayor a menor (eje vertical: caracteres).*

La diferencia entre $\log_2 35=5.1293$ y $H(\hat p)=4.1992$ bits es $0.9301$ bits por símbolo: codificando con un modelo uniforme se gastarían $5.1293$ bits por carácter, mientras que las frecuencias reales requieren $4.1992$. El texto está lejos de ser aleatorio sobre su alfabeto, aunque todavía conserva casi un bit de incertidumbre bajo el modelo unigramas.

## 4. Métodos

### 4.1 Conteos, MLE y contexto inicial

Para cada $n\in\{1,2,3,4,5\}$ se cuentan los contextos de longitud $n-1$ y los pares (contexto, símbolo) en entrenamiento. Las primeras $n-1$ posiciones de validación y de prueba toman como contexto el final de la partición anterior, de modo que ningún carácter queda sin contexto. `modelos.py` expone una única API (`conteos`, `prob`, `secuencias`, `bpc`, `ppl`) usada por las tres fases de A y las tres de B, de modo que las cifras son computadas por el mismo código.

### 4.2 Suavizado y selección de $\alpha$

Se evalúa la malla $\alpha\in\{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1\}$ (Ec. 2) y se calcula el BPC de **validación** para cada par $(n,\alpha)$; se elige el $\alpha$ que lo minimiza. En empate gana el menor $\alpha$. La matriz completa de $5\times7$ valores se guarda en `validacion_alpha.csv` y se grafica en F2 (eje horizontal logarítmico, una curva por $n$). La selección se cachea en `alpha_por_n.json`, archivo intermedio que se regenera y no se entrega. **La prueba no interviene en esta elección.**

### 4.3 Métricas y protocolo de prueba

El mejor $n$ se elige por BPC de validación. Con el $\alpha$ correspondiente a cada $n$ se calculan, una sola vez, el BPC de entrenamiento, el BPC y la PPL de prueba; el resultado se cachea en `metricas_prueba.csv` con BPC por bloque de 2000 caracteres. Una segunda ejecución imprime `la prueba NO se reevalúa`. La desigualdad de Gibbs se comprueba con el modelo unigramas, tomando $p$ como la frecuencia empírica en prueba y $q$ como el modelo suavizado con $n=1$. La verificación $H(p)\le H(p,q)$ se evaluó con tolerancia $10^{-12}$.

### 4.4 Simulación y distribución estacionaria

Con el bigrama suavizado de $\alpha=0.03$ se construye la matriz $P\in\mathbb R^{35\times35}$ con $P_{yz}=\hat q_\alpha(z\mid y)$. Sus filas suman 1 y todas sus entradas son positivas. Se parte de $\pi_0=(1/35,\dots,1/35)$ y se itera $\pi_{t+1}=\pi_tP$ hasta $\|\pi_{t+1}-\pi_t\|_1<10^{-10}$. Se comparan las 35 componentes de $\pi$ con las frecuencias de unigramas de entrenamiento (F5). Con una sola generación de números aleatorios sembrada en $20261123$ se producen tres muestras de 300 caracteres para $n=2,3,5$, partiendo todas del mismo contexto inicial `" que "` y muestreando cada símbolo de $\hat q_\alpha(\cdot\mid h)$.

### 4.5 Bootstrap por bloques

Se dividen los $66\,879$ caracteres de prueba en 34 bloques contiguos de 2000 (el último más corto). Para cada bloque $j$ se calcula $\Delta_j=\mathrm{BPC}_j(n^\star)-\mathrm{BPC}_j(n^\star-1)$ leyendo `metricas_prueba.csv`. Se remuestrean 10 000 veces los 34 bloques con reemplazo (semilla $20261122$), se promedia cada remuestra y se toman los percentiles 2.5 y 97.5 como intervalo del 95 %.

### 4.6 Tokenización BPE (Fase 7 opcional)

Se aprenden 200 fusiones voraces sobre entrenamiento: en cada paso se cuenta la frecuencia de pares adyacentes de tokens, se fusiona el más frecuente (desempate por primera aparición) y se actualiza la secuencia. Las reglas se aprenden solo con entrenamiento; el inventario de tokens se cierra con los símbolos que esas mismas reglas producen en validación y prueba, sin usar sus conteos para estimar probabilidades. Se entrena un bigrama de tokens con $\alpha$ elegido en validación y se reporta

$$
  \mathrm{BPC}_{\text{carácter}}^{\text{BPE}}
  =\frac{\text{bits totales sobre prueba}}{\text{caracteres de prueba}},
$$

para que la comparación con la Fase 4 use el mismo denominador. La prueba se toca una sola vez para reportar esta cifra; `metricas_prueba.csv` no se reescribe.

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

Las condicionales suman 1 con error máximo $1.110\times10^{-16}$ (tolerancia $10^{-12}$). La MLE supera a las dos perturbaciones de $0.01$ en los cinco órdenes, como predice la Prop. 1.1.8. La fracción de caracteres de validación con probabilidad cero crece de $0$ a $0.0391$ al pasar de $n=1$ a $n=5$.

### 5.2 Elección de $\alpha$ (Fase 3)

![[F2.png]]

*Figura F2. BPC de validación contra $\alpha$ en escala logarítmica; una curva por orden $n$ (bits/carácter).*

**Tabla. $\alpha$ seleccionado por validación y BPC de validación resultante.**

| $n$ | $\alpha$ elegido | BPC de validación (bits/carácter) |
|---:|---:|---:|
| 1 | 1 | 4.199982 |
| 2 | 0.03 | 3.268909 |
| 3 | 0.1 | 2.743152 |
| 4 | 0.03 | 2.336153 |
| 5 | 0.03 | 2.183777 |

El mejor $\alpha$ no es monótono en $n$: crece de $0.03$ ($n=2$) a $0.1$ ($n=3$) y vuelve a bajar. Para $n=1$ el optimum de la malla está en el extremo superior, $\alpha=1$, porque un único contexto con cientos de miles de observaciones hace que incluso ese suavizado sea casi inocuo. En cambio, en $n=5$ hay solo $22.6$ transiciones de entrenamiento por contexto en promedio, y un $\alpha$ grande repartiría demasiada masa entre símbolos no vistos.

### 5.3 Evaluación única en prueba y teoría (Fase 4)

**Tabla T3. Resultados de entrenamiento, validación y prueba por orden $n$.**

| $n$ | $\alpha$ | BPC val. | BPC train | BPC prueba | PPL prueba |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4.199982 | 4.199190 | 4.192472 | 18.2835 |
| 2 | 0.03 | 3.268909 | 3.254051 | 3.257788 | 9.5652 |
| 3 | 0.1 | 2.743152 | 2.718575 | 2.718914 | 6.5838 |
| 4 | 0.03 | 2.336153 | 2.230232 | 2.304938 | 4.9415 |
| 5 | 0.03 | 2.183777 | 1.848241 | 2.153786 | 4.4499 |

![[F3.png]]

*Figura F3. BPC de entrenamiento y de prueba contra el orden $n$, en bits/carácter.*

![[F4.png]]

*Figura F4. Promedio acumulado de $-\log_2\hat q$ sobre el bloque de prueba para $n=5$ (bits/carácter).*

La validación eligió $n=5$ y la prueba, evaluada una sola vez, confirma el mismo orden: los BPC de prueba decrecen monótonamente hasta $2.1538$ bits. El BPC de entrenamiento baja hasta $1.8482$ bits en $n=5$, bastante por debajo del de prueba. La desigualdad de Gibbs se cumplió con el unigramas: $H(\hat p_{\mathrm{prueba}})=4.190885\le H(\hat p_{\mathrm{prueba}},\hat q)=4.192472$, con $KL=0.001587$ bits.

### 5.4 Simulación y distribución estacionaria (Fase 5)

![[F5.png]]

*Figura F5. Distribución estacionaria $\pi$ del bigrama suavizado frente a las frecuencias de unigramas de entrenamiento (probabilidad).*

**Tabla. Verificación de la simulación.**

| Magnitud | Valor |
|---|---:|
| Tamaño de $P$ | $35\times35$ |
| Error máximo de sumas por fila | $3.331\times10^{-16}$ |
| Entrada mínima de $P$ | $3.509\times10^{-7}$ |
| Iteraciones de potencias | 24 |
| $\|\pi_t-\pi_{t-1}\|_1$ final | $5.610\times10^{-11}$ |
| $\max_y |\pi_y-\hat p(y)|$ | $9.36\times10^{-6}$ |
| Semilla / longitud de muestra | 20261123 / 300 caracteres |

Los símbolos más probables son el espacio ($\pi=0.1598$), la $e$ ($0.1017$), la $a$ ($0.0905$) y `#` ($0.0702$), en el mismo orden que en entrenamiento. La diferencia máxima entre $\pi$ y la frecuencia empírica es menor que $10^{-5}$. Las tres muestras de 300 caracteres se transcriben en el Anexo C.

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

El intervalo excluye el cero, de modo que la mejora de $n=5$ sobre $n=4$ no es compatible con una diferencia nula bajo este esquema de remuestreo.

### 5.6 Tokenización BPE (Fase 7)

**Tabla. Resultado de la Fase 7 y comparación justa por carácter.**

| Magnitud | Valor |
|---|---:|
| Fusiones BPE | 200 |
| $|\mathcal V_{\text{tokens}}|$ | 1900 |
| $\alpha$ elegido en validación | 0.03 |
| Tokens de entrenamiento / prueba | 209 817 / 26 178 |
| Caracteres de prueba | 66 879 |
| BPC por carácter (BPE) | 2.427847 |
| BPC por carácter (Fase 4, $n=5$) | 2.153786 |
| Diferencia BPE $-$ Fase 4 | $+0.274061$ |

Con el mismo denominador (caracteres de prueba), el bigrama de tokens queda $0.2741$ bits/carácter por encima del modelo de caracteres de orden 5. Las 200 fusiones reducen la secuencia de entrenamiento de $535\,028$ a $209\,817$ tokens, pero el vocabulario resultante es mucho más grande que el alfabeto de 35 símbolos, por lo que cada condicional se estima con menos datos y el suavizado tiene más masas que repartir.

## 6. Verificaciones

**Tabla. Comprobaciones ejecutadas y tolerancias declaradas.**

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

La comparación $H(\hat p)=4.1992$ frente a $\log_2 35=5.1293$ bits no significa que el texto ``contenga'' menos información que una cadena aleatoria: significa que la distribución empírica es más concentrada. Codificando con un modelo uniforme se gastarían $5.1293$ bits por carácter; usando las frecuencias reales se gastan $4.1992$, un ahorro de $0.9301$ bits por símbolo. La dependencia entre caracteres, que los unigramas ignoran, es lo que miden los BPC de T3.

### 7.2 Conteos MLE y probabilidades cero (Fase 2)

El BPC de la MLE en validación es infinito en cuanto aparece un solo cero porque promedia $-\log_2\hat q$ y $-\log_2 0=+\infty$: un único evento imposible domina la media. Esto no es un defecto de la métrica, sino una consecuencia de haber asignado probabilidad cero a un evento que sí ocurrió.

Los ceros crecen con $n$ porque el espacio de contextos posibles crece como $|\mathcal V|^{n-1}$ y las $535\,028$ posiciones de entrenamiento no pueden cubrirlo. Para $n=5$ hay $66\,833$ pares observados repartidos en $23\,672$ contextos, es decir, $2.82$ sucesos por contexto en promedio; muchas transiciones posibles nunca se ven. La perturbación de $0.01$ en un contexto frecuente confirmó la Prop. 1.1.8 en los cinco órdenes.

### 7.3 Elección de $\alpha$ (Fase 3)

El $\alpha$ óptimo cambia con $n$ porque mide un compromiso entre sesgo y varianza que depende de cuántos datos hay por contexto. Con $n=1$ hay un solo contexto y $535\,028$ observaciones: incluso $\alpha=1$ mueve cada probabilidad en menos de $2\times10^{-6}$, por lo que la malla se queda en su extremo superior y cualquier valor pequeño daría casi el mismo BPC. Con $n=2$ hay unas $15\,286$ transiciones por contexto; con $n=3$, $816$; con $n=4$, $99$; con $n=5$, $22.6$. A medida que el contexto se rarece, aumentar $\alpha$ ya no solo suaviza ruido: introduce masa en símbolos que no deberían ser probables. Por eso los órdenes 4 y 5 prefieren $\alpha=0.03$. El valor $0.1$ de $n=3$ muestra que la relación no es estrictamente monótona: depende del reparto real de conteos, no solo de $n$.

Con $\alpha=1$ en un contexto que aparece pocas veces el efecto es severo. *Cálculo ilustrativo derivado de (2):* si $c(h)=1$ y el símbolo observado tiene $c(h,y)=1$, entonces $\hat q_1(y\mid h)=2/36=0.0556$ y los otros 34 símbolos reciben $34/36=0.9444$ en conjunto; con $\alpha=0.03$, en cambio, $\hat q(y\mid h)=1.03/2.05=0.5024$ y el resto recibe $0.4976$. Es decir, $\alpha=1$ casi olvida la única observación del contexto, mientras que $\alpha=0.03$ conserva la mitad de la masa en lo visto.

### 7.4 Orden, prueba y sobreajuste (Fase 4)

La validación eligió $n=5$ ($2.1838$ bits frente a $4.2000$, $3.2689$, $2.7432$ y $2.3362$ de los órdenes 1 a 4) y la prueba, evaluada una sola vez, confirma el orden con BPC $2.1538$ y PPL $4.4499$. El BPC de entrenamiento sigue bajando ($1.8482$ bits en $n=5$) porque cada condicional se estima con menos datos y se ajusta a idiosincrasias del entrenamiento; validación y prueba pagan ese coste y por eso su curva se aplana. La estabilización de F4 ilustra la ley de los grandes números: el promedio acumulado de $-\log_2\hat q$ converge al costo medio. La desigualdad de Gibbs se verificó numéricamente con $KL=0.0016$ bits.

### 7.5 Simulación (Fase 5)

La distribución estacionaria se parece a las frecuencias de unigramas porque, para una cadena de Markov ergódica, la marginal de $X_t$ en el régimen estacionario es precisamente $\pi$, y una realización suficientemente larga tiene frecuencias empíricas que convergen a esa marginal. Aquí el bigrama suavizado conserva casi todos los conteos observados ($\alpha=0.03$ es pequeño frente a los conteos de los pares frecuentes), por lo que su $\pi$ hereda la distribución del texto: la diferencia máxima fue $9.36\times10^{-6}$.

Las muestras capturan regularidades locales: agrupaciones típicas como ``de'', ``que'', ``la'', alternancia consonante-vocal, presencia de `#` para puntuación y una ortografía razonable en $n=2$ y $n=3$. No capturan semántica, sintaxis a largo plazo ni coherencia narrativa: son cadenas de caracteres sin significado. En $n=5$ se aprecia además una degeneración hacia secuencias de vocales y símbolos acentuados poco frecuentes: con solo $22.6$ transiciones por contexto, incluso $\alpha=0.03$ reparte masa apreciable entre símbolos raros y la cadena se desvía del español. Este efecto ilustra el coste de aumentar la memoria sin más datos.

### 7.6 Incertidumbre (Fase 6)

El intervalo $[-0.1632,-0.1410]$ excluye el cero, así que la mejora de $n=5$ sobre $n=4$ no es compatible con una diferencia nula bajo este esquema de remuestreo. Se remuestrean bloques y no caracteres individuales porque los caracteres no son intercambiables: la cadena de Markov introduce dependencia local y un bootstrap de caracteres destruiría la estructura que el modelo aprovecha, subestimando la variabilidad. Los bloques de 2000 caracteres conservan esa dependencia; el último bloque, más corto, se promedia por su longitud en `metricas_prueba.csv` y no se descarta.

### 7.7 Limitaciones

Los resultados son propios de este corpus y de este alfabeto: otra obra u otra tokenización puede cambiar el orden óptimo. El modelo es una cadena de Markov sobre caracteres: no representa palabras, semántica ni dependencias más allá de $n-1$ caracteres. La Fase 7 muestra que tokenizar no ayuda automáticamente: con 200 fusiones y un vocabulario de 1900 tokens el BPC por carácter empeora respecto de $n=5$. Finalmente, el intervalo bootstrap cuantifica la variabilidad muestral dentro del bloque de prueba, pero no cubre incertidumbre sobre la elección del propio modelo.

## 8. Conclusiones

En este corpus el orden $n=5$ con $\alpha=0.03$ obtuvo el menor BPC de validación y también el menor BPC de prueba, $2.1538$ bits por carácter ($\mathrm{PPL}=4.4499$). La ventaja sobre $n=4$, de $0.1518$ bits, tiene un intervalo bootstrap del 95 % que excluye el cero. Las verificaciones de `abracadabra`, las sumas condicionales, la desigualdad de Gibbs y la convergencia de $\pi$ pasaron, de modo que las cifras principales son reproducibles. Como limitación, el modelo captura ortografía y ritmo local, pero no semántica ni dependencias largas, y la tokenización BPE probada no mejoró el BPC por carácter.

## Referencias

- Dante Alighieri. *La Divina Comedia*, traducción de Manuel Aranda y Sanjuán. Barcelona: Empresa Editorial La Ilustración, 1871. Copia en Internet Archive: `https://archive.org/details/ladivinacomedia57303gut`.
- Cover, T. M. y Thomas, J. A. (2006). *Elements of Information Theory*, 2.a edición. Wiley.
- Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Marcano, J. *Apuntes de Probabilidad*, sección S01: modelos de lenguaje, entropía y bootstrap. Material del curso.
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
