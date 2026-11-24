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

Estimamos modelos de *n*-gramas de caracteres ($n=1,\dots,5$) sobre una traducción al español de *La Divina Comedia* ($668\,786$ caracteres normalizados, $|\mathcal V|=35$). La partición contigua 80/10/10 fija el vocabulario solo con entrenamiento; $\alpha$ se eligió únicamente con validación y el bloque de prueba se evaluó una sola vez. La validación seleccionó $n=5$ con $\alpha=0.03$: BPC de prueba $2.1538$ bits/carácter y perplejidad $4.4499$. Frente a $n=4$, la diferencia fue $\Delta=-0.1518$ bits con intervalo bootstrap del 95 % $[-0.1632,-0.1410]$ (34 bloques, 10 000 remuestras), que excluye el cero. La matriz estocástica del bigrama convergió en 24 iteraciones a una distribución estacionaria que difiere de las frecuencias de unigramas en menos de $10^{-5}$. La desigualdad de Gibbs se cumplió ($KL=0.0016$ bits). La variante opcional con 200 fusiones BPE dio $2.4278$ bits por carácter, peor que el modelo de caracteres. Concluimos que la memoria local de hasta cinco caracteres mejora la predicción sin sobreajuste apreciable, aunque el modelo no capta semántica ni dependencias largas. Todas las cifras proceden de los CSV generados por el código entregado.

## 1. Introducción

**Planteamiento.** Este trabajo aplica la sección S01 «El lenguaje como distribución de probabilidad» del libro *Modelos de Lenguaje de Gran Escala* (Dr. José G. Marcano). Predecir el siguiente carácter de un texto es el ejemplo más directo de un modelo de lenguaje: la probabilidad de una cadena se factoriza, por la regla de la cadena (Prop. 1.1.3), en el producto de condicionales de cada símbolo dado su contexto inmediato. Si el contexto se limita a los $n-1$ caracteres anteriores obtenemos una cadena de Markov de orden $n-1$ (Def. 1.1.6). La pregunta del proyecto se concreta en tres puntos: qué orden $n$ predice mejor este texto español, qué suavizado $\alpha$ conviene a cada orden y si la ganancia del mejor orden sobre el anterior es distinguible de cero.

**Cómo se responde cada punto.** El orden y el suavizado se fijan con el bloque de validación (`validacion_alpha.csv` y `alpha_por_n.json`); la comparación final se calcula una sola vez con prueba (`metricas_prueba.csv` y `T3.csv`); la incertidumbre se obtiene remuestreando bloques de ese mismo archivo (`T4.csv`). El corpus se normalizó y dividió en bloques contiguos 80/10/10 con `preparar.py`, y el vocabulario se fijó solo con entrenamiento. La Sección 2 enuncia las cuatro fórmulas que implementa `modelos.py`; la Sección 3 documenta la procedencia y limpieza del texto; la Sección 4 explica los procedimientos; la Sección 5 lee los CSV y las figuras; la Sección 6 reproduce las verificaciones; la Sección 7 interpreta las cifras y la Sección 8 cierra el informe.

## 2. Marco probabilístico

**Notación y factorización.** Sea $\mathcal V$ el alfabeto finito de símbolos (Def. 1.1.1); aquí $|\mathcal V|=35$, fijado por `preparar.py` a partir de entrenamiento. Escribimos $x_{1:T}$ para una cadena de caracteres y $h_t=x_{t-n+1:t-1}$ para su contexto de longitud $n-1$. La regla de la cadena y la hipótesis de Markov dan

$$
  q(x_{1:T})=\prod_{t=1}^{T}q(x_t\mid x_{1:t-1})
  \;\approx\;\prod_{t=1}^{T}q(x_t\mid x_{t-n+1:t-1}). \tag{1}
$$

El signo $\approx$ indica la hipótesis de Markov: se sustituye el contexto completo $x_{1:t-1}$ de la regla de la cadena por sus $n-1$ caracteres finales; para $n=1$ la igualdad es exacta. Esta ecuación es la que hay detrás del bucle de generación de muestras de la Fase 5: en cada paso se conservan solo los últimos $n-1$ símbolos del contexto.

**Estimación.** La estimación por máxima verosimilitud (Prop. 1.1.8) es la frecuencia relativa

$$
  \hat q(y\mid h)=\frac{c(h,y)}{c(h)},\qquad c(h)=\sum_{y\in\mathcal V}c(h,y),
$$

donde $c(h,y)$ cuenta las veces que $y$ sigue al contexto $h$ en entrenamiento y $c(h)$ es el número de visitas a ese contexto. `ModeloNgramas._ajustar` calcula ambos contadores y `prob` devuelve esta razón cuando $\alpha=0$. Si el par no aparece, la MLE asigna probabilidad cero, comportamiento que provoca los ceros cuantificados en T2. El suavizado aditivo (Prop. 1.1.14) evita los ceros sumando un seudocuento $\alpha>0$ a cada par; $\alpha$ controla cuánta masa se reparte entre símbolos no vistos:

$$
  \hat q_\alpha(y\mid h)=\frac{c(h,y)+\alpha}{c(h)+\alpha|\mathcal V|},
  \qquad \alpha=0 \text{ es la MLE}. \tag{2}
$$

El denominador incluye $\alpha|\mathcal V|$, de modo que $\sum_{y\in\mathcal V}\hat q_\alpha(y\mid h)=1$ para todo contexto; la Fase 2 comprueba esa identidad con tolerancia $10^{-12}$. La deducción de la Ec. (2) por Lagrange y su lectura como media predictiva Beta/Dirichlet están en el Anexo A.

**Métricas de codificación.** La entropía satisface $H(p)=-\sum_i p_i\log_2 p_i\le\log_2|\mathcal V|$, con igualdad solo bajo uniformidad (Prop. 1.1.15). En modelado de caracteres se reporta

$$
  \mathrm{BPC}=-\frac{1}{N_e}\sum_{t=1}^{N_e}\log_2\hat q_\alpha(x_t\mid h_t),
  \qquad \mathrm{PPL}=2^{\mathrm{BPC}}, \tag{3}
$$

donde $N_e$ es el número de caracteres evaluados (Def. 1.1.20): son $66\,879$ en validación y en cada bloque de prueba porque el prefijo completa el contexto de los primeros caracteres, y $535\,028-(n-1)$ en entrenamiento porque allí no se antepone prefijo. El BPC es la entropía cruzada empírica y su exponencial en base 2 es la perplejidad; ambos se calculan en `modelos.py` y se escriben en `T3.csv`.

**Comparación con la entropía y Gibbs.** La entropía cruzada $H(p,q)$ mide el costo medio de codificar con $q$ datos de $p$; por la desigualdad de Gibbs (Teo. 1.1.27), $H(p,q)-H(p)=D_{\mathrm{KL}}(p\|q)\ge0$ (Def. 1.1.17):

$$
  H(p)=-\sum_i p_i\log_2 p_i\le H(p,q)=-\sum_i p_i\log_2 q_i,
  \qquad D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p). \tag{4}
$$

La Fase 4 aplica esta desigualdad con $p$ igual a las frecuencias empíricas de prueba y $q$ el unigramas suavizado, y guarda los tres términos en T3. La diferencia entre $\log_2 35$ y $H(\hat p)$ de T1 es otra aplicación de la misma cota, con $q$ uniforme.

**Procesos y remuestreo.** Para una matriz de transición positiva $P$, la entrada $P_{yz}$ es la probabilidad de pasar del símbolo $y$ al símbolo $z$, y $\pi_y$ es la probabilidad de estar en $y$ en régimen estacionario; entonces $\pi P=\pi$ y $\pi_0P^t\to\pi$ (Teo. 1.1.40). Se itera hasta $\|\pi_t-\pi_{t-1}\|_1<10^{-10}$; la positividad de $P$ proviene del suavizado, que hace todas las entradas mayores que cero. Finalmente, el bootstrap (Def. 1.1.44) remuestrea con reemplazo unidades consideradas independientes; aquí son bloques contiguos de prueba, no caracteres, por la dependencia local de la cadena. Las tres herramientas —iteración de potencias, remuestreo y entropía— se programaron desde cero con NumPy y Python estándar, sin librerías de modelos de lenguaje.

## 3. Datos

**Procedencia de la fuente.** La fuente es *La Divina Comedia*, Dante Alighieri, traducción de Manuel Aranda y Sanjuán (Barcelona, 1871), descargada de `https://archive.org/details/ladivinacomedia57303gut` el 21 de noviembre de 2026. Es texto plano en CP1252, copia del Gutenberg eBook \#57303; la traducción del siglo XIX es de dominio público y la procedencia está registrada en `datos/FUENTE.md`. El archivo crudo se guarda en `datos/raw/divina_comedia.txt` y no se incluye en la entrega; solo se entrega su ficha de procedencia.

**Cómo se limpió el texto.** `preparar.py` decodifica el archivo como CP1252 y `leer_y_recortar` localiza los marcadores `*** START OF THIS PROJECT GUTENBERG EBOOK` y `*** END OF THIS PROJECT GUTENBERG EBOOK`; entre ellos toma el tramo que va desde `_CANTO PRIMERO_` hasta `_INDICE_`. Después elimina con expresiones regulares los encabezados de canto, la palabra `FIN`, las etiquetas `[Ilustración]` y las notas numéricas `[n]`. `normalizar` pasa a minúsculas, conserva solo el alfabeto permitido —letras `a`--`z`, vocales acentuadas `áéíóú`, `ñ`, `ü`, espacio y el propio `#`—, sustituye por `#` cualquier otro carácter y colapsa espacios y saltos repetidos en uno solo. El resultado son $668\,786$ caracteres normalizados.

**Cómo se fijaron vocabulario y partición.** `particionar` divide ese texto en tres bloques contiguos: 80 % para entrenamiento, 10 % para validación y 10 % para prueba. El vocabulario $\mathcal V$ se construye con `sorted(set(train))`, es decir solo con el primer bloque: tiene 35 símbolos. Las frecuencias se obtienen con `train.count(ch)` y la entropía unigrama se calcula como $-\sum_y \hat p(y)\log_2\hat p(y)$ sobre esas frecuencias; la cota es $\log_2 35$. Los tres tamaños, $|\mathcal V|$, $H(\hat p)$ y $\log_2|\mathcal V|$ se escriben en `resultados/T1.csv`, las frecuencias en `resultados/vocabulario.json` y el histograma ordenado en `resultados/F1.pdf` y `F1.png`.

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

La diferencia entre ambas entropías es $5.1293-4.1992=0.9301$ bits por símbolo y se interpreta en §7.1. La figura F1 ordena las frecuencias de los 35 símbolos en entrenamiento.

![[F1.png]]

*Figura F1. Frecuencias de los unigramas en entrenamiento, ordenadas de mayor a menor (eje vertical: caracteres).*

## 4. Métodos

Esta sección describe el recorrido completo que transforma el corpus de la Sección 3 en las cifras de la Sección 5. El orden de ejecución es `preparar.py` → `pruebas.py` → `experimentos.py A/B`: el primero escribe los tres bloques y `resultados/vocabulario.json`; el segundo verifica la API; el tercero lee esos archivos mediante `cargar_datos()` y genera tablas y figuras. `modelos.py` contiene la única implementación de conteos, probabilidades, secuencias, BPC y PPL, de modo que las seis fases comparten el mismo código. Las fases de A (2, 4 y 6) y de B (3, 5 y 7) comparten también `calcular_alpha_por_n`, así que el suavizado se decide una sola vez con validación y la misma selección alimenta todas las mediciones. Cada figura se guarda en PDF para la entrega y en PNG para incrustarla en este informe.

### 4.1 Conteos, MLE y contexto inicial

**Cómo se obtienen los conteos.** La función `ModeloNgramas._ajustar` recorre `train` a partir de la posición $k=n-1$ y acumula dos contadores: $c(h)$, el número de veces que aparece el contexto $h=x_{t-k:t-1}$, y $c(h,y)$, el número de veces que $y$ sigue a ese contexto. Para $n=1$ hay un único contexto, la cadena vacía, y los conteos son las frecuencias de los 35 símbolos. Los pares no observados no se almacenan: por eso el número de pares de T2 crece de 35 ($n=1$) a 66 833 ($n=5$), mientras los contextos observados pasan de 1 a 23 672. Estas dos columnas de T2 son literalmente `len(modelo.contextos)` y la suma de los tamaños de `modelo.pares`.

**MLE y comprobación de las condicionales.** Con esos conteos, la probabilidad estimada es la frecuencia relativa

$$
  \hat q_0(y\mid h)=\frac{c(h,y)}{c(h)},\qquad \hat q_0(y\mid h)=0\ \text{si}\ c(h,y)=0. \tag{5}
$$

La suma sobre $\mathcal V$ es exactamente 1 porque $\sum_{y\in\mathcal V}c(h,y)=c(h)$. Se verifica con `suma_condicionales(h)` para cada contexto observado; el error máximo es $0$ para $n=1,2$ y $1.110\times10^{-16}$ para $n=3,4,5$, dentro de la tolerancia $10^{-12}$ (columna `max_error_sumas` de T2).

**Contexto inicial de las particiones.** `prefijo_de(train, n)` toma los últimos $n-1$ caracteres de entrenamiento y `secuencias(val, prefijo)` los antepone a validación antes de generar los pares. Así, las primeras $n-1$ posiciones de validación tienen contexto completo y se evalúan los $66\,879$ caracteres para los cinco órdenes, sin descartar ninguno. En prueba se aplica el mismo criterio, pero el prefijo se toma de `val + test[:inicio]`; en entrenamiento no se añade prefijo, por lo que la Fase 4 descarta las primeras $n-1$ posiciones y usa $N_e=535\,028-(n-1)$.

**Ceros y verosimilitud de entrenamiento.** Recorriendo `secuencias(val, prefijo)` se cuenta cada carácter con $\hat q_0=0$: son 0, 2, 122, 749 y 2 615 caracteres para $n=1,\dots,5$, es decir las fracciones $0$, $0.000030$, $0.001824$, $0.011199$ y $0.039100$ de T2. La log-verosimilitud en base 2 se obtiene sumando solo sobre los pares observados,

$$
  \ell_0=\sum_{h}\sum_{y:\,c(h,y)>0} c(h,y)\,\log_2\frac{c(h,y)}{c(h)}, \tag{6}
$$

y da los valores de la columna `log2_verosimilitud_train` de T2. Un par no visto no entra en esta suma, pero si aparece en validación su $-\log_2 0$ es infinito y el BPC de la MLE diverge.

**Cómo se construyen las perturbaciones.** `_elegir_perturbaciones` selecciona el contexto más frecuente, dentro de él el símbolo más probable como origen y dos destinos distintos, priorizando los ya observados en ese contexto. Se mueve $\delta=0.01$ de masa del origen a cada destino, uno por variante, sin modificar los datos: $\hat q_1(y_o\mid h)=q_o-\delta$, $\hat q_1(y_d\mid h)=q_d+\delta$. La nueva verosimilitud se calcula restando y sumando únicamente los términos afectados,

$$
  \ell_1=\ell_0+c(h,y_o)\log_2\frac{q_o-\delta}{q_o}
        +c(h,y_d)\log_2\frac{q_d+\delta}{q_d}. \tag{7}
$$

La comparación $\ell_0>\ell_1$ en los cinco órdenes (tolerancia $10^{-9}$) verifica la Prop. 1.1.8: la MLE es el máximo sobre los datos observados. El contexto y los símbolos usados quedan registrados en las últimas columnas de T2; el símbolo `#` se escribe escapado como `\#` para no romper el CSV.

### 4.2 Suavizado y selección de $\alpha$

**Cómo se genera la malla de valores.** Se aplica la Ec. (2) con $\alpha>0$, que asigna masa a los 35 pares del contexto incluso si no se han visto. Para cada $n\in\{1,\dots,5\}$ y cada $\alpha\in\{0.001,0.003,0.01,0.03,0.1,0.3,1\}$ se ajusta un modelo con `train` y se mide su BPC en `val` con el prefijo de entrenamiento:

$$
  \mathrm{BPC}_{\mathrm{val}}(n,\alpha)
  =-\frac{1}{66\,879}\sum_{t=1}^{66\,879}\log_2\hat q_\alpha(x_t\mid h_t). \tag{8}
$$

Son $5\times7=35$ ajustes. Ninguna de estas expresiones lee `test`: la función `bpc_validacion_por_alpha` recibe `_test` y lo ignora explícitamente.

**Cómo se elige $\alpha$.** Para cada orden se toma el mínimo de la fila con `min(ALPHAS, key=lambda a: (bpc[a], a))`, de modo que un empate se resuelve por el $\alpha$ más pequeño. `asegurar_alpha_por_n` guarda la selección en `resultados/alpha_por_n.json`; en corridas posteriores la lee en lugar de recalcularla, y la Fase 3 vuelve a derivarla y comprueba que coincide con la caché antes de graficar. Los 35 valores, con una marca `es_mejor` en el mínimo de cada fila, se escriben en `validacion_alpha.csv`; la figura F2 los dibuja con eje horizontal logarítmico y una curva por orden. El resultado es $\alpha=\{1:1,\ 2:0.03,\ 3:0.1,\ 4:0.03,\ 5:0.03\}$, con los BPC de validación que se recogen en T5. El orden $n=1$ tiene un solo contexto con cientos de miles de observaciones y apenas distingue valores de $\alpha$; en los órdenes altos, en cambio, un $\alpha$ grande repartiría demasiada masa entre símbolos improbables.

### 4.3 Métricas y protocolo de prueba

**Cómo se decide el orden.** Con los $\alpha$ ya fijados, `bpc_validacion_por_n` recalcula el BPC de validación para $n=1,\dots,5$ y `mejor_n_por_validacion` toma el argmin. El mínimo corresponde a $n^\star=5$ con $2.183777$ bits/carácter. Esta decisión precede a cualquier lectura del bloque de prueba.

**Cómo se mide la prueba una sola vez.** `evaluar_prueba_una_vez` comprueba primero si existe `resultados/metricas_prueba.csv`: si existe, imprime `la prueba NO se reevalúa` y termina; solo la opción `--forzar-prueba` la regenera deliberadamente. En la primera ejecución divide los $66\,879$ caracteres de prueba en 34 bloques contiguos de 2000 (el último de $879$), estima un modelo por orden con su $\alpha$ y escribe, para cada bloque, `n`, `alpha`, `bloque`, `caracteres` y `bpc` con diez decimales. El prefijo de cada bloque se toma de `val + test[:inicio]`, de modo que el primer carácter del bloque también tiene contexto completo. En la misma función se dibuja F4: el promedio acumulado de $-\log_2\hat q$ sobre prueba para el mejor orden, usando como contexto inicial el final de validación.

**Cómo se agregan las métricas.** El BPC de prueba por orden es un promedio ponderado por la longitud real de los bloques,

$$
  \mathrm{BPC}_{\mathrm{prueba}}(n)
  =\frac{\sum_j \mathrm{BPC}_j(n)\,\ell_j}{\sum_j \ell_j}, \tag{9}
$$

y la PPL es $2^{\mathrm{BPC}_{\mathrm{prueba}}}$, según la Ec. (3). El BPC de entrenamiento se calcula con `.bpc(train)` sobre el mismo modelo. Los cinco renglones, junto con el BPC de validación, la PPL y una marca `es_mejor_n`, forman `T3.csv`; la figura F3 enfrenta las curvas de entrenamiento y prueba. Con el mejor modelo, el BPC de entrenamiento es $1.848241$ y el de prueba $2.153786$, con PPL $4.4499$.

**Cómo se verifica Gibbs.** Se toma $p$ como la frecuencia empírica de cada símbolo en prueba —solo los de conteo positivo— y $q$ como el unigramas suavizado con $\alpha=1$. Se calculan $H(p)$, $H(p,q)$ y su diferencia $D_{\mathrm{KL}}(p\|q)$ de la Ec. (4); la comprobación $D_{\mathrm{KL}}\ge-10^{-12}$ da $4.190885\le4.192472$ y $KL=0.001587$ bits. Los tres números se repiten en todas las filas de T3 para dejar constancia de la verificación.

### 4.4 Simulación y distribución estacionaria

**Cómo se construye la matriz.** Con el bigrama suavizado de $\alpha=0.03$ —el elegido en la Fase 3 para $n=2$— se calcula $P_{yz}=\hat q_\alpha(z\mid y)$ para los 35 símbolos del alfabeto ordenado. Antes de usarla se comprueba que cada fila suma 1 (error máximo $3.331\times10^{-16}$) y que la entrada mínima es positiva ($3.509\times10^{-7}$), condición que garantiza el suavizado. Ambas cifras se guardan en `verificacion_simulacion.csv`.

**Cómo se obtiene $\pi$.** Se parte de $\pi_0=(1/35,\dots,1/35)$ y se itera $\pi_{t+1}=\pi_tP$ hasta $\|\pi_{t+1}-\pi_t\|_1<10^{-10}$. La convergencia se alcanza en 24 iteraciones, con norma final $5.610\times10^{-11}$. Para comparar, se calculan las frecuencias de unigramas de entrenamiento como $c(y)/535\,028$; la diferencia máxima entre $\pi_y$ y esa frecuencia es menor que $10^{-5}$. Los 35 pares $\pi_y$, $\hat p(y)$ y su diferencia se escriben en `estacionaria.csv` y se dibujan en F5, con las barras ordenadas por frecuencia de entrenamiento.

**Cómo se generan las muestras.** Un único generador `numpy.random.default_rng(20261123)` produce las tres muestras, para $n=2,3,5$, con longitudes de 300 caracteres y el mismo contexto inicial `" que "`, cuya presencia en `train` se comprueba antes de empezar. En cada paso se evalúa $\hat q_\alpha(\cdot\mid h)$ sobre $\mathcal V$, se normaliza y se elige un símbolo con `rng.choice`; después el contexto se desplaza y conserva solo los $n-1$ caracteres finales. Las tres secuencias, su $\alpha$ y la semilla se guardan en `muestras_simulacion.csv` y se transcriben en el Anexo C.

### 4.5 Bootstrap por bloques

**Datos de entrada.** Esta fase no evalúa la prueba: llama a `leer_metricas_prueba()` y lee el CSV generado por la Fase 4. El orden de referencia se obtiene de nuevo por validación con los $\alpha$ cacheados ($n^\star=5$) y el orden comparado es $n^\star-1=4$.

**Diferencia por bloque.** Para cada uno de los 34 bloques se calcula

$$
  \Delta_j=\mathrm{BPC}_j(5)-\mathrm{BPC}_j(4), \tag{10}
$$

donde los BPC ya están divididos por la longitud real del bloque (2000 salvo el último, de 879). La media de las 34 diferencias es $-0.151797$ bits/carácter.

**Intervalo por remuestreo.** Con `numpy.random.default_rng(20261122)` se extraen 10 000 remuestras de los 34 valores de $\Delta$ con reemplazo —una matriz $10\,000\times34$— y se promedia cada fila; el intervalo del 95 % son los percentiles 2.5 y 97.5 de esas 10 000 medias:

$$
  \mathrm{IC}_{95\%}=
  \big[Q_{0.025}(\bar\Delta^*),\ Q_{0.975}(\bar\Delta^*)\big]
  =[-0.163172,\ -0.140998]. \tag{11}
$$

Como el extremo superior es negativo, la marca `intervalo_excluye_cero` de `T4.csv` es 1. Se remuestrean bloques y no caracteres porque la cadena tiene dependencia local; cada bloque pesa $1/34$, incluido el último, que es más corto. Por eso la media de T4 ($-0.151797$) no coincide exactamente con la diferencia global de T3 ($-0.151152$): T3 pondera por caracteres y T4 por bloques.

### 4.6 Tokenización BPE (Fase 7 opcional)

**Cómo se aprenden las fusiones.** `_aprender_bpe` separa los caracteres de `train` con un delimitador interno y, en cada una de 200 iteraciones, cuenta los pares adyacentes de tokens con `Counter`. Fusiona el más frecuente —con desempate por primera aparición— mediante un reemplazo global y guarda la regla. Las reglas se aprenden solo con entrenamiento, sin consultar validación ni prueba.

**Cómo se codifican los bloques.** `_tokenizar` aplica las 200 reglas en orden a `train`, `val` y `test`. El inventario es la unión ordenada de los tokens resultantes: 1900 símbolos, de los cuales 107 aparecen solo en validación o prueba. Esos tokens adicionales permiten codificar los bloques, pero no aportan conteos al ajuste. Cada token se sustituye por un carácter único (`chr(i+1)`) para poder reutilizar `ModeloNgramas` sin cambiar la API.

**Cómo se elige $\alpha$ y se mide la prueba.** Se entrena un bigrama de tokens y, para cada $\alpha$ de la malla, se calcula el BPC por token en validación con prefijo de entrenamiento y se convierte a bits por carácter con el factor $N_{\text{tokens,val}}/N_{\text{caracteres,val}}$:

$$
  \mathrm{BPC}_{\text{carácter}}^{\text{BPE}}
  =\mathrm{BPC}_{\text{token}}\,
   \frac{N_{\text{tokens}}}{N_{\text{caracteres}}}. \tag{12}
$$

El mínimo en validación selecciona $\alpha=0.03$. Solo entonces se mide prueba una vez, con prefijo de validación, y se convierte con $26\,178/66\,879$, lo que da $2.427847$ bits/carácter. El BPC de referencia se lee de `T3.csv` (fila con `es_mejor_n=1`, $2.153786$) y la diferencia es $+0.274061$. Todo queda en `bpe_resultados.csv`, incluida la marca `metricas_prueba_csv_reescrito=0`: la comparación usa caracteres como denominador común y no modifica la evaluación única de la Fase 4.

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

**Procedencia de las cifras.** Cada fila de T2 la escribe `fase2_mle` con un modelo MLE distinto, uno por orden. Las columnas `contextos_observados` y `pares_observados` salen de los contadores descritos en §4.1; `max_error_sumas` es el mayor $|1-\sum_y\hat q_0(y\mid h)|$; `fraccion_ceros_validacion` es `ceros_validacion / caracteres_validacion`, con $66\,879$ caracteres evaluados en los cinco órdenes; y las tres log-verosimilitudes se calculan con las Ecs. (6) y (7) y se redondean a cuatro decimales al escribir el CSV. Las columnas finales registran el contexto y los símbolos efectivamente perturbados, de modo que cada comparación es reproducible. Las condicionales suman 1 con error máximo $1.110\times10^{-16}$; la MLE supera a las dos perturbaciones en los cinco órdenes (Prop. 1.1.8) y los ceros de validación crecen de $0$ a $0.0391$, como predice la rarefacción de los contextos.

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

**Procedencia de las cifras.** F2 la dibuja `fase3_alpha` directamente desde la matriz `bpc_validacion_por_alpha` (35 ajustes, todos con `train` y `val`); T5 no es una tabla nueva: es el mínimo de cada una de las cinco curvas, es decir la intersección de `validacion_alpha.csv` con `alpha_por_n.json`. Los BPC de validación se escriben con diez decimales en el CSV y con seis en T3, de modo que las cifras del informe coinciden con la precisión declarada en cada tabla. La elección no es monótona en $n$; la explicación está en §7.3 y el protocolo de selección en §4.2.

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

**Procedencia de las cifras.** Las cinco filas de T3 las produce `fase4_evaluacion` con `alpha_por_n.json`, los BPC de validación ya calculados, `.bpc(train)` y el promedio ponderado de los bloques de `metricas_prueba.csv` descrito en la Ec. (9). La PPL se obtiene como $2^{\mathrm{BPC}_{\mathrm{prueba}}}$. Las columnas `H_p_prueba`, `H_pq_prueba` y `KL_prueba` se repiten iguales en todas las filas porque son una sola verificación global, calculada con el unigramas y las frecuencias de prueba. F3 se dibuja en la misma función a partir de `bpc_train` y `bpc_test`; F4 se genera dentro de `evaluar_prueba_una_vez` en la única evaluación de prueba. La prueba confirma el orden elegido: BPC mínimo $2.1538$ bits y $KL=0.001587$ bits para el unigramas ($4.190885\le4.192472$).

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

**Procedencia de las cifras.** Los 35 pares que se comparan en F5 están en `estacionaria.csv`, con columnas $\pi$, frecuencia de entrenamiento y diferencia; T6 resume los escalares que `fase5_simulacion` escribe en `verificacion_simulacion.csv` porque son comprobaciones, no resultados de predicción. Las tres cadenas generadas se guardan en `muestras_simulacion.csv` con su orden, su $\alpha$, la semilla y el contexto inicial, y se transcriben sin cambios en el Anexo C. Los símbolos más probables son espacio ($0.1598$), $e$ ($0.1017$), $a$ ($0.0905$) y `#` ($0.0702$), en el mismo orden que en entrenamiento; la diferencia máxima entre $\pi$ y la frecuencia empírica es menor que $10^{-5}$.

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

**Procedencia de las cifras.** T4 la escribe `fase6_bootstrap` a partir de `leer_metricas_prueba()`: toma las columnas `bpc` de los órdenes 5 y 4, forma las 34 diferencias de la Ec. (10), las remuestrea con la semilla `20261122` y guarda la media, los percentiles y la marca `intervalo_excluye_cero`. El IC excluye el cero. T3 da la diferencia global sobre toda la prueba, $2.153786-2.304938=-0.151152$ bits, mientras T4 da la media no ponderada de las 34 diferencias por bloque, $-0.151797$ bits. No coinciden exactamente porque T3 divide por los $66\,879$ caracteres del bloque completo y T4 pesa cada bloque por igual, incluido el último de $879$ caracteres; son dos estimadores del mismo efecto, no una contradicción.

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

**Procedencia de las cifras.** Todas las entradas de T7 provienen de `bpe_resultados.csv`, escrito por `fase7_bpe`: el número de fusiones y de tokens se cuenta durante la codificación, el $\alpha$ se selecciona en validación con la Ec. (12) y el BPC de prueba se mide con prefijo de validación. El BPC de referencia se lee de `T3.csv` (fila marcada `es_mejor_n=1`) y la última fila del CSV confirma que `metricas_prueba.csv` no se reescribió. Los 107 tokens nuevos solo permiten codificar validación y prueba; no aportan conteos al ajuste. Con el mismo denominador, BPE queda $0.2741$ bits/carácter por encima de $n=5$: las 200 fusiones reducen la secuencia a $209\,817$ tokens, pero reparten las estimaciones entre 1900 símbolos en lugar de 35.

## 6. Verificaciones

**Cómo se ejecutaron.** Hay dos grupos de comprobaciones. El primero corresponde a `python codigo/pruebas.py`: la función `prueba_abracadabra` contrasta los seis números de la tabla 2 del enunciado usando directamente la API de `modelos.py`; `prueba_gibbs` comprueba la Ec. (4) con una distribución 2×2; y `prueba_estacionaria` itera una matriz conocida hasta recobrar $(2/3,1/3)$. El segundo grupo está repartido en las fases: `fase2_mle` verifica las sumas condicionales, `fase4_evaluacion` verifica Gibbs sobre la prueba, `fase5_simulacion` verifica la matriz $P$ y la convergencia de $\pi$, y la segunda ejecución de `evaluar_prueba_una_vez` verifica el protocolo. Las tolerancias están declaradas en el código y cualquier incumplimiento lanza un `AssertionError` en lugar de imprimir una cifra.

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

**Cómo se lee la tabla.** Los cinco primeros renglones son una prueba de cordura: no usan el corpus, sino la cadena `abracadabra`, y confirman que `entropia`, `prob`, `bpc` y `ppl` implementan las definiciones de la Sección 2. Los siguientes cuatro son verificaciones internas sobre el corpus y sus resultados están también en T2, T3 y `verificacion_simulacion.csv`. El último renglón no es numérico: se comprueba ejecutando dos veces el pipeline y confirmando que el segundo mensaje es `la prueba NO se reevalúa` y que `metricas_prueba.csv` conserva su contenido. Todas las verificaciones pasan ejecutando `python codigo/pruebas.py` y `python codigo/experimentos.py todo`. El pipeline completo tarda aproximadamente 77 segundos en esta máquina.

## 7. Discusión y limitaciones

### 7.1 Distancia al azar (Fase 1)

**De dónde sale la comparación.** Los dos términos están en T1 y los calcula `preparar.py` con las frecuencias de entrenamiento: $H(\hat p)=-\sum_y\hat p(y)\log_2\hat p(y)=4.1992$ bits y $\log_2|\mathcal V|=\log_2 35=5.1293$ bits. Un modelo uniforme sobre 35 símbolos asignaría $5.1293$ bits por carácter, mientras las frecuencias reales tienen $4.1992$ bits: el texto es más predecible porque la distribución es desigual, no porque ``contenga'' menos información. El ahorro, $5.1293-4.1992=0.9301$ bits por símbolo, es la diferencia entre ambas filas de T1. Esa comparación no mide la dependencia entre posiciones vecinas; para eso están los BPC de T3.

### 7.2 Conteos MLE y probabilidades cero (Fase 2)

**De dónde sale el crecimiento de los ceros.** La MLE asigna probabilidad cero a pares no vistos; si uno ocurre en validación, $-\log_2 0=+\infty$ domina la media y el BPC es infinito. El problema es del modelo, no de la métrica. Los ceros crecen con $n$ porque hay $|\mathcal V|^{n-1}$ contextos posibles —$35^4=1\,500\,625$ para $n=5$— y las $535\,028$ observaciones de entrenamiento no pueden cubrirlos. T2 registra para $n=5$ un total de $23\,672$ contextos observados y $66\,833$ pares distintos: son $66\,833/23\,672=2.82$ pares distintos por contexto, frente a las $535\,028/23\,672=22.6$ visitas totales por contexto que se calculan en §7.3. La fracción de caracteres de validación con probabilidad cero pasa de $0$ a $0.0391$ según la columna correspondiente de T2. Las dos perturbaciones de $0.01$ confirmaron la Prop. 1.1.8 en los cinco órdenes.

### 7.3 Elección de $\alpha$ (Fase 3)

**Cómo se calculan las visitas por contexto.** El cociente entre los caracteres de entrenamiento y la columna `contextos_observados` de T2 da $535\,028/35=15\,286$ visitas para $n=2$, $535\,028/656=816$ para $n=3$, $535\,028/5\,398=99$ para $n=4$ y $535\,028/23\,672=22.6$ para $n=5$; para $n=1$ hay un único contexto con las $535\,028$ observaciones. El $\alpha$ óptimo depende de esa densidad: incluso $\alpha=1$ cambia cada probabilidad en $1/(535\,028+35)\approx1.9\times10^{-6}$ cuando $n=1$, por lo que la malla termina en su extremo superior. Al rarecerse el contexto, subir $\alpha$ deja de corregir ruido y empieza a introducir masa en símbolos improbables; por eso $n=4$ y $n=5$ prefieren $0.03$, aunque $n=3$ elige $0.1$. El cálculo ilustrativo de la Ec. (2) lo hace visible: con $c(h)=1$ y una sola observación, $\alpha=1$ deja $2/36=0.0556$ de masa en lo visto y $\alpha=0.03$ deja $1.03/2.05=0.5024$. Los $\alpha$ efectivamente elegidos están en `alpha_por_n.json` y sus BPC en T5.

### 7.4 Orden, prueba y sobreajuste (Fase 4)

**Cómo se obtienen las ganancias.** Restando filas consecutivas de la columna `bpc_prueba` de T3 se obtiene $4.192472-3.257788=0.9347$ bits de $n=1$ a $2$, $3.257788-2.718914=0.5389$ de $2$ a $3$, $2.718914-2.304938=0.4140$ de $3$ a $4$ y $2.304938-2.153786=0.1512$ de $4$ a $5$. La prueba confirma la selección de validación: el BPC baja de $4.1925$ a $2.1538$ bits, con $\mathrm{PPL}=4.4499$. La columna `bpc_train` sigue bajando hasta $1.8482$ porque los contextos largos se ajustan a idiosincrasias del entrenamiento; validación y prueba pagan ese coste. La estabilización del promedio acumulado en F4 es una consecuencia de la ley de los grandes números y la desigualdad de Gibbs se verificó con $KL=0.001587$ bits, cifra que aparece en T3.

### 7.5 Simulación (Fase 5)

**Qué datos respaldan la afirmación.** La comparación entre $\pi$ y los unigramas está en `estacionaria.csv`, con un error máximo de $9.36\times10^{-6}$ (T6). El bigrama suavizado es ergódico: su marginal estacionaria es $\pi$ y las frecuencias de una realización larga convergen a ella. Como $\alpha=0.03$ conserva casi todos los conteos, $\pi$ hereda la distribución del texto. Las tres cadenas de `muestras_simulacion.csv` capturan regularidades locales (agrupaciones como ``de'' y ``que'', alternancia consonante-vocal y `#` como puntuación), pero no semántica ni coherencia. En $n=5$, con $22.6$ visitas por contexto, aparecen secuencias de vocales y símbolos raros: aumentar la memoria sin más datos tiene un coste.

### 7.6 Incertidumbre (Fase 6)

**De dónde sale el intervalo.** El IC $[-0.163172,-0.140998]$ es la fila `IC95_inferior`/`IC95_superior` de T4, obtenida de los 34 valores de $\Delta_j$ en `metricas_prueba.csv`; excluye el cero porque su extremo superior es negativo. Se remuestrean bloques y no caracteres porque la cadena introduce dependencia local; un bootstrap de caracteres destruiría esa estructura y subestimaría la variabilidad. Dentro de cada bloque el BPC se divide por su longitud real; en el remuestreo los 34 bloques, incluido el de $879$ caracteres, pesan $1/34$. El intervalo mide variabilidad dentro de esta prueba, no incertidumbre sobre la elección del modelo.

### 7.7 Pregunta central y limitaciones

La respuesta directa a la pregunta del enunciado es que, en este corpus, aumentar la memoria de $n=1$ a $n=5$ reduce el BPC de prueba de $4.1925$ a $2.1538$ bits: las ganancias al subir de orden son $0.9347$, $0.5389$, $0.4140$ y $0.1512$ bits, calculadas de T3 como se muestra en §7.4. Por tanto, la memoria sigue compensando hasta $n=5$ y la mejora de $4$ a $5$ se distingue de cero con el bootstrap de T4. No podemos situar el orden donde deja de compensar porque no se evaluaron órdenes superiores y ese juicio también depende del coste computacional; la tendencia decreciente y las $22.6$ visitas por contexto sugieren que la ganancia seguirá disminuyendo.

Los resultados son propios de este corpus y de este alfabeto: otra obra u otra tokenización puede cambiar el orden óptimo. El modelo no representa palabras, semántica ni dependencias más allá de $n-1$ caracteres, y T7 muestra que tokenizar no ayuda automáticamente: con 200 fusiones y 1900 tokens el BPE queda $0.2741$ bits/carácter por encima de $n=5$.

## 8. Conclusiones

**Resultado central.** En este corpus el orden $n=5$ con $\alpha=0.03$ obtuvo el menor BPC de validación (T5) y también el menor BPC de prueba, $2.153786$ bits por carácter ($\mathrm{PPL}=4.4499$, T3). La ventaja sobre $n=4$, de $0.1512$ bits según T3 y $0.1518$ según la media de T4, tiene un intervalo bootstrap del 95 % $[-0.1632,-0.1410]$ que excluye el cero.

**Cadena de evidencia.** T1 describe el corpus y su vocabulario; T2 documenta los pares no vistos que justifican el suavizado; T5 fija $\alpha$ solo con validación; T3, F3 y F4 confirman el orden en la evaluación única de prueba; T4 cuantifica la incertidumbre de la última mejora; T6 y F5 verifican la simulación; T7 evalúa la variante BPE con el mismo denominador de caracteres. Las verificaciones de `abracadabra`, las sumas condicionales, la desigualdad de Gibbs y la convergencia de $\pi$ pasaron con las tolerancias de T8, de modo que las cifras principales son reproducibles desde los comandos del Anexo B.

**Alcance.** El modelo captura ortografía y ritmo local, pero no semántica ni dependencias largas, y la tokenización BPE probada no mejoró el BPC por carácter: las muestras del Anexo C y la comparación de T7 ilustran ambos límites.

## Referencias

Las referencias se ordenan según el papel que cumplen en el informe. La fuente del corpus y su licencia se detallan en `datos/FUENTE.md` y en la Sección 3; las definiciones y proposiciones citadas en las Secciones 2, 4 y el Anexo A proceden del texto de Marcano; las nociones de entropía y codificación siguen a Shannon y a Cover y Thomas; y el método de remuestreo de la Sección 4.5 y de la discusión de la incertidumbre sigue a Efron y Tibshirani.

- Dante Alighieri. *La Divina Comedia*, traducción de Manuel Aranda y Sanjuán. Barcelona: Empresa Editorial La Ilustración, 1871. Copia en Internet Archive: `https://archive.org/details/ladivinacomedia57303gut`.
- Cover, T. M. y Thomas, J. A. (2006). *Elements of Information Theory*, 2.a edición. Wiley.
- Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Marcano, J. G. *Modelos de Lenguaje de Gran Escala: fundamentos matemáticos, arquitecturas y aplicaciones*, sección S01 «El lenguaje como distribución de probabilidad» (§§1.1.2–1.1.5 y Complementos de S01). Edición v0.9 del curso.
- Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27, 379--423.

## Anexo A. Deducción de la MLE y caso Beta

**Deducción de la Ec. (5).** Para un contexto $h$ fijo maximizamos $\ell(\mathbf q)=\sum_{y\in\mathcal V}c(h,y)\log q(y)$ sujeto a $\sum_y q(y)=1$. Los contadores $c(h,y)$ son los que produce `ModeloNgramas._ajustar` y aparecen en T2. Con un multiplicador de Lagrange $\lambda$ para la restricción de normalización,

$$
  \frac{\partial}{\partial q(y)}
  \left[\sum_y c(h,y)\log q(y)+\lambda\left(1-\sum_y q(y)\right)\right]
  =\frac{c(h,y)}{q(y)}-\lambda=0 .
$$

De la igualdad a cero se sigue $q(y)=c(h,y)/\lambda$ para todo $y$. Imponiendo la restricción, $\sum_y c(h,y)/\lambda=1$, se obtiene $\lambda=\sum_y c(h,y)=c(h)$ y por tanto

$$
  \hat q(y\mid h)=\frac{c(h,y)}{c(h)},
$$

que es exactamente la fórmula implementada por `prob` cuando $\alpha=0$. La solución es única porque $\ell$ es cóncava en el simplex: la log-verosimilitud (6) alcanza su máximo en ese punto, lo que explica por qué las dos perturbaciones de T2 siempre dan un valor menor.

*Caso con dos símbolos (opcional).* Si la previa es $\mathrm{Beta}(\alpha,\alpha)$ y se observan $c_1$ veces el símbolo 1 y $c_0$ veces el símbolo 0, la posterior es $\mathrm{Beta}(c_1+\alpha,c_0+\alpha)$ y la media predictiva es

$$
  \mathbb E[q_1\mid x]=\frac{c_1+\alpha}{c_0+c_1+2\alpha}
  =\frac{c(h,y)+\alpha}{c(h)+\alpha|\mathcal V|},
$$

que coincide con la Ec. (2) para $|\mathcal V|=2$: el suavizado aditivo es la media de una posterior de Dirichlet (Prop. 1.1.14). Este paralelismo explica por qué el denominador de la Ec. (2) contiene $\alpha|\mathcal V|$. En el experimento, $\alpha$ no se deduce teóricamente sino que se elige por validación, como se describe en §4.2 y se registra en T5.

## Anexo B. Reproducibilidad

**Entorno.** Python 3.14, NumPy 2.2.6 y Matplotlib 3.10.7, instalados desde `requirements.txt`. La biblioteca `fpdf` se usa solo en `reportes/generar_pdf.py` para un borrador interno del informe y no forma parte de la entrega. No se emplearon librerías de modelos de lenguaje ni de n-gramas: conteos, suavizado, BPC, iteración de potencias y bootstrap están implementados en `codigo/modelos.py` y `codigo/experimentos.py`.

**Comandos.** Desde la raíz del repositorio, en este orden:

```bash
python -m pip install -r requirements.txt
python codigo/preparar.py
python codigo/pruebas.py
python codigo/experimentos.py A
python codigo/experimentos.py B
```

`preparar.py` genera el corpus y T1/F1; `pruebas.py` corre las verificaciones de la Sección 6; `experimentos.py A` produce T2, T3, T4, F3, F4 y `metricas_prueba.csv`; `experimentos.py B` produce F2, F5, las muestras y los CSV de las fases 3, 5 y 7. También puede ejecutarse `python codigo/experimentos.py todo`, o una fase suelta como `python codigo/experimentos.py 3 5`. El tiempo total medido es **77.1 s**.

**Semillas y fecha fija.** Bootstrap: $20261122$ (Fase 6); simulación: $20261123$ (Fase 5). Cada una usa un generador propio con semilla declarada, de modo que las 10 000 remuestras y las tres cadenas de 300 caracteres se reproducen exactamente. Se fija además `SOURCE_DATE_EPOCH=1795392000` en `preparar.py` y `experimentos.py` para que los metadatos de los PDF sean constantes: dos ejecuciones completas producen archivos byte-idénticos, incluidos F1--F5.

**Protocolo de prueba.** La evaluación en prueba ocurre únicamente dentro de `evaluar_prueba_una_vez`. Si `metricas_prueba.csv` ya existe, una segunda corrida imprime `la prueba NO se reevalúa` y no modifica el archivo; solo `--forzar-prueba` lo regenera deliberadamente y no se usó para elegir $n$ ni $\alpha$. La Fase 6 y la Fase 7 leen ese CSV o `T3.csv` sin volver a evaluar el bloque de prueba.

## Anexo C. Muestras generadas y declaración de IA

**Cómo se obtuvieron.** Las tres cadenas se transcriben sin cambios de la columna `muestra` de `resultados/muestras_simulacion.csv`, escrita por `fase5_simulacion`. Se generaron con un único `numpy.random.default_rng(20261123)`, partiendo del contexto inicial común `" que "` y conservando en cada paso los $n-1$ caracteres finales; cada cadena tiene 300 caracteres. El CSV registra además el orden $n$, el $\alpha$ y la semilla de cada muestra. Se muestran los 300 caracteres generados después del contexto.

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

**Qué se delegó y qué se verificó.** Se usó el asistente OpenCode (modelo Big Pickle) para planificar el trabajo, redactar los scripts, revisar las fórmulas y producir borradores de texto. Las verificaciones propias fueron:

- ejecutar `pruebas.py` y cotejar los seis valores de `abracadabra` con la tabla 2 del enunciado;
- comprobar que las condicionales suman 1 con error máximo $1.110\times10^{-16}$, según T2;
- verificar las filas de $P$, la convergencia de $\pi$ en 24 iteraciones y el error frente a los unigramas, según T6;
- ejecutar dos veces el pipeline y confirmar que la segunda no reescribe `metricas_prueba.csv` ni cambia los archivos generados;
- inspeccionar T1--T4 y los CSV de las fases 3, 5 y 7 para que cada cifra del informe provenga de un archivo producido por el código;
- contrastar las ecuaciones del informe con las funciones implementadas en `modelos.py` y `experimentos.py`.

Ni la fuente ni las cifras se tomaron de la salida del asistente sin esta comprobación.

### Tabla de contribución

El reparto se deriva de las responsabilidades anotadas en `PLAN.md` y de las funciones implementadas en `experimentos.py`: A escribió `fase2_mle`, `fase4_evaluacion` y `fase6_bootstrap`; B, `fase3_alpha`, `fase5_simulacion` y `fase7_bpe`. `modelos.py` y `pruebas.py` se revisaron entre ambos.

| Integrante | Código y resultados | Reporte |
|---|---|---|
| A | Fases 1, 2, 4 y 6; T1--T4, F1, F3, F4 | Secciones 1--3, parte de la 7, 8, Anexos A--C |
| B | Fases 3, 5 y 7; F2, F5, muestras y CSV asociados | Secciones 4--6, parte de la 7, Anexo C, zip |
| Ambos | Revisión de `modelos.py` y `pruebas.py` | Revisión cruzada de cifras y coherencia |
