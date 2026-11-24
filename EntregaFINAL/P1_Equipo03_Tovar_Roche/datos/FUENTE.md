# Fuente del corpus

- **Título:** La Divina Comedia
- **Autor:** Dante Alighieri (1265–1321)
- **Traductor:** Manuel Aranda y Sanjuán
- **Edición:** Barcelona, Empresa Editorial La Ilustración, 1871 (con notas de Paolo Costa)
- **URL de descarga:** https://archive.org/details/ladivinacomedia57303gut
- **Fecha de descarga:** 2026-11-21
- **Licencia / condición:** dominio público (traducción del siglo XIX). El ítem de Internet Archive es copia del Project Gutenberg eBook #57303, liberado el 10 de junio de 2018.
- **Formato descargado:** texto plano (`57303-8.txt`, ISO-8859-1 / CP1252), 726 601 bytes.
- **Nota de procedencia:** la obra fue aprobada para el curso; el archivo descargado corresponde al espejo de Gutenberg en Internet Archive. El texto coincide con la edición de elejandria revisada en la semana 1.

## Preparación del corpus

El texto crudo se conserva en `datos/raw/divina_comedia.txt` y **no se entrega** en el zip final. `codigo/preparar.py`:

1. decodifica CP1252;
2. recorta entre los marcadores `*** START ***` y `*** END ***` de Gutenberg;
3. toma desde `_CANTO PRIMERO_` hasta antes de `_INDICE_`;
4. elimina encabezados de canto, `[Ilustración]`, marcadores `[n]` y la palabra `FIN`;
5. normaliza a minúsculas, sustituye caracteres fuera del alfabeto por `#` y colapsa espacios;
6. fija $\mathcal{V}$ solo con entrenamiento y divide en bloques contiguos 80 % / 10 % / 10 %.
