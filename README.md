# Bricks to Taxis AI

Proyecto de **IIC2613 · Inteligencia Artificial · Tarea 1 (2026-2)**.

La parte **DCCubos / LEGO** contiene un notebook ejecutado en Colab que compara clasificación visual, tabular y combinada, junto a un borrador de análisis de sus resultados reales. **DCCUber / Taxi** implementa Q-Learning tabular, los cinco experimentos obligatorios y el bonus DCCampeonato. Los informes finales `respuesta.pdf` de ambas partes quedan para la redacción y revisión del estudiante.

## Empezar en Colab

1. Abre [Google Colab](https://colab.research.google.com/) y sube [Tarea1_LEGO.ipynb](01%20-%20DCCubos/Tarea1_LEGO.ipynb).
2. Sube [piezas_dataset.csv](01%20-%20DCCubos/piezas_dataset.csv) al panel de archivos. El notebook también lo solicita si falta.
3. Selecciona un entorno **CPU** y ejecuta las celdas en orden. Estos modelos de scikit-learn no utilizan automáticamente una GPU.
4. Espera la descarga de Kaggle: aproximadamente 1 GB según el enunciado. Si exige autenticación, ejecuta `import kagglehub; kagglehub.login()` y repite la carga. No guardes credenciales en el notebook.
5. Revisa máscaras, tablas y figuras. El bloque final genera `resultados_LEGO.zip`, con un borrador PDF y toda la evidencia.
6. Descarga también **el notebook con sus salidas** desde Archivo → Descargar → .ipynb.
7. Revisa el borrador, añade tu interpretación y el anexo de IA y produce `respuesta.pdf`.

La caché y los resultados locales de Colab se pierden al cerrar la sesión. Cada ejecución crea una carpeta propia en `resultados/` para no mezclar evidencia anterior. Puedes ajustar `BASE_DIR` y `LOCAL_IMAGES` para una carpeta persistente de Drive ya montada.

## Estructura

```text
01 - DCCubos/
  Tarea1_LEGO.ipynb       Desarrollo autocontenido, decisiones y referencias
  piezas_dataset.csv     Original, sin modificar
  GUIA_ENTREGA.md         Rúbrica, evidencia y revisión final
  ANEXO_IA.md             Registro de asistencia
  ANALISIS_RESULTADOS.md  Borrador de análisis de la ejecución real
02 - DCCUber/
  TaxiAgent.py            Q-table, política ε-greedy y actualización Q
  train.py                Entrenamiento, experimentos y bonus DCCampeonato
  q_table_campeonato.csv  Tabla del bonus, 500 estados y 6 acciones
  main.py, test.py        Plantillas originales sin cambios de contenido
  GUIA_EJECUCION.md       Comandos y archivos de evidencia
  ANALISIS_TAXI_BORRADOR.md  Borrador teórico y experimental
material_profesor/
  T1_IIC2613_2026_2.pdf   Enunciado original
  T1_Clasificación.ipynb  Notebook original de referencia
docs/plan-lego.md         Diseño y alcance
tests/test_lego.py        Pruebas metodológicas y de preprocesamiento
tests/smoke_lego.py       Comprobación integral con imágenes sintéticas
requirements-lego.txt     Dependencias de ejecución
requirements-dev.txt      Dependencias de validación
requirements-taxi.txt     Dependencias de DCCUber
tests/test_taxi_*.py      Pruebas del agente y experimentos
```

`resultados/`, `cache/`, imágenes, entornos virtuales y pruebas temporales se excluyen de Git. Tras una ejecución real se pueden versionar resultados seleccionados con `git add -f`, comprobando primero que correspondan al mismo experimento.

## Protocolo experimental

- Partición estratificada 80/20 y cinco folds compartidos por A/B/C. Copias visuales exactas se mantienen juntas.
- Imputación, escalado y selección aprendidos exclusivamente dentro de cada fold.
- SVM RBF, Random Forest, árbol y KNN: **tres configuraciones por modelo y track, 36 en total**.
- A: HOG y seis medidas de silueta. B: cinco variables físicas/económicas. C: selección visual dentro del fold y combinación.
- `filename` nunca entra como predictor; `category` solo aparece en un diagnóstico separado del ranking.
- Ranking top 10 por validación, evaluación de los doce ganadores por familia/track y del top 3 global, con matrices de confusión.

Las decisiones y limitaciones se explican junto al código. En C también cambian la selección y el peso de los bloques: una mejora no se atribuye únicamente a la fusión.

## Pruebas locales

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lego.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tests/smoke_lego.py
```

El smoke test ejecuta las celdas de código en orden con 160 imágenes artificiales y bosques reducidos, guardando todo exclusivamente en `tmp/`. **Sus métricas no son resultados de LEGO y no se entregan.** No sustituye la ejecución real en Colab.

## DCCUber obligatorio

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-taxi.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_taxi_*.py' -v
.\.venv\Scripts\python.exe '02 - DCCUber/train.py' --experiments
```

El ejecutor entrena Taxi-v4 en CPU y guarda curvas, métricas por episodio, comparaciones y la Q-table del agente final en `02 - DCCUber/resultados/<fecha>/`. La [guía de DCCUber](02%20-%20DCCUber/GUIA_EJECUCION.md) indica cómo leerlos; [el borrador de análisis](02%20-%20DCCUber/ANALISIS_TAXI_BORRADOR.md) conecta las cifras con las preguntas de la tarea. El estudiante debe redactar y revisar su `respuesta.pdf`, con referencias y anexo de IA.

Para regenerar la tabla opcional del campeonato, entra en `02 - DCCUber/` y ejecuta `..\.venv\Scripts\python.exe train.py --campeonato`. La función `DCCampeonato()` entrena exactamente 650 episodios y escribe `q_table_campeonato.csv` en el directorio de ejecución.

## Entrega

En `01 - DCCubos/`: notebook ejecutado, CSV original y `respuesta.pdf` con respuestas y anexo de IA. En `02 - DCCUber/`: `TaxiAgent.py`, `train.py` y su propio `respuesta.pdf`. El ZIP final de la tarea contiene ambas carpetas. No incluir imágenes, cachés, entorno virtual ni pruebas sintéticas. Consulta [GUIA_ENTREGA.md](01%20-%20DCCubos/GUIA_ENTREGA.md) y [GUIA_EJECUCION.md](02%20-%20DCCUber/GUIA_EJECUCION.md).

La tarea es individual. El enunciado permite IA con declaración de uso, prompts y respuestas relevantes y adaptación sustancial a los resultados propios.
