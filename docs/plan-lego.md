# Plan de LEGO y decisiones de diseño

Fecha: 2026-09-22. Alcance: DCCubos; Taxi se conserva sin cambios.

## Objetivo

Entregar un notebook autocontenido para Colab, didáctico y reproducible, que cubra los tres tracks del enunciado. El usuario ejecutará el entrenamiento con imágenes reales en Colab. Los resultados y el informe final se obtendrán de esa ejecución, nunca de datos simulados.

## Decisiones

- Notebook único y CSV junto a él: solo dos archivos para iniciar Colab. Las funciones están en celdas explicadas; no necesita importar módulos del repositorio.
- Conservar la plantilla original en `material_profesor/` y trabajar en `01 - DCCubos/Tarea1_LEGO.ipynb`. El CSV se traslada, sin alterar bytes, junto al notebook final.
- Partición estratificada 80/20 y cinco folds idénticos en los tres tracks. El test se abre después de cerrar las selecciones y el ranking.
- Selección supervisada de características dentro de cada fold: evita que las etiquetas de validación determinen previamente qué variables se conservan.
- Track A: HOG y seis medidas de silueta. Track B: cinco medidas físicas/económicas, con auditoría de variables auxiliares. Track C: combinar ambas fuentes y seleccionar dentro de la validación.
- KNN como cuarto modelo: comparación comprensible entre distancias, márgenes y árboles. Doce configuraciones fijas, reutilizadas sin variaciones en A/B/C.
- Experimento adicional con category, separado del ranking y con hiperparámetros del ganador de B fijados antes de abrir test.
- Reglas físicas deterministas para valores inválidos; imputación y escalado dentro del pipeline. No imputar utilizando la clase real.
- Exportar datos de cada fold, predicciones, partición, parámetros, versiones, figuras y un borrador de informe con evidencia. No presentar el borrador como respuesta personal final.
- CPU: implementaciones de scikit-learn; Colab GPU no implica aceleración automática.

## Ejecución y verificación

- [x] Crear estructura, documentación y conservar originales.
- [x] Escribir pruebas de integridad del cruce, limpieza, partición y aislamiento del test.
- [x] Desarrollar carga, auditoría, características y validación de los tres tracks.
- [x] Desarrollar gráficos, ranking, evaluación final y borrador de informe.
- [x] Ejecutar pruebas sobre CSV real y casos sintéticos de imágenes (9 pruebas aprobadas).
- [x] Ejecutar las 25 celdas funcionales en orden con un fixture sintético claramente marcado, verificando 36 candidatos, 12 finalistas, matrices, PDF y ZIP. La celda de instalación se omite porque las dependencias ya están instaladas.
- [x] Revisar PDF generado y comprobar cobertura de todas las preguntas del enunciado.

## Revisión independiente

La revisión no detectó leakage ni omisiones del protocolo obligatorio. Se corrigió la mezcla potencial de artefactos de distintas ejecuciones usando subdirectorios con identificador temporal y se eliminaron opciones de configuración sin efecto. ANOVA trata explícitamente las coordenadas constantes de fondo, con una prueba de regresión.

## Límites y revisión del estudiante

La inspección y el entrenamiento completos con las imágenes de Kaggle quedan para Colab. El estudiante debe revisar máscaras, resultados, ejemplos de error, citas y explicaciones, completar autoría y anexar las conversaciones de IA relevantes. No se afirmará que se ha validado la descarga real o el rendimiento sobre LEGO antes de hacerlo.
