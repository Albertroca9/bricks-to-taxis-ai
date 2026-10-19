# Guía de ejecución, interpretación y entrega

El notebook incluye el código de LEGO, decisiones, visualizaciones y exportación. El informe se genera como **borrador** con resultados reales de tu ejecución. Antes de entregar debes revisar las salidas, completar la autoría, desarrollar las interpretaciones marcadas y adjuntar las conversaciones de IA relevantes.

## Qué comprobar en Colab

1. Se encuentran las 2.400 imágenes y las ocho clases. Resuelve cualquier error de archivos faltantes antes de entrenar.
2. `particion.csv` separa train/test y `folds.csv` fija la validación. Si hay duplicados visuales, se agrupan y el 80/20 puede ser aproximado.
3. Revisa originales, máscaras y HOG. `auditoria_mascaras.csv` marca áreas extremas; una máscara errónea puede contaminar A y C.
4. Hay 12 filas en cada `validacion_A/B/C.csv` y 36 en `ranking_36_validacion.csv`.
5. `seleccion_antes_del_test.json` registra candidatos antes de evaluar test.
6. `comparacion_12_finalistas.csv` contiene los cuatro modelos de cada track. El top 3 global puede añadir candidatos: `evaluacion_test.csv` puede tener más de 12 filas.
7. Las tres matrices usan el mismo test. `comparacion_category.csv` permanece separado del ranking.
8. Descarga el ZIP y el notebook ejecutado antes de cerrar Colab.

## Correspondencia con el enunciado

| Pregunta | Evidencia | Interpretación propia |
|---|---|---|
| A.a | Máscaras/HOG e histogramas geométricos | Cada descriptor, piezas concretas y solapamientos |
| A.b | Pipelines y explicación de StandardScaler | Unidades, distancias/márgenes y árboles |
| A.c | Partición, folds y conteos | Representatividad, reproducibilidad y reserva del test |
| A.d | Validación A y comparación final | Mejor modelo, funcionamiento y brechas |
| A.e | Tres configuraciones de cada familia | Un cambio concreto de hiperparámetro con cifras |
| A.f | Tabla A ordenada por CV | Mejor/peor, distinguiendo familia y configuración |
| B.a | Auditoría de columnas, distribuciones y atípicos | Problemas específicos de las once columnas |
| B.b | Faltantes antes/después y reglas | Razón de cada limpieza y límites de la imputación |
| B.c | Diagnósticos auxiliares y category | Diferencia con/sin category, sin presuponer aumento |
| B.d | Tabla B, gráficos y test | Mejor/peor, parámetros y generalización |
| C.a | Selección visual C y pipeline | Información complementaria y redundancia |
| C.b | Validación C y ganador por track | Mejor/peor de C y comparación con A/B |
| C.c | Comparación entre tracks y tiempos | Coste, ruido y límites para atribuir efectos a la fusión |
| C.d | Tres matrices, confusiones y ejemplos | Pares concretos y explicación visual plausible |

## Convertir el borrador en una respuesta propia

El PDF automático reúne metodología, cifras y figuras; el Markdown contiene el mismo contenido para editarlo. Puedes usar tu editor preferido o completar `REPORT_AUTHOR`, `PERSONAL_ANALYSIS` y `AI_ANNEX` en las celdas finales del notebook. Con esos campos completos y `FINAL_REPORT=True`, vuelve a ejecutar desde esa celda hasta la exportación para generar `respuesta.pdf` sin repetir los entrenamientos. Mantén tablas y figuras legibles.

En cada interpretación indica qué observas, una cifra/figura concreta, una explicación compatible con las piezas y el algoritmo y qué no permite concluir el experimento. Si no hay diferencias o errores, dilo: no inventes mejoras por category ni confusiones inexistentes.

Si cambias parámetros después de mirar test, ese test pasa a formar parte del desarrollo. Documenta las iteraciones y utiliza una evaluación independiente.

Conserva las referencias formales del notebook/informe, semilla, partición y versiones. Los renders no representan toda la variabilidad de cámara, luz y fondos de una fábrica real.

## Archivos finales

```text
01 - DCCubos/
  Tarea1_LEGO.ipynb       Todas las celdas ejecutadas, sin errores
  piezas_dataset.csv     Original
  respuesta.pdf          A.a–f, B.a–d, C.a–d y anexo de IA
```

El ZIP de evidencia no es el ZIP final de la tarea. No enviar imágenes, cachés, `.venv/`, `tmp/` ni resultados sintéticos. Taxi se incorporará cuando se complete esa parte.
