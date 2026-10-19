# DCCUber: ejecución y evidencia

Esta carpeta contiene la Parte 2 obligatoria y el bonus DCCampeonato. `TaxiAgent.py` implementa Q-Learning; `train.py` ejecuta los experimentos y el entrenamiento opcional de 650 episodios. `main.py` y `test.py` conservan el contenido original.

## Preparar el entorno

Desde la raíz del repositorio, en PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-taxi.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_taxi_*.py' -v
```

El algoritmo usa una Q-table pequeña, NumPy y el entorno `Taxi-v4`. No utiliza automáticamente la GPU. La prueba `--smoke` comprueba el flujo con diez episodios por condición; sus cifras **no** son evidencia académica.

```powershell
.\.venv\Scripts\python.exe '02 - DCCUber/train.py' --smoke
.\.venv\Scripts\python.exe '02 - DCCUber/train.py' --experiments
```

Cada ejecución crea una carpeta distinta en `02 - DCCUber/resultados/` con marca temporal UTC. Puede indicarse una ruta mediante `--output`. Para probar la interfaz humana de la plantilla original, entra en `02 - DCCUber/` y ejecuta `main.py`; esa ruta entrena 10.000 episodios y espera una pulsación antes de abrir la ventana. El ejecutor de experimentos evalúa sin render y no requiere esa ventana.

## Archivos de resultados

| Archivo | Contenido |
|---|---|
| `manifest.json` | Condiciones, semillas, versiones y criterio de elección del agente final |
| `episode_metrics.csv` | Recompensa, pasos, éxito y ε de cada episodio de entrenamiento |
| `block_metrics.csv` | Media, máximo, tasa de éxito y pasos por bloque de 100 |
| `evaluation_metrics.csv` | Evaluación codiciosa sobre semillas fijas |
| `summary.csv` | Resumen de cada condición y del agente final |
| `learning_rate_blocks.csv` | Tabla solicitada de media y récord cada 100 episodios para α |
| `duration_summary.csv` | Comparación de 500, 2.500, 5.000, 10.000 y 20.000 episodios |
| `gamma.png`, `learning_rate.png`, `exploration.png`, `duration.png`, `ideal.png` | Curvas de recompensa media por bloque |
| `q_table_ideal.npy`, `q_table_ideal.csv` | Política tabular del agente final; CSV con 500 filas y 7 columnas, sin cabecera |

El score medio de entrenamiento incluye las acciones exploratorias. Para valorar la política aprendida se usa también `eval_mean_score` y `eval_success_rate`, obtenidos con `training=False`. Un episodio cuenta como exitoso cuando el entorno retorna `terminated=True`; la recompensa total positiva no se usa como sustituto. Al truncar por 200 pasos, la actualización Q conserva el valor estimado del siguiente estado, aunque se detenga la simulación.

Cada condición empieza con una tabla nueva y usa la misma secuencia de semillas de estado inicial. Las condiciones normales se evalúan con 100 semillas fijas; el agente final se vuelve a entrenar con otra semilla y se evalúa con otras 100. Una sola semilla de entrenamiento por condición limita las conclusiones sobre estabilidad. Los resultados con `SMOKE_ONLY.txt` o `INVALIDATED.txt` no se deben citar.

## Bonus DCCampeonato

Desde `02 - DCCUber/` ejecuta:

```powershell
..\.venv\Scripts\python.exe train.py --campeonato
```

Esta ruta llama a `DCCampeonato()`, entrena exactamente 650 episodios con una semilla y configuración fijas y escribe `q_table_campeonato.csv` en el directorio de ejecución. El CSV no lleva cabecera: tiene 500 filas y 7 columnas, con el estado `0` a `499` seguido de los seis valores Q en el orden de acciones de Taxi-v4. La función también se puede importar mediante `from train import DCCampeonato`; no hace falta modificar `main.py`. El CSV está separado de `q_table_ideal.csv`, que procede del experimento obligatorio de 20.000 episodios y no cumple el límite del campeonato.

## Preparar `respuesta.pdf`

El estudiante redactará el informe final con las cifras de la ejecución válida, las tablas y figuras, las respuestas teóricas de Taxi y SlimeVolley, referencias y la declaración/anexo de uso de IA exigidos por el enunciado. `ANALISIS_TAXI_BORRADOR.md` ofrece una base de análisis para contrastar y adaptar. No basta con adjuntar gráficos sin explicar los mecanismos y limitaciones.

Fuentes técnicas: [Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/), [truncación y terminación](https://gymnasium.farama.org/main/tutorials/handling_time_limits/), [SlimeVolleyGym](https://github.com/hardmaru/slimevolleygym).
