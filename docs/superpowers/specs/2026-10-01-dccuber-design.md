# Diseño de DCCUber obligatorio

Fecha: 2026-10-01. Aprobado en conversación: un ejecutor reproducible en `train.py`, `main.py` y `test.py` sin cambios, tablas y gráficos para que el estudiante redacte el informe. El bonus DCCampeonato queda fuera.

## Objetivo y alcance

Completar la Parte 2 obligatoria del enunciado IIC2613 2026-2: implementar Q-Learning tabular para `Taxi-v4`, ejecutar los cinco experimentos descritos, conservar evidencia cuantitativa y preparar un borrador explicativo de las preguntas teóricas de Taxi y SlimeVolley. El estudiante redactará y revisará `respuesta.pdf`.

## Estructura y límites

- Mover los cuatro archivos originales a `02 - DCCUber/`, como exige la entrega. `main.py` y `test.py` conservan sus bytes y su comportamiento.
- Completar `TaxiAgent.py` con Q-table de 500×6, elección ε-greedy y actualización Q con estado terminal. Permitir inyectar hiperparámetros y semilla sin romper `TaxiAgent(env)`.
- Unificar en `train.py` el bucle de entrenamiento, registro por episodio y evaluación sin render. La función `train` mantiene la firma que utiliza `main.py`; una entrada de línea de comandos separada ejecuta los experimentos.
- Guardar en `resultados/<id de ejecución>/` un manifiesto JSON con configuración y versiones, datos por episodio, resumen por bloques de 100, tablas comparativas, gráficos y Q-table del agente elegido. La carpeta queda fuera de Git por volumen; los resultados de la ejecución que sustentan el informe se pueden seleccionar para la entrega.
- Añadir `requirements-taxi.txt`, pruebas locales y una guía de ejecución. No instalar SlimeVolleyGym: el enunciado pide investigación y explicación, no entrenar ese entorno.

## Protocolo experimental

- Entorno `gymnasium.make('Taxi-v4')`, sin render para entrenar. La documentación oficial especifica 500 estados, seis acciones, recompensa +20 al entregar, -10 por recogida/entrega ilegal y -1 por paso; episodios de hasta 200 pasos.
- Q-learning actualiza `Q[s,a] += α × (r + γ × max Q[s',:] - Q[s,a])`, con valor futuro cero al terminar por entrega exitosa. Al truncarse por el límite externo de 200 pasos se conserva el valor futuro, según la guía oficial de Gymnasium. ε-greedy explora durante entrenamiento y explota al evaluar. Un generador local y semillas de `env.reset` reproducen cada corrida.
- Cada configuración inicia una Q-table nueva; no se encadenan entrenamientos entre condiciones. Se registra recompensa total, pasos, terminación exitosa y ε por episodio. El score medio del enunciado es la recompensa media de los últimos 100 episodios; se añade evaluación codiciosa con un conjunto fijo de semillas para comparar políticas sin ruido de exploración.
- Experimento 1: γ=0 y γ=1, más tres valores internos, al menos 5.000 episodios por condición; gráfico de curvas conjuntas.
- Experimento 2: α=0,01 y α=0,7, al menos 5.000 episodios, tabla de score medio y máximo por bloques de 100.
- Experimento 3: decaimiento de ε prematuro (0,05) y exploración perpetua (ε mínimo=máximo=0,8), al menos 5.000 episodios; comparar curva de aprendizaje y política final.
- Experimento 4: presupuestos de 500, 2.500, 5.000, 10.000 y 20.000 episodios, con la misma configuración base; tabla de score final y máximo.
- Experimento 5: elegir una configuración a partir de experimentos anteriores, entrenar una Q-table nueva y guardar curva y evaluación final. La elección y sus razones se documentan sin inventar métricas antes de correr.

## Verificación y riesgos

- Pruebas pequeñas para dimensiones, política de exploración/explotación, actualización terminal y no terminal, exportación CSV y registros del bucle. Probar también que `main.py` y `test.py` no cambian.
- Smoke test corto del ejecutor antes de los experimentos completos. Después validar cantidades de episodios, archivos, formas y consistencia de las métricas.
- La recompensa de entrenamiento mezcla calidad de política y exploración; la evaluación codiciosa aclara esa diferencia. Un score total positivo no equivale exactamente a éxito: se usa `terminated` para contarlo.
- Los resultados de una semilla no garantizan una superioridad general. Las curvas y la evaluación permiten discutir variabilidad, estancamiento y compromisos sin afirmar causalidad más allá de los controles fijados.
- SlimeVolleyGym original usa la API antigua de Gym; para las preguntas conceptuales se cita su repositorio oficial y no se instala ni ejecuta.
