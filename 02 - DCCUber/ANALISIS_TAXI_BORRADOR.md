# DCCUber: borrador de análisis para elaborar `respuesta.pdf`

Este texto desarrolla la Parte 2 obligatoria con los resultados de `resultados/obligatorios-20261001-v2/`. Es material de trabajo: el estudiante debe comprobar las cifras y figuras, adaptar la redacción y preparar su propio informe con declaración y anexo de IA. La corrida anterior, `obligatorios-20261001/`, está marcada como inválida y no se cita. El bonus DCCampeonato se implementó después y se documenta por separado al final.

## Protocolo y lectura de las métricas

La Q-table tiene 500 estados por seis acciones. Cada condición empieza con una tabla a cero y usa estados iniciales reproducibles. La curva representa la **recompensa media de bloques de 100 episodios de entrenamiento**; su último valor no equivale al rendimiento de una política que siempre elige la mejor acción, porque durante el entrenamiento se sigue explorando. Para separar ambas cosas se evalúa la política codiciosa en 100 episodios con semillas comunes. Se registra como éxito la entrega efectiva del pasajero (`terminated=True`). El agente final se vuelve a entrenar con otra semilla y se evalúa en otras 100 situaciones. Una sola corrida de entrenamiento por condición permite describir esta ejecución, pero no estimar con precisión la variabilidad entre semillas.

La actualización implementada es `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`. El término futuro es cero al terminar por entrega exitosa. Si se alcanza el límite externo de 200 pasos, el episodio se trunca: se detiene la simulación, pero el siguiente estado conserva valor para el aprendizaje. Esto sigue la [distinción de Gymnasium entre terminación y truncación](https://gymnasium.farama.org/main/tutorials/handling_time_limits/).

## Actividad 1. Comprender los hiperparámetros

### 1. Tasa de aprendizaje y descuento

**1.a. α=0,1 frente a α=1.** α determina qué fracción del error temporal corrige cada actualización. Con 0,1, la tabla incorpora gradualmente la observación nueva: puede necesitar más episodios, pero evita sustituir todo lo aprendido por una sola transición. Con 1, el valor anterior se reemplaza por el objetivo temporal completo. En un entorno determinista podría avanzar rápido, aunque con estimaciones futuras todavía imperfectas, cambios de política o ruido sería más sensible e inestable. La comparación solo fija γ; el resultado concreto depende también de ε y de los episodios disponibles.

**1.b. γ=0,1 frente a γ=1.** γ pondera la recompensa futura. Con 0,1, una recompensa que llega varios pasos después influye poco en la decisión actual; Taxi penaliza cada movimiento y recompensa la entrega al final, de modo que este valor puede dificultar aprender el trayecto. Con 1, la entrega lejana no se descuenta por distancia temporal y puede propagarse hacia las decisiones de recogida y navegación. Eso no garantiza mejores resultados en cualquier tarea: también cambia la escala de los valores y puede hacer más difícil valorar trayectorias largas si hay incertidumbre.

### 2. Comparación de los agentes A y B del enunciado

**2.a. Rapidez y estabilidad.** A usa α=0,9 y B α=0,2. En principio A cambia la Q-table más deprisa después de cada experiencia; B integra cada transición de forma más gradual y suele tener una curva menos sensible a una observación aislada. No es una garantía de que B obtenga más recompensa en todos los puntos de la curva.

**2.b. Consecuencias futuras.** B, con γ=0,8, concede mucho más peso a una entrega futura que A, con γ=0,1. En Taxi eso favorece elegir una ruta que acerque al pasajero y luego al destino aunque los primeros pasos den −1. A tiende a valorar sobre todo el costo inmediato de moverse.

**2.c. Expectativa tras 10.000 episodios.** Esperaría que B desarrollase una política final más útil: el premio de entrega puede propagarse y α=0,2 permite consolidar experiencias. A podría cambiar la tabla rápido, pero γ=0,1 hace difícil que una acción temprana reciba crédito por una entrega lejana. Es una predicción razonada para esos parámetros, no un resultado medido de A y B como pareja: en esta ejecución se aislaron γ y α en experimentos separados.

**2.d. Aplicación a rutas.** El taxi de A podría reaccionar enseguida a castigos o premios cercanos, sin valorar suficientemente que varios movimientos con −1 llevan a una entrega de +20. B puede transmitir esa recompensa hacia estados previos y aprender rutas más eficientes a largo plazo, aunque actualizará cada entrada de la tabla con menos fuerza y podría necesitar más experiencias.

### 3. Exploración ε-greedy

**3.a. Exploración y explotación.** En cada decisión de entrenamiento se elige una acción aleatoria con probabilidad ε; con probabilidad `1−ε` se elige una acción de máximo valor Q. Explorar descubre consecuencias aún desconocidas y explotar utiliza las estimaciones actuales. En evaluación se fija la política codiciosa sin exploración.

**3.b. Decay próximo a 1.** Para la fórmula del enunciado `ε_n = ε_inicial × decay^n`, un `decay` próximo a 1 reduce ε lentamente: la búsqueda aleatoria persiste durante muchos episodios. Esta fórmula es la del ejercicio teórico. El código de la plantilla usa otra parametrización, `ε_min + (ε_max−ε_min) exp(−tasa × episodios)`, donde una **tasa más grande** provoca una caída más rápida; no deben confundirse los dos parámetros.

**3.c. Cálculo.** Con ε inicial 1 y `decay=0,995`, tras 1.000 episodios `ε = 0,995^1000 ≈ 0,006654`, es decir, alrededor de 0,665 % de probabilidad de explorar en cada decisión.

**3.d. Relación entre α y ε.** ε controla qué experiencias se generan; α determina cuánto cambian los valores Q cuando llegan esas experiencias. Explorar mucho con α muy pequeño puede visitar estados sin consolidar lo aprendido con rapidez. Explorar poco mientras la tabla es inmadura puede fijar una política pobre aunque α sea alto. Por eso se observan conjuntamente la curva de entrenamiento y una evaluación sin exploración.

## Actividad 3. Experimentos

### Experimento 1: agente miope y previsor

En `gamma.png`, γ=0 mejora muy poco respecto del comportamiento inicial y termina con mean score −202,15 en el último bloque; su evaluación codiciosa acaba los 100 episodios por límite de pasos (0 % de éxito, score −200). γ=1 sube mucho más rápido y termina con 7,15 en el último bloque; resuelve los 100 episodios de evaluación, con score medio 7,86 y 13,14 pasos. Los tres valores intermedios confirman una progresión en esta ejecución: γ=0,25, 0,5 y 0,75 logran respectivamente 49 %, 54 % y 91 % de éxito codicioso. γ=0,75 obtiene 6,94 en el último bloque y γ=1 obtiene 7,15. La explicación es que la entrega está diferida: con γ bajo, su recompensa se atenúa antes de influir en los primeros movimientos. Para estas condiciones elegiría γ=1, con la cautela de que la comparación usa una semilla de entrenamiento y este entorno concreto.

### Experimento 2: velocidad frente a estabilidad

`learning_rate_blocks.csv` contiene media y récord de **cada** bloque de 100. Al final de 5.000 episodios, α=0,01 registra mean score −73,47; α=0,7 registra 6,96. El récord de ambos entrenamientos llega a 15, por lo que el mejor episodio aislado no distingue una política sostenida. La curva de `learning_rate.png` muestra que α=0,7 alcanza recompensas cercanas a positivas mucho antes; α=0,01 progresa lentamente y sigue lejos al final. La evaluación codiciosa es más decisiva: 1 % de éxito y score −197,87 para α=0,01, frente a 100 % y 7,86 para α=0,7. Con 5.000 episodios elegiría 0,7 entre estos dos valores. Que 0,7 funcione bien aquí no demuestra que toda tasa alta sea estable ni que sea óptima para otros entornos.

### Experimento 3: sabotear la exploración

Con decaimiento prematuro (`epsilon_decay_rate=0,05`), la curva sube pronto y el último bloque obtiene 6,56, pero la política codiciosa solo resuelve 82 de 100 episodios, con score medio −28,91. Al dejar de buscar nuevas acciones demasiado temprano, algunas rutas quedan poco aprendidas. Con exploración perpetua (`ε_min=ε_max=0,8`), el último bloque de entrenamiento cae a −276,64 porque el agente sigue tomando acciones aleatorias en 80 % de las decisiones; aun así, **su evaluación codiciosa resuelve 100/100** y alcanza 7,86. Este contraste es esencial: la recompensa observada durante entrenamiento mide también el costo de explorar, no solo la calidad de la tabla aprendida. Mantener exploración alta da cobertura de estados, pero impide aprovechar la política mientras está activa. Una reducción gradual permite recopilar información al principio y explotar lo aprendido después.

### Experimento 4: la curva de la experiencia

Las cinco corridas usan idéntica configuración base y semilla. Por ello las curvas de `duration.png` **se superponen exactamente hasta que acaba la más corta**; no son cinco réplicas independientes. `duration_summary.csv` ofrece una comparación más legible de sus puntos finales:

| Episodios | Mean score del último bloque | Máximo de entrenamiento | Éxito codicioso | Score codicioso |
|---:|---:|---:|---:|---:|
| 500 | −323,34 | 2 | 1 % | −197,87 |
| 2.500 | 4,07 | 15 | 62 % | −70,18 |
| 5.000 | 6,62 | 15 | 84 % | −24,99 |
| 10.000 | 7,19 | 15 | 89 % | −14,64 |
| 20.000 | 7,37 | 15 | 98 % | 3,78 |

El máximo de un episodio llega a 15 ya con 2.500 episodios, pero eso no significa que la política sea fiable: solo resuelve 62 % de las evaluaciones codiciosas. La recompensa media de entrenamiento crece mucho entre 500 y 2.500, luego sus ganancias son menores; de 10.000 a 20.000 solo sube 0,18 puntos en el último bloque. Sin embargo, la política codiciosa todavía mejora de 89 % a 98 % de éxito en ese tramo. Por tanto, el aparente estancamiento de una métrica no demuestra que más episodios sean inútiles, ni que siempre vayan a aportar una mejora proporcional. La tabla corresponde a una semilla y a 100 episodios de evaluación por duración.

### Experimento 5: agente elegido

Entre las condiciones aisladas, γ=1 y α=0,7 obtuvieron 100 % de éxito en las 100 evaluaciones codiciosas de selección; entre los presupuestos de entrenamiento, 20.000 episodios alcanzaron el mejor éxito (98 %). Con ese criterio se entrenó **una tabla nueva**, sin reutilizar las anteriores, con γ=1, α=0,7, el decaimiento gradual original de ε y 20.000 episodios. Se eligió el decaimiento original porque los dos cambios del experimento 3 estaban diseñados como sabotajes y mostraron problemas distintos. El agente nuevo usa una semilla de entrenamiento diferente; su evaluación final usa 100 semillas que tampoco participaron en la elección.

En `ideal.png`, el mean score pasa de −722,51 en el primer bloque a 5,01 en el bloque 2.500, llega a 8,00 en el bloque 10.000 y termina en 7,26. El récord global es 15. La evaluación final codiciosa resuelve **100/100** episodios, con recompensa media 7,94 y 13,06 pasos medios. Estas cifras sostienen que la política funciona muy bien en las situaciones evaluadas, pero no prueban un agente «perfecto» para todos los estados ni para variantes estocásticas, pasajeros cambiantes o distintos límites. Para mejorar la confianza, repetiría el entrenamiento con varias semillas y compararía la distribución de éxito; después examinaría los estados de salida difíciles y probaría una política de exploración que concentre visitas allí, sin volver a elegir hiperparámetros usando la evaluación final.

## Actividad 4. SlimeVolleyGym: respuestas conceptuales

**1. Espacios de estado.** Taxi-v4 entrega un entero entre 0 y 499; ese entero indexa una fila de la Q-table, con seis columnas de acciones. SlimeVolley-v0 entrega un vector continuo de 12 números que describe posiciones y movimiento implícito del agente, rival y pelota. Una política para SlimeVolley debe funcionar con valores nuevos entre los ya vistos, no solo recuperar una fila exacta. [Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/), [SlimeVolleyGym](https://github.com/hardmaru/slimevolleygym).

**2. Por qué no basta la misma tabla.** Cada componente continuo puede tomar muchísimos valores; enumerar combinaciones de 12 dimensiones produciría una tabla enorme, y la mayoría de sus filas nunca se visitarían. Redondear todas las coordenadas para forzar índices discretos perdería precisión y seguiría produciendo muchas combinaciones. El problema no es solo memoria: estados muy parecidos quedarían desconectados si no comparten exactamente el mismo índice.

**3. Por qué redes neuronales.** Una red usa un número finito de parámetros compartidos para aproximar una función sobre el espacio continuo, por lo que puede generalizar entre observaciones próximas. El repositorio oficial incluye políticas neuronales y experimentos con PPO, evolución y self-play; esto muestra una forma práctica de aprender decisiones sin enumerar estados. La red no garantiza por sí sola aprendizaje fácil: la recompensa de SlimeVolley es más escasa y depende de un rival. [Repositorio y entrenamiento oficiales](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md).

**4. Qué reemplaza a la Q-table.** En Taxi, la tabla asigna explícitamente valores a cada par `(estado, acción)`. En SlimeVolley, una red recibe el vector de 12 números y produce una representación de decisiones, por ejemplo probabilidades para los tres controles binarios de `MultiBinary(3)` en una política como PPO, o valores estimados para acciones en una arquitectura basada en Q. Una red de política también puede estimar el valor del estado mediante otra salida o componente. La misma función parametrizada sirve a muchos vectores de entrada sin almacenar una fila por cada uno.

**5. Ventaja concreta.** Si la pelota aparece en una posición ligeramente distinta de otra ya observada, la red puede reutilizar patrones aprendidos para reaccionar de forma parecida. Una Q-table indexada por el vector exacto no tendría una entrada entrenada para esa nueva observación. Esta capacidad de generalizar es especialmente útil en el espacio continuo de SlimeVolley.

## Bonus DCCampeonato (opcional)

`DCCampeonato()` entrena una tabla nueva durante 650 episodios, con `gamma=0.99`, `alpha=0.8`, decaimiento de exploración `0.005`, exploración mínima `0.05` y semilla `42`. El CSV `q_table_campeonato.csv` tiene 500 filas, 7 columnas y ninguna cabecera. Al volver a cargar el CSV y evaluar la política codiciosa en 200 episodios con semillas independientes (`300001` a `300200`), completó 187 viajes (93,5 %). Esta prueba local no garantiza la posición en el torneo ni sustituye los resultados obligatorios, entrenados con otro presupuesto.

## Referencias para el informe del estudiante

- Enunciado original: `material_profesor/T1_IIC2613_2026_2.pdf`, páginas 8–15.
- Farama Foundation, [documentación de Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/) y [manejo de límites de tiempo](https://gymnasium.farama.org/main/tutorials/handling_time_limits/).
- D. Ha, [Slime Volleyball Gym Environment](https://github.com/hardmaru/slimevolleygym), GitHub, 2020; [ejemplos de entrenamiento](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md).
