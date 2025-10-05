# Taller-4
Leydy Sossa_Milena Silva_ De Regresión Lineal a Machine Learning

**El Dilema del Ajuste: Sobreajuste y Subajuste**

1. Entrenas un modelo y obtienes un 99% de exactitud sobre los datos de entrenamiento, pero solo un 75% sobre los datos de prueba. ¿Qué problema indica este resultado y por qué?
**RTA** Este resultado indica sobreajuste (overfitting). El modelo aprendió demasiado bien los datos de entrenamiento, al punto de memorizar detalles irrelevantes (ruido, particularidades no generalizables). Por eso logra un desempeño casi perfecto en entrenamiento (99%), pero falla con datos nuevos (75%) porque no generaliza adecuadamente. Es como un estudiante que memorizó las respuestas de la guía, pero no entiende el tema y se equivoca en el examen real.
2. Si el error de tu modelo es muy alto tanto en el conjunto de entrenamiento como en el de validación, ¿cuál es el problema más probable? ¿Creerías que añadir más datos de entrenamiento solucionaría el problema?
**RTA** El problema más probable es subajuste (underfitting). El modelo es demasiado simple o con poca capacidad para capturar la complejidad de los datos. Por eso, ni siquiera logra un buen desempeño en el entrenamiento. En este caso, añadir más datos no soluciona el problema, porque el modelo no tiene la capacidad suficiente para aprender los patrones. La solución sería usar un modelo más complejo, ajustar mejor los hiperparámetros o incorporar variables más relevantes.

**El Dilema del Modelo y la Regularización Ridge y Lasso**

1. En un problema para predecir fallas en una máquina, tienes 100 variables provenientes de sensores, pero sospechas que solo unas pocas son realmente importantes. ¿Usarías Ridge o Lasso? Justifica tu respuesta. **RTA** La mejor opción es Lasso (L1). Lasso tiene la capacidad de llevar algunos coeficientes exactamente a cero, eliminando las variables irrelevantes. Esto permite quedarnos solo con las variables de sensores realmente significativas y descartar el ruido o redundancia. En cambio, Ridge solo reduce el valor de los coeficientes, pero no elimina ninguno, lo que haría más difícil interpretar qué sensores son realmente importantes.

2. Si entrenas un modelo Lasso y aumentas gradualmente el valor del hiperparámetro de penalización (λ), ¿qué efecto esperarías observar en los coeficientes del modelo?
**RTA** A medida que λ aumenta:Cada vez más coeficientes se reducen y algunos se vuelven exactamente cero. El modelo se simplifica progresivamente, quedándose solo con las variables más influyentes. Si λ es muy grande, el modelo puede terminar con muy pocos predictores o incluso con todos en cero, perdiendo capacidad de predicción.

3. Al ejecutar el código de regularización 3D, ¿qué sucede con los coeficientes del modelo a medida que aumenta el valor de λ? ¿Qué interpretación le das a la forma diferente en que Ridge y Lasso aplican sus penalizaciones?
**RTA** Lo que sucede: En Ridge (L2): los coeficientes se reducen gradualmente hacia valores más pequeños, pero rara vez se vuelven cero. El modelo mantiene todas las variables, pero con menor peso. En Lasso (L1): los coeficientes se encogen y varios se vuelven exactamente cero, lo que equivale a una selección automática de variables. Interpretación: Ridge es útil cuando creemos que casi todas las variables aportan algo y queremos evitar la inestabilidad por multicolinealidad. Lasso es útil cuando sospechamos que muchas variables son ruido y necesitamos simplificar el modelo, identificando las más relevantes.

**GridSearchCV: Encontrando la Mejor Configuración para tu Modelo**
1. Quieres optimizar un modelo Ridge y pruebas manualmente alpha=10, obteniendo un buen resultado. ¿Por qué sigue siendo metodológicamente superior usar GridSearchCV en lugar de quedarte con ese valor?
**RTA** Porque probar un solo valor manualmente no garantiza que sea el mejor. GridSearchCV evalúa de manera sistemática varios valores de alpha en una rejilla predefinida. Utiliza validación cruzada (k-fold), lo que asegura que el rendimiento medido no dependa de una división aleatoria de datos, sino de múltiples particiones. Así, se obtiene un valor óptimo más robusto, confiable y generalizable para nuevos datos.

2. Además del modelo en sí (ej. Lasso()), ¿cuáles son los dos componentes principales que debes proporcionar a GridSearchCV para iniciar la búsqueda de hiperparámetros?
**RTA** Los dos componentes clave son: La rejilla de parámetros (param_grid) → lista de valores de hiperparámetros a probar (ej. diferentes valores de alpha). La estrategia de validación cruzada (cv) → define en cuántas particiones (folds) se dividirán los datos para evaluar cada configuración.

3. Si GridSearchCV selecciona un alpha muy pequeño (cercano a cero) como el mejor parámetro para tu modelo, ¿qué te sugiere esto sobre el nivel de sobreajuste que tenía tu modelo original sin regularizar?
**RTA** Significa que tu modelo original no sufría un problema grave de sobreajuste. Si el mejor alpha ≈ 0, la regularización apenas fue necesaria. Esto indica que los datos eran suficientemente representativos y que el modelo ya generalizaba bien sin requerir una penalización fuerte. En otras palabras: el modelo estaba balanceado y la regularización solo ajustó ligeramente los coeficientes.

**Construir un Árbol de Decisión: El Diagrama de Flujo Inteligente para la Optimización de Procesos**

1. En un árbol de decisión para optimizar la logística de un almacén, ¿qué podría representar un nodo hoja?
**RTA** Un nodo hoja representa la predicción final o decisión concreta después de evaluar todas las condiciones previas. En logística de un almacén, un nodo hoja podría indicar, por ejemplo:"Ruta óptima de despacho: Ruta 3". "Tiempo estimado de entrega: 24 horas". "Asignar a transportadora A". "Ubicar el producto en la zona de almacenamiento refrigerada". En resumen: el nodo hoja es la acción final o clasificación logística obtenida tras el análisis de todas las variables del proceso (inventario, rutas, tiempos, demanda, etc.).
2. Un ingeniero crea un árbol para predecir fallos en una máquina. El árbol es extremadamente profundo y tiene reglas muy específicas como "Si la temperatura es 75.3°C y la vibración es 0.152 m/s² y el operador es Juan...". ¿Qué problema de ajuste es este y por qué no sería fiable en la práctica diaria de la planta?
 **RTA** El problema es sobreajuste (overfitting). El árbol se volvió tan complejo que memoriza condiciones muy específicas de los datos históricos (ruido o coincidencias) en lugar de aprender patrones generales. En la práctica diaria de la planta, este modelo no sería fiable porque no generalizaría: Fallaría al predecir nuevos casos que no coincidan exactamente con esas condiciones tan específicas. Sería incapaz de adaptarse a la variabilidad normal de la operación (distintos operadores, fluctuaciones en temperatura o vibración).

**Evaluando el Diagnóstico: La Matriz de Confusión y el F1-Score**

1. Al visualizar la "importancia de las características" de tu árbol, descubres que el "proveedor de materia prima" es la variable más importante. ¿Qué acción inmediata podrías tomar en la planta con esta información?
**RTA** Activa un control de entrada reforzado por proveedor. Segrega y bloquea temporalmente los lotes del proveedor dominante en fallas. Aumenta el muestreo/inspección (AQL más estricto o 100% si el riesgo es alto) solo para ese proveedor. Traza por lote (etiquetado) para aislar rápidamente defectos en producción. Dispara un 8D/CAPA con el proveedor y solicita certificados/PPAP y registros de proceso. En paralelo, corre un piloto A/B comparando ese proveedor vs. otro para confirmar el impacto. Objetivo: reducir de inmediato FN (defectos que se escapan) y FP (falsas alarmas por material variable), atacando la causa más influyente.
2. Si tu árbol de decisión está clasificando perfectamente los datos históricos pero falla mucho con los datos de la última semana (sobreajuste), ¿qué parámetro de poda ajustarías primero para que generalice mejor?
**RTA** Empieza por limitar la profundidad con max_depth. Por qué: los árboles muy profundos aprenden reglas ultra específicas del histórico (ruido). Reducir max_depth obliga a reglas más generales que generalizan mejor. Si aún hay sobreajuste, sube min_samples_leaf (hojas con más muestras) y, si hace falta, min_samples_split o ccp_alpha (poda de costo-complejidad) para podar ramas débiles.


