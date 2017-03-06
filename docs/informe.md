
# Introducción

  Las redes neuronales (también conocidos como sistemas conexionistas) son un enfoque computacional, que se basa en una gran colección de unidades neurales (también conocido como neuronas artificiales), para modelar libremente la forma en que un cerebro biológico resuelve problemas con grandes grupos de neuronas biológicas conectados por los axones. Cada unidad neuronal está conectada con otras, y los enlaces se pueden aplicar en su efecto sobre el estado de activación de unidades neuronales conectadas. Cada individuo de la unidad neuronal puede tener una función de suma, que combina conjuntamente los valores de todas las entradas. 
  
  Estos sistemas son auto-aprendizaje y formación, en lugar de programar de forma explícita, sobresalen en las zonas donde la solución o función de detección es difícil de expresar en un programa de ordenador tradicional.
  
  Para este proyecto se quiere realizar la implementacion de una red neuronal multicapa capaz de identificar a que conjunto pertenece un punto dentro de un plano de dimensiones definidas de 20 x 20, dentro o fuera de un circulo de radio 6 ubicado en el centro de lienzo. Para ello se evalua la eficiencia de la red tomando en cuenta 6 conjuntos de diferentes para probar el aprendizaje de la red en diferentes casos.
  
  La idea es poder proyectar un algoritmo que no solo identifique los patrones dentro del lienzo si no que su capacidad de respuesta sea optima en comparacion a sus homologos, a fin de tener una medida de eficiencia del mismo.

# Implementación

Haciendo uso de nuestro generador de ejemplos, creamos los conjuntos de prueba 
de tamaño 500, 1000 y 2000 con los siguientes comandos:

```
python generador.py 42 500
python generador.py 5872 1000
python generador.py 7219735 2000
```

Usando como semillas para la generación de números aleatorios *42*, *5872* y 
*7219735* respectivamente.

# Ejercicio 1

Para este ejercicio implementamos, en *neural_net.py* la clase NeuralNetwork,
que dada una lista con las dimensiones de cada capa de neuronas 
(incluyendo entrada y salida), internamente esto generara una lista de 
matrices representando las conexiones entre cada una de las capas. Los pesos
son inicializados en valores al azar dados por una semillas fija.

La clase NeuralNet implementa los metodos *forward_prop* que al recibir una
entrada y el valor esperado de salida, realiza las multiplicaciones de matrices
y activaciones necesarias para obtener un valor de salida, adicionalmente a esto
calcula el costo asociado a esta propagación.

El método *predict* realiza una propagación y dado un *treshold* entrega la 
predicción más cercana a los valores de salida que entregó la red.

El método *train* dada una entrada, salidas esperadas, un tasa de aprendizaje $\alpha$, y
un conjunto de validación, entrena la red almacenando todos los costos para los 
conjuntos de entrenamiento y validación.

*test* realiza el conteo de clasificaciones correctas y las métricas asociadas
como precision, accuracy, sensibilidad, entre otros.

*plot_prediction* dato un conjunto de puntos realiza un *scatter plot*, coloreando
de rojos los puntos clasificados como afuera del círculo y como azules los internos.
Adicionalmente se muestra el área del círculo.

*plot_convergence* entrena la red y grafica las funciones de costo vs iteraciones
para el conjunto de entrenamiento y de validación dados,

# Ejercicio 2

# Ejercicio 3

# Resultados y discusión

# Conclusiones

