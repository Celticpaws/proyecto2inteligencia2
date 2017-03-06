
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
para el conjunto de entrenamiento y de validación dados.

# Resultados y discusion



## Ejercicio 1
Se realizo la implementación de backpropagation en una red multicapa feedforward, contenida en el archivo "neural_net.py". Cada red fue entrenada de tal forma que se logro que aprendieran la clasificaci´on correcta de los puntos con los 6 conjuntos de entrenamiento como se puede apreciar en las siguientes imagenes.


### Caso Training

\begin{center}
\includegraphics[height=6cm]{images/e2/451arq_2000iter_0d3alpha_scat_500tr.png} 
\includegraphics[height=6cm]{images/e2/461arq_2000iter_0d3alpha_scat_1000tr.png} 
\includegraphics[height=6cm]{images/e2/471arq_2000iter_0d3alpha_scat_2000tr.png} 
\end{center}

### Caso Test

\begin{center}
\includegraphics[height=6cm]{images/e2/421arq_2000iter_0d3alpha_scat_500te.png} 
\includegraphics[height=6cm]{images/e2/431arq_2000iter_0d3alpha_scat_1000te.png} 
\includegraphics[height=6cm]{images/e2/441arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}


## Ejercicio 2

### Errores de Configuración
Los errores de configuracion fueron las siguientes:


Los mejores conjuntos de entrenamiento encontrados variando en correlacion al numero de neuronas en la red fue el siguiente:

* Para 2 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/421arq_2000iter_0d3alpha_conv_2000tr.png} 
\includegraphics[height=4cm]{images/e2/421arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}

* Para 3 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/431arq_2000iter_0d3alpha_conv_1000tr.png} 
\includegraphics[height=4cm]{images/e2/431arq_2000iter_0d3alpha_scat_1000te.png} 
\end{center}

* Para 4 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/441arq_2000iter_0d3alpha_conv_1000tr.png} 
\includegraphics[height=4cm]{images/e2/441arq_2000iter_0d3alpha_scat_1000te.png} 
\end{center}

* Para 5 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/451arq_2000iter_0d3alpha_conv_1000tr.png} 
\includegraphics[height=4cm]{images/e2/451arq_2000iter_0d3alpha_scat_1000te.png} 
\end{center}

* Para 6 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/461arq_2000iter_0d3alpha_conv_1000tr.png} 
\includegraphics[height=4cm]{images/e2/461arq_2000iter_0d3alpha_scat_1000te.png} 
\end{center}

* Para 7 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/471arq_2000iter_0d3alpha_conv_2000tr.png} 
\includegraphics[height=4cm]{images/e2/471arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}

* Para 8 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/481arq_2000iter_0d3alpha_conv_2000tr.png} 
\includegraphics[height=4cm]{images/e2/481arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}

* Para 9 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/491arq_2000iter_0d3alpha_conv_2000tr.png} 
\includegraphics[height=4cm]{images/e2/491arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}

* Para 10 neuronas en 1 capa

\begin{center}
\includegraphics[height=4cm]{images/e2/4101arq_2000iter_0d3alpha_conv_2000tr.png} 
\includegraphics[height=4cm]{images/e2/4191arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}

* Para 2 neuronas en 2 capas

\begin{center}
\includegraphics[height=4cm]{images/e2/4221arq_2000iter_0d3alpha_conv_500tr.png} 
\includegraphics[height=4cm]{images/e2/4221arq_2000iter_0d3alpha_scat_500te.png} 
\end{center}

* Para 5 neuronas en 2 capas

\begin{center}
\includegraphics[height=4cm]{images/e2/4551arq_2000iter_0d3alpha_conv_500tr.png} 
\includegraphics[height=4cm]{images/e2/4551arq_2000iter_0d3alpha_scat_500te.png} 
\end{center}

* Para 10 neuronas en 2 capas

\begin{center}
\includegraphics[height=4cm]{images/e2/410101arq_2000iter_0d3alpha_conv_2000tr.png} 
\includegraphics[height=4cm]{images/e2/410101arq_2000iter_0d3alpha_scat_2000te.png} 
\end{center}

## Ejercicio 3
Entrenando la red sobre los datos del conjunto Iris Data Set se obtienen los siguientes resultados de manera general
\begin{center}
\includegraphics[height=5cm]{images/e3/441arq_20iter_0d3alpha_scat_500te.png}
\end{center}


### Clasificador binario

* Para 4 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/441arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/441arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/441arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/441arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/441arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 5 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/451arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/451arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/451arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/451arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/451arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 6 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/461arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/461arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/461arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/461arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/461arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 7 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/471arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/471arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/471arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/471arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/471arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 8 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/481arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/481arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/481arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/481arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/481arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 9 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/491arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/491arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/491arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/491arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/491arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 10 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/4101arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/4101arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/4101arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/4101arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/4101arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 5 neuronas en 2 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/4551arq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/4551arq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/4551arq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/4551arq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/4551arq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

### Clasificador de 3 clases

* Para 4 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/443seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/443seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/443seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/443seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/443seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 5 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/453seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/453seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/453seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/453seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/453seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 6 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/463seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/463seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/463seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/463seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/463seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 7 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/473seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/473seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/473seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/473seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/473seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 8 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/483seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/483seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/483seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/483seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/483seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 9 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/493seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/493seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/493seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/493seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/493seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 10 neuronas en 1 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/4103seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/4103seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/4103seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/4103seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/4103seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

* Para 5 neuronas en 2 capas

\begin{center}
\includegraphics[height=3cm]{images/e3/4553seq_50iter_0d05alpha_conv_0.5tr.png} 
\includegraphics[height=3cm]{images/e3/4553seq_50iter_0d05alpha_conv_0.6tr.png} 
\includegraphics[height=3cm]{images/e3/4553seq_50iter_0d05alpha_conv_0.7tr.png} 
\includegraphics[height=3cm]{images/e3/4553seq_50iter_0d05alpha_conv_0.8tr.png} 
\includegraphics[height=3cm]{images/e3/4553seq_50iter_0d05alpha_conv_0.9tr.png} 
\end{center}

# Conclusiones