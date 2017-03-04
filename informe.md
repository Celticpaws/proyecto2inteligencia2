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
