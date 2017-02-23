# -*- coding: utf-8 -*-
# Proyecto 2 - Generador de puntos de prueba
# Universidad Simón Bolívar, 2017.
# Authors: Carlos Farinha   09-10270
#          Javier López     11-10552
#          Nabil J. Marquez 11-10683
# Last Revision: 23/02/17

from math import exp
from random import seed
from random import random

archivo = open('pruebas.txt', 'w+')

for i in range(100):
	x = random()*10
	y = random()*10
	archivo.write(str(x)+" "+str(y)+" \n")

archivo.close()