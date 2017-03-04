# -*- coding: utf-8 -*-
# Proyecto 2 - Generador de puntos de prueba
# Universidad Simón Bolívar, 2017.
# Authors: Carlos Farinha   09-10270
#          Javier López     11-10552
#          Nabil J. Marquez 11-10683
# Last Revision: 23/02/17

# from math import exp
import sys
from random import seed, random

if (len(sys.argv)!=3):
    print("Error, usar:")
    print("    python generador.py semilla num_ejemplos")
    exit()

seed     = seed(int(sys.argv[1]))
num_tests = int(sys.argv[2])



archivo = open('prueba_N'+str(num_tests)+'.txt', 'w')

for i in range(num_tests):
    x = random()*20
    y = random()*20
    is_in_circ = int((x - 10) ** 2 + (y - 10) ** 2 <= 36)
    archivo.write("{} {} {}\n".format(x,y,is_in_circ))

archivo.close()