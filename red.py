# -*- coding: utf-8 -*-
# Proyecto 2 - Red Neural
# Universidad Simón Bolívar, 2017.
# Authors: Carlos Farinha   09-10270
#          Javier López     11-10552
#          Nabil J. Marquez 11-10683
# Last Revision: 23/02/17

from math import exp
from random import seed
from random import random
 
# Inicializar la red
def inicializar_red(n_entradas, n_ocultas, n_salidas):
	red = list()
	capa_oculta = [{'pesos':[random() for i in range(n_entradas + 1)]} for i in range(n_ocultas)]
	red.append(capa_oculta)
	capa_salida = [{'pesos':[random() for i in range(n_ocultas + 1)]} for i in range(n_salidas)]
	red.append(capa_salida)
	return red
 
# Calcula el valor de la neuronaa para una entrada
def activar(pesos, entradas):
	activacion = pesos[-1]
	for i in range(len(pesos)-1):
		activacion += pesos[i] * entradas[i]
	return activacion
 
# Transferencia de activacion por sigmoidal
def transferencia(activacion):
	return 1.0 / (1.0 + exp(-activacion))
 
# Propagacion de salida
def propagacion(red, fila):
	entradas = fila
	for capa in red:
		nuevas_entradas = []
		for neurona in capa:
			activacion = activar(neurona['pesos'], entradas)
			neurona['salida'] = transferencia(activacion)
			nuevas_entradas.append(neurona['salida'])
		entradas = nuevas_entradas
	return entradas
 
# Calculate the derivada of an neurona salida
def transferencia_derivada(salida):
	return salida * (1.0 - salida)
 
# Error de propagacion en las neuronas
def error_de_propagacion(red, esperados):
	for i in reversed(range(len(red))):
		capa = red[i]
		errores = list()
		if i != len(red)-1:
			for j in range(len(capa)):
				error = 0.0
				for neurona in red[i + 1]:
					error += (neurona['pesos'][j] * neurona['delta'])
				errores.append(error)
		else:
			for j in range(len(capa)):
				neurona = capa[j]
				errores.append(esperados[j] - neurona['salida'])
		for j in range(len(capa)):
			neurona = capa[j]
			neurona['delta'] = errores[j] * transferencia_derivada(neurona['salida'])
 
# Actualizamos los pesos de las neuronas con el error
def actualizar_pesos(red, fila, alpha):
	for i in range(len(red)):
		entradas = fila[:-1]
		if i != 0:
			entradas = [neurona['salida'] for neurona in red[i - 1]]
		for neurona in red[i]:
			for j in range(len(entradas)):
				neurona['pesos'][j] += alpha * neurona['delta'] * entradas[j]
			neurona['pesos'][-1] += alpha * neurona['delta']
 
# Entrenamos la red para un conjunto en particular 
def entrenar_red(red, entrenamiento, alpha, n_epoch, n_salidas):

	for epoch in range(n_epoch):
		sum_error = 0
		for fila in entrenamiento:
			salidas = propagacion(red, fila)
			esperados = [0 for i in range(n_salidas)]
			esperados[int(fila[-1])] = 1
			sum_error += sum([(esperados[i]-salidas[i])**2 for i in range(len(esperados))])
			error_de_propagacion(red, esperados)
			actualizar_pesos(red, fila, alpha)
 
# Lectura del archivo para el entrenamiento de la red
dataset = []
file = open("datos_P2_EM2017_N2000.txt","r")
lines = file.read().splitlines()
for x in lines:
	dataset.append([float(elem) for elem in x.split(' ')])
	
n_entradas = len(dataset[0]) - 1
n_salidas = len(set([fila[-1] for fila in dataset]))
red = inicializar_red(n_entradas, 2, n_salidas)
entrenar_red(red, dataset, 0.5, 100, n_salidas)
for capa in red:
	print(capa)
