# -*- coding: utf-8 -*-
import numpy as np

class NeuralNetwork():
    def __init__(self,layer_list,threshold=0.5):
        self.matrices  = []
        self.layers    = layer_list
        self.threshold = threshold

        for r,c in zip(layer_list[:-1],layer_list[1:]):
            # Deberíamos romper la posible simetría de la matriz que agrega el random
            self.matrices.append(np.random.random((r+1,c)))

    def forward_prop(self,inp):
        inp = np.append([1],inp)

        for layer in self.layers:
            # Multiplicar, aplicar función logística y repetir en siguiente capa
            inp = inp * layer
            # FALTA APLICAR SIGMOIDAL

        return inp

    def get_cost(self):
        pass

    def back_propagate(self):
        pass

    def __str__(self):
        res = "Neural network with {} layer of length {}:\n".format(len(self.layers),self.layers)
        res += "\n\n".join([str(i) for i in self.matrices])
        return res
            

            

net = NeuralNetwork([2,3,1])
print(net)
print(net.forward_prop(np.array([2,4])))