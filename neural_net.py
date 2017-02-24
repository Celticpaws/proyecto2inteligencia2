# -*- coding: utf-8 -*-
import numpy as np

# Seed para inicialización al azar para las matrices en las redes neurales
np.random.seed(42+5)

# Función sigmoidal y versión vectorizada
def sigmoid(value):
    return 1 / (1+np.exp(-value))
vsigmoid = np.vectorize(sigmoid)

# Gradiente para función sigmoidal
def sigmoid_grad(value):
    return sigmoid(value) * (1 - sigmoid(value))

class NeuralNetwork():
    def __init__(self,layer_list):
        self.matrices  = []
        self.layers    = layer_list

        for r,c in zip(layer_list[:-1],layer_list[1:]):
            # Deberíamos romper la posible simetría de la matriz que agrega el random
            self.matrices.append(np.matrix(np.random.random((r+1,c))))

    def forward_prop(self,inp):
        activations = []
        for layer in self.matrices:
            # Multiplicar, aplicar función logística y repetir en siguiente capa
            ones_vect   = np.matrix(np.ones( (inp.shape[0],1) ))
            inp         = np.append(ones_vect,inp,axis=1)
            # print(inp.shape)
            # print(layer.shape)
            # print("---------")
            inp         = vsigmoid(inp*layer)
            # print(type(inp))
            # print("appending")
            activations += [inp]
            # print(activations)

        return (inp,activations)

    def predict(self,inp,treshold=0.5):
        val = self.forward_prop(inp)[0]
        return val <= treshold

    def get_cost(self,X,y):
        predictions = self.forward_prop(X)[0]
        # print(y.shape)
        # print(predictions.shape)

        J  = np.sum( -y * np.log(predictions).T - (1-y) * np.log(1-predictions.T) )
        J /= y.shape[0]
    
        # print(J)
        return J
        # J /= m;

    def back_propagate(self,X,y,iterations=200,alpha=0.5):
        m           = len(self.matrices)
        delta = [ np.zeros(i.shape) for i in self.matrices]
        
        for i in range(iterations):
            # From last to first layer
            print(self.get_cost(X,y) )
            activations = self.forward_prop(X)[1]
            last_error  = [activations[0]-y]
            for i,a in enumerate(activations[1:]):
                # print(a)
                sig = np.multiply(a,(1-a))
                # print(self.matrices[m-i-1].shape)
                # print(last_error[i].shape)
                # import pdb; pdb.set_trace()
                last_error += [np.multiply(last_error[i]*self.matrices[m-i-2] , sig ) ]

             # print("UPDATE")
            for i in range(len(last_error)-1):
                # print(self.matrices[i].shape)
                # print(last_error[i].shape)
                # print(activations[i].shape)
                # last_error[i] 
                delta[i] = delta[i] +  activations[i].T * last_error[i]

            for i in range(len(last_error)-1):
                self.matrices[i] = self.matrices[i] -  delta[i] * alpha
            
                

    def __str__(self):
        res = "Neural network with {} layer of length {}:\n".format(len(self.layers),self.layers)
        res += "\n\n".join([str(i) for i in self.matrices])
        return res
            

            

net = NeuralNetwork([2,3,1])
# print(net)
# print(net.forward_prop(np.matrix([[2,4],[1,3]])))
data_500 = np.matrix(np.loadtxt('datos_P2_EM2017_N500.txt'))

X = data_500[:,:-1]
y = data_500[:,-1]

net.get_cost(X,y)
net.back_propagate(X,y)