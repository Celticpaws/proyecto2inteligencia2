# -*- coding: utf-8 -*-
import numpy as np

# Seed para inicialización al azar para las matrices en las redes neurales
np.random.seed(42+5)

# np.seterr(all='warn')
# np.seterr( over='ignore' )

# Función sigmoidal y versión vectorizada
def sigmoid(value,derivative=False):
    value = np.clip( value, -500, 500 )

    sig = 1.0 / (1+np.exp(-value))

    if derivative:
        return np.multiply(sig, 1-sig)
    else:
        return sig
vsigmoid = np.vectorize(sigmoid)

# Gradiente para función sigmoidal
# def sigmoid_grad(value):
#     return sigmoid(value) * (1 - sigmoid(value))
# vsigmoid_grad = np.vectorize(sigmoid_grad)

class NeuralNetwork():
    def __init__(self,layer_list):
        self.matrices  = []
        self.layers    = layer_list

        for c,r in zip(layer_list[:-1],layer_list[1:]):
            # Deberíamos romper la posible simetría de la matriz que agrega el random
            self.matrices.append(np.matrix(np.random.random((r,c+1))))

    def forward_prop(self,inp):
        z = []
        a = []
        for layer in self.matrices:
            # Multiplicar, aplicar función logística y repetir en siguiente capa
            ones_vect   = np.matrix(np.ones( (inp.shape[0],1) ))
            inp         = np.append(ones_vect,inp,axis=1)
            a           += [inp]
            inp         = (layer*inp.T).T
            z += [inp]
            inp         = vsigmoid(inp)

        return (inp,z,a)

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

    def back_propagate(self,X,y,iterations=100,alpha=0.1):
        m     = len(self.matrices)
        delta = [ 0 for i in self.matrices]
        costs = []
        
        for i in range(iterations):
            # From last to first layer
            print(self.get_cost(X,y) )
            costs += [self.get_cost(X,y)]
            final,activation,pre_act = self.forward_prop(X)
            last_error  = [final-y]
            # activations = activations[:-1]
            # activations.reverse()
            # import pdb; pdb.set_trace()

            for i in reversed(range(m-1)):
                # print(a)
                # sig = np.matrix(np.multiply(activation[i],(1-activation[i])))
                sig = vsigmoid(pre_act[i],True)
                # sig = np.matrix(np.multiply(activations[i],(1-activations[i])))
                # print(self.matrices[m-i-1].shape)
                # print(last_error[i].shape)
                # print(i)
                # print(m-i-1)
                
                # aux_mat = np.matrix(np.copy(self.matrices[1]))
                # aux_mat = aux_mat
                err = last_error[-1] * np.delete(self.matrices[i+1],0,1) 
                last_error += [ np.multiply(err,sig)  ]

            # print("UPDATE")
            last_error.reverse()
            
            # import pdb; pdb.set_trace()
            # self.matrices[-1] -= sum(np.multiply(last_error[-1],self.matrices[-1].T)).T * alpha

            for i in range(m):
                delta[i] = (pre_act[i].T * last_error[i]).T / y.shape[0]

            for i in range(m):
                # import pdb; pdb.set_trace()
                self.matrices[i] -= alpha * delta[i]

        import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()

        plt.plot(range(iterations), costs)
        plt.show()
            
                

    def __str__(self):
        res = "Neural network with {} layer of length {}:\n".format(len(self.layers),self.layers)
        res += "\n\n".join([str(i) for i in self.matrices])
        return res
            

            

net = NeuralNetwork([2,3,1])
# print(net)
# print(net.forward_prop(np.matrix([[2,4],[1,3]])))
data_500 = np.matrix(np.loadtxt('datos_P2_EM2017_N500.txt'),dtype=np.float128)



def normalizacion(matrix,mean=None,std=None,columns=None):
    if mean is None:
        mean = matrix.mean(0)
        std  = matrix.std(0)
    
    if not columns:
        columns = range(0,matrix.shape[1])
    for i in columns:
        if std[i] != 0:
            matrix[:,i] = (matrix[:,i] - mean[i]) / std[i]

    return mean,std

# import pdb; pdb.set_trace()

# data_500 = data_500 - data_500.mean(0) / data_500.std(0)

X = data_500[:,:-1]
y = data_500[:,-1]

net.get_cost(X,y)
net.back_propagate(X,y)