# -*- coding: utf-8 -*-
import numpy as np

# Seed para inicialización al azar para las matrices en las redes neurales
np.random.seed(42+5)

# Función sigmoidal y versión vectorizada
def sigmoid(value,derivative=False):
    value = np.clip( value, -500, 500 ) # Evitamos overflow y underflow

    sig = 1.0 / (1+np.exp(-value))

    if derivative:
        return np.multiply(sig, 1-sig)
    else:
        return sig
vsigmoid = np.vectorize(sigmoid)

class NeuralNetwork():
    def __init__(self,layer_list):
        self.matrices  = []
        self.layers    = layer_list

        # Se crea una lista de matrices de acuerdo a la lista de
        # de capas
        for c,r in zip(layer_list[:-1],layer_list[1:]):
            # Deberíamos romper la posible simetría de la matriz que agrega el random
            self.matrices.append(np.matrix(np.random.random((r,c+1))))

    def forward_prop(self,inp):
        # Implementación de forward propagation, dada una entrada,
        # se le agrega un vector bias, se multiplica por una matriz de pesos
        # se aplica la función sigmoidal y se repite capa a capa
        z = []
        a = []
        for layer in self.matrices:
            ones_vect   = np.matrix(np.ones( (inp.shape[0],1) ))
            inp         = np.append(ones_vect,inp,axis=1)
            a           += [inp]
            inp         = (layer*inp.T).T
            z += [inp]
            inp         = vsigmoid(inp)


        cost = np.sum( -y * np.log(inp).T - (1-y) * np.log(1-inp.T) ) / inp.shape[0]

        return (inp,z,a,cost)

    def predict(self,inp,treshold=0.5):
        # Predicción para un ejemplo o conjunto de ejemplos, se aplica forward
        # propagation y se estudia si la activación fue mayor a un treshold 
        val = self.forward_prop(inp)[0]
        return val >= treshold

    def get_cost(self,X,y):
        # Calculo de costo, útil para analizar convergencia del entrenamiento
        predictions = self.forward_prop(X)[0]

        J  = np.sum( -y * np.log(predictions).T - (1-y) * np.log(1-predictions.T) )
        J /= y.shape[0]
    
        return J

    def train(self,X,y,iterations=None,alpha=0.1,epsilon=None):
        # Entrenamiento mediante forward y backpropagations
        # hasta llegar a la convergencia o cumplir con las iteraciones 
        # dadas

        if (not iterations) and (not epsilon):
            raise Exception('Missign alpha or epsilon')
        elif (epsilon and not iterations):
            iterations = 9999

        m     = len(self.matrices)
        delta = [ 0 for i in self.matrices]
        costs = []

        last_cost = 0
        new_cost  = 500

        
        for i in range(iterations):
            # Aplicamos forward propagation y calculamos el error de la
            # última capa
            final,activation,pre_act,new_cost = self.forward_prop(X)

            # Almacenamos costos para graficar convergencia
            if epsilon and (abs(last_cost - new_cost) < epsilon):
                print("Convergence in {} iterations".format(i))
                break
            last_cost =  new_cost
            costs    += [new_cost]

            last_error  = [final-y]


            # Calculamos los errores de las última capas hacia las primeras
            for i in reversed(range(m-1)):
                sig = vsigmoid(activation[i],True)
                err = last_error[-1] * np.delete(self.matrices[i+1],0,1) 
                last_error += [ np.multiply(err,sig)  ]
            last_error.reverse()
            
            # Calculamos gradiente
            for i in range(m):
                delta[i] = (pre_act[i].T * last_error[i]).T / y.shape[0]

            # Decrementamos gradiente por la tasa de aprendizaje
            for i in range(m):
                self.matrices[i] -= alpha * delta[i]

        return (range(len(costs)), costs)
            
    def plot_convergence(self,X,y,iterations=100,alpha=0.1,epsilon=None):
        x,y = self.train(X,y,iterations,alpha,epsilon)
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.show()

    def __str__(self):
        # Representación de la red neural como string
        res = "Neural network with {} layer of length {}:\n".format(len(self.layers),self.layers)
        res += "\n\n".join([str(i) for i in self.matrices])
        return res
            

            

net = NeuralNetwork([2,5,5,1])
data_500 = np.matrix(np.loadtxt('datos_P2_EM2017_N500.txt'),dtype=np.float128)


X = data_500[:,:-1]
y = data_500[:,-1]

net.get_cost(X,y)
net.plot_convergence(X,y,epsilon=0.001)