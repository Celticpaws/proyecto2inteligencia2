# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

show_plot = raw_input("Desea mostrar los gráficos o guardarlos? (save*/show): ") == "show"

img_prefix = "images/"

# Seed para inicialización al azar para las matrices en las redes neurales
np.random.seed(3527)

# Función sigmoidal y versión vectorizada
def sigmoid(value,derivative=False):
    value = np.clip( value, -1000, 1000 ) # Evitamos overflow y underflow

    sig = 1.0 / (1+np.exp(-value))

    if derivative:
        return np.multiply(sig, 1-sig)
    else:
        return sig
vsigmoid = np.vectorize(sigmoid)

# Función tanh y versión vectorizada
def tanh(value,derivative=False):
    sig = (2.0 / (1+np.exp(-2*value))) -1

    if derivative:
        return  1.0 - sig**2
    else:
        return sig
vtanh = np.vectorize(sigmoid)

def split(arr, cond):
  return [arr[cond], arr[~cond]]


class NeuralNetwork():
    def __init__(self,layer_list):
        self.matrices  = []
        self.layers    = layer_list

        # Se crea una lista de matrices de acuerdo a la lista de
        # de capas
        for c,r in zip(layer_list[:-1],layer_list[1:]):
            # Deberíamos romper la posible simetría de la matriz que agrega el random
            self.matrices.append(np.matrix(np.random.random((r,c+1))))

    def forward_prop(self,inp,y=None):
        # Implementación de forward propagation, dada una entrada,
        # se le agrega un vector bias, se multiplica por una matriz de pesos
        # se aplica la función sigmoidal y se repite capa a capa
        z = []
        a = []
        for layer in self.matrices:
            # import pdb;pdb.set_trace()
            ones_vect   = np.matrix(np.ones( (inp.shape[0],1) ))
            inp         = np.append(ones_vect,inp,axis=1)
            a           += [inp]
            inp         = (layer*inp.T).T
            z += [inp]
            inp         = vsigmoid(inp)


        # cost = np.sum( -y * np.log(inp).T - (1-y) * np.log(1-inp.T) ) / inp.shape[0]
        if not(y is None):
            cost =  np.sum( np.abs(y - inp)) / y.shape[0] 
            # cost =  np.sum( np.sqrt( np.power(y-inp,2))) / (y.shape[0] * 2) 
        else:
            cost = 0

        return (inp,z,a,cost)

    def predict(self,inp,treshold=0.5):
        # Predicción para un ejemplo o conjunto de ejemplos, se aplica forward
        # propagation y se estudia si la activación fue mayor a un treshold 
        print(inp)
        val = self.forward_prop(inp)[0]
        # import pdb; pdb.set_trace()
        return val >= treshold
        # return val <= val.mean()

    def get_cost(self,X,y):
        # Calculo de costo, útil para analizar convergencia del entrenamiento
        predictions = self.forward_prop(X)[0]

        # J  = np.sum( -y * np.log(predictions).T - (1-y) * np.log(1-predictions.T) )
        J  = np.sum( np.abs(y - predictions)) / y.shape[0] 

        # J  = np.sum( np.sqrt( np.power(y-predictions,2))) / (y.shape[0] * 2) 
        
    
        return J

    def train(self,X,y,iterations=None,alpha=0.1,epsilon=None,cross_val=None):
        # Entrenamiento mediante forward y backpropagations
        # hasta llegar a la convergencia o cumplir con las iteraciones 
        # dadas
        # print(iterations)
        # print(epsilon)
        if (not iterations) and (not epsilon):
            raise Exception('Missign alpha or epsilon')
        elif (epsilon and not iterations):
            iterations = 9999

        m     = len(self.matrices)
        delta = [ 0 for i in self.matrices]
        costs = []
        cross_costs = []

        last_cost = 0
        new_cost  = 500

        # print(iterations)
        for i in range(iterations):
            # Aplicamos forward propagation y calculamos el error de la
            # última capa
            final,activation,pre_act,new_cost = self.forward_prop(X,y)

            # Almacenamos costos para graficar convergencia
            if epsilon and (abs(last_cost - new_cost) < epsilon):
                print("Convergence in {} iterations".format(i))
                break
            last_cost   =  new_cost
            costs      += [new_cost]
            last_error  = [final-y]

            if not (cross_val is None):
                # import pdb; pdb.set_trace()
                cross_costs += [self.get_cost(cross_val[:,:-y.shape[1]],cross_val[:,-y.shape[1]:])]

            # Calculamos los errores de las última capas hacia las primeras
            for i in reversed(range(m-1)):
                sig = vsigmoid(activation[i],derivative=True)
                err = last_error[-1] * np.delete(self.matrices[i+1],0,1) 
                last_error += [ np.multiply(err,sig)  ]
            last_error.reverse()
            
            # Calculamos gradiente
            for i in range(m):
                delta[i] = (pre_act[i].T * last_error[i]).T / y.shape[0]

            # Decrementamos gradiente por la tasa de aprendizaje
            for i in range(m):
                self.matrices[i] -= alpha * delta[i]

        print("last cost: {}".format(new_cost))

        return (range(len(costs)),costs,cross_costs)

    def test(self,X,y,Xorg):
        sum_of = dict(true_positive  = 0
                     ,false_negative = 0
                     ,false_positive = 0
                     ,true_negative  = 0)

        sm_bias = 0.1**(20)

        predictions = self.predict(X)
        logic_y     = y > 0


        # import pdb; pdb.set_trace()

        sum_of['true_positive' ] = float(np.sum( predictions &  logic_y))
        sum_of['false_negative'] = float(np.sum(~predictions &  logic_y))
        sum_of['false_positive'] = float(np.sum( predictions & ~logic_y))
        sum_of['true_negative' ] = float(np.sum(~predictions & ~logic_y))
        # import pdb; pdb.set_trace()
        tot = sum(sum_of.values())

        accuracy  = (sum_of['true_positive']+sum_of['true_negative'])/tot
        precision = (sum_of['true_positive'])/(sum_of['true_positive']+sum_of['false_positive']+sm_bias)
        recall    = sum_of['true_positive'] / (sum_of['true_positive']+sum_of['false_negative']+sm_bias)
        f_score   = 2*precision*recall/(precision+recall+sm_bias)
        print(sum_of)

        print("| metric    | value  |")
        print("|-----------|--------|")
        print("| accuracy  | {}|".format(accuracy))
        print("| precision | {}|".format(precision))
        print("| recall    | {}|".format(recall))
        print("| f-score   | {}|".format(f_score))



    def plot_prediction(self,X,y,Xorg,file="stump.png",title="stump"):
        predictions = self.predict(X)

        ######Scatter PLot
        Xorg=np.append(Xorg,predictions, axis=1)
        Xorg=Xorg[np.argsort(Xorg.A[:, 0])]
        Xorg=np.asarray(Xorg)
        fig = plt.figure() 
        fig.suptitle(title, fontsize=14)
        ax  = fig.add_subplot(111, aspect='equal')
        
        ax.add_artist(plt.Circle((10, 10), 6, color='b', alpha=0.25, fill=False)) #Circle
        ax.add_artist(plt.Rectangle((0, 0), 20, 20, color='r', alpha=0.25, fill=False)) #Square

        graph_points(Xorg,1) #Positives
        graph_points(Xorg,0) #Negatives
        plt.show() if show_plot else plt.savefig(file)
        plt.close()

    def plot_convergence(self,X,y,iterations=None,alpha=0.1,epsilon=None,cross_val=None,file="stump.png",title="stump"):
        x,y,test_costs = self.train(X,y,iterations,alpha,epsilon,cross_val)

        fig = plt.figure()
        fig.suptitle(title, fontsize=14)

        ax = fig.add_subplot(111)
        ax.set_xlabel('iteraciones')
        ax.set_ylabel('error')

        ax.plot(x,y,label="Conjunto de entrenamiento")
        ax.plot(x,test_costs,label="Conjunto de prueba")
        ax.legend()

        plt.show() if show_plot else plt.savefig(file)
        plt.close()
        

    def __str__(self):
        # Representación de la red neural como string
        res = "Neural network with {} layer of length {}:\n".format(len(self.layers),self.layers)
        res += "\n\n".join([str(i) for i in self.matrices])
        return res
            

def divide_data(data,perc,y_cols=1):
    from math import ceil
    data_size  = data.shape[0]
    train_size = int(ceil(data_size * perc))

    return dict(x_train = data[:train_size,:-y_cols],
                y_train = data[:train_size,-y_cols:],
                cross_t = data[train_size:,:])
            
def graph_points(data,b):
    N   = data.shape[0]
    aux = data[data[:,2]==b]
    if len(aux) > 0:
        x = aux[:,0]
        y = aux[:,1]
        p = aux[:,2]
        area = np.pi * (3 * np.random.rand(N))**2  # 0 to 15 point radii
        newX = np.linspace(0, x.ptp(), N)
        scaled_z = (newX - newX.min()) / newX.ptp()
        if b:
            colors = plt.cm.Blues(scaled_z)
            color = 'b'
        else:
            colors = plt.cm.Oranges(scaled_z)
            color = 'r'

        plt.scatter(x, y, marker='.', edgecolors=colors, s=area, color=color, linewidths=4)
        

def make_title(arq,alpha,iter,size,is_train):
    capas    =  "una capa" if len(arq)==3 else "dos capas"
    data_set = ("training" if is_train else "test") + str(size)
    res = ("Red de {} con {} neuronas. Alpha = {}.\n"+
           "Iter = {}. Dataset = {}").format(capas,arq[1],alpha,iter,data_set)
    return res

def make_filename(arq,alpha,iter,size,is_train,type):
    dataset = str(size) + ("tr" if is_train else "te")
    iter    = str(iter) + "iter_"
    arq     = "".join(map(str,arq)) + "arq_"
    alpha   = str(alpha).replace(".","d") + "alpha_"
    # print(img_prefix + iter + arq  + alpha + dataset + type +".png")
    return img_prefix + arq + iter + alpha + type + "_" + dataset +".png"