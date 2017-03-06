# -*- coding: utf-8 -*-
from neural_net import *

try:
    alpha = float(raw_input("Introduzca un training rate (default 0.3): "))
except Exception as e:
    alpha = 0.3
print(alpha)

try:
    iter = int(raw_input("Introduzca número de iteraciones (default 20): "))
except Exception as e:
    iter = 20
print(iter)

def to_binary(row):
    if row[-1] == 'Iris-setosa':
        return np.append(row[:-1],"1.0")
    else:
        return np.append(row[:-1],"0.0")
v_to_binary = np.vectorize(to_binary)

def to_ternary(row):
    if row[-1] == 'Iris-setosa':
        return np.append(row[:-1],[1,0,0],axis=0)
    elif row[-1] == 'Iris-versicolor':
        return np.append(row[:-1],[0,1,0],axis=0)
    else:
        return np.append(row[:-1],[0,0,1],axis=0)
v_to_ternary = np.vectorize(to_ternary)


data =  np.loadtxt('bezdekIris.data',dtype=str,delimiter=",")
np.random.shuffle(data)
np.random.shuffle(data)

'''
bin_dat = np.matrix(map(to_binary,data),dtype=np.float128)

neural_arq = [[4,i,1] for i in range(4,11)] + [[4,5,5,1]]

for train_size in [0.5,0.6,0.7,0.8,0.9]:
    dat = divide_data(bin_dat,train_size)
    for arq in neural_arq:
        net = NeuralNetwork(arq)

        title = make_title(arq,alpha,iter,train_size,is_train=True)
        fn    = make_filename(arq,alpha,iter,train_size,True,"conv")
        net.plot_convergence(dat["x_train"],dat["y_train"],iterations=iter
                            ,alpha = alpha,cross_val=dat["cross_t"],title=title,file=fn)
'''

neural_arq = [[4,i,3] for i in range(4,11)] + [[4,5,5,3]]
tern_dat = np.matrix(map(to_ternary,data),dtype=np.float128)

print(tern_dat)

for train_size in [0.5,0.6,0.7,0.8,0.9]:
    dat = divide_data(tern_dat,train_size,3)
    for arq in neural_arq:
        net = NeuralNetwork(arq)

        title = make_title(arq,alpha,iter,train_size,is_train=True)
        fn    = make_filename(arq,alpha,iter,train_size,True,"conv")
        net.plot_convergence(dat["x_train"],dat["y_train"],iterations=iter
                            ,alpha = alpha,cross_val=dat["cross_t"],title=title,file=fn)
        break
        # title = make_title(arq,alpha,iter,train_size,is_train=True)
        # fn    = make_filename(arq,alpha,iter,train_size,True,"conv")
        # net.plot_convergence(dat["x_train"],dat["y_train"],iterations=iter
        #                     ,alpha = alpha,cross_val=dat["cross_t"],title=title,file=fn)
