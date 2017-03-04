# -*- coding: utf-8 -*-
from neural_net import *


try:
    alpha = float(raw_input("Introduzca un training rate (default 0.3): "))
except Exception as e:
    alpha = 0.3
print(alpha)

try:
    iter = int(raw_input("Introduzca número de iteraciones (default 2000): "))
except Exception as e:
    iter = 200
print(iter)


# Todas las arquitecturas con una capa oculta con 2 a 10 neuronas 
# y de dos capas ocultas con 2,5 y 10 neuronas en cada una
neural_arq = [[4,i,1] for i in range(2,11)] + [[4,2,2,1],[4,5,5,1],[4,10,10,2]] 

for sz in [500,1000,2000]:
    for arq in neural_arq:
        net = NeuralNetwork(arq)
        train_s = np.matrix(np.loadtxt('datos_P2_EM2017_N'+str(sz)+'.txt'),dtype=np.float128)
        test_s  = np.matrix(np.loadtxt('prueba_N'+str(sz)+'.txt'),dtype=np.float128)

        X    = train_s[:,:-1]
        Xorg = X
        test_orig = np.matrix(np.copy(test_s))
        size = X.shape[0]

        # Normalizamos datos de entrenamiento y prueba con media y varianza
        # de datos de entrenamiento
        test_s[:,:-1] = (test_s[:,:-1] - X.mean(axis=0)) / X.std(axis=0)
        X             = (X - X.mean(axis=0)) / X.std(axis=0)

        # Aumentamos los datos terminos al cuadrado
        X = np.concatenate((X,np.power(X,2)),axis=1)
        y = train_s[:,-1]
        test_s = np.concatenate( (np.concatenate(
                                                 (test_s[:,:-1],np.power(test_s[:,:-1],2))
                                                 ,axis=1)
                                  ,test_s[:,-1])
                               ,axis=1)

        # Costo inicial, entrenamiento y graficación de convergencia
        net.get_cost(X,y)
        title = make_title(arq,alpha,iter,size,is_train=True)
        fn    = make_filename(arq,alpha,iter,size,True,"conv")
        net.plot_convergence(X,y,iterations=iter,alpha = alpha,cross_val=test_s
                            ,title=title,file=fn)

        print("Prediction for datasets size " + str(y.shape[0]))

        # Cálculo de error y gráfico con predicciones para trainig set
        print("On training set")
        net.test(X,y,Xorg)
        title = make_title(arq,alpha,iter,size,is_train=True)
        fn    = make_filename(arq,alpha,iter,size,True,"scat")
        net.plot_prediction(X,y,Xorg,fn,title)
        
        # Caso análogo para test set
        print("On test set")
        title = make_title(arq,alpha,iter,size,is_train=False)
        fn    = make_filename(arq,alpha,iter,size,False,"scat")
        net.test(test_s[:,:-1],test_s[:,-1],test_orig)
        net.plot_prediction(test_s[:,:-1],test_s[:,-1],test_orig,fn,title)
        print("-----------------------------------------------\n\n\n")