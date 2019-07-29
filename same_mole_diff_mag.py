import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

a=sys.argv[1]
######################################
def get_all_file(directory,molecule):
    return_list = []
    dire = []
    for x in os.listdir(directory):
        if x[:13] == 'MX2_H_LDA_mag':
            dire.append(x)
    for a in dire:
        os.chdir(directory+"/"+a)
        for y in os.listdir():
            if y == molecule:
                return_list.append(os.getcwd()+"/"+molecule+"/"+"vector_vs_energy")
   #     num_list = []
   #     for y in os.listdir():
   #         try:
   #             x = float(y)
   #             num_list.append(x)
   #         except ValueError:
   #             pass

   #     for z in num_list:
        os.chdir("../")
    return return_list
######################################
all_vec_eng = get_all_file(os.getcwd(),a)

#######################################
def str_list2float_list(str_l):
    return_list = []
    for x in str_l:
        return_list.append(float(x))
    return return_list
#######################################
def get_coord(the_list):
   
    para_list = []
    energy_list = []
    bond_list = []
    for c in the_list:
        with open(c, "r") as p:
            for i, line in enumerate(p):
                coord_1 = []
                if "#" not in line:
                    #d = str_list2float_list(line.split())[0:3]
                    length = float(line.split()[0])
                    bond = float(line.split()[3])
                    mag = float(line.split()[2])
                    f = float(line.split()[1]) 
                    para_list.append(length)
                    para_list.append(bond)
                    para_list.append(mag)
                    bond_list.append(bond)
                    energy_list.append(f)
    a = len(energy_list)
    parameter = np.array(para_list).reshape(a,3)
    energy = np.array(energy_list)
    bond = np.array(bond_list)
    return parameter, energy,bond
######################################
parameter, y, bond = get_coord(all_vec_eng)

X_train, X_test, y_train, y_test = train_test_split(parameter, y, random_state = 0)
######################################
def regress():

    score_list = []
    predict_list = []
    #solver_list = ['lbfgs','sgd','adam']
    unit_list = [5,10,15,20,25,30,35,40,45,50]
    unit = [15,30,45,60,75,100]
#    for a in unit:
    product_model = MLPRegressor(hidden_layer_sizes=(100,),
                                 activation='relu',
                                 solver='lbfgs',
                                 learning_rate='constant',
                                 max_iter=2000,
                                 learning_rate_init=0.01,
                                 alpha=0.0003)
    product_model.fit(X_train, y_train)
    score = product_model.score(X_test, y_test)
    score_list.append(score)
    predict_y =product_model.predict(X_test)
        #predict =product_model.predict(j).tolist()
        #predict_list.append(predict)
#tt = [3.22236,2.55410784594]
#j= np.array(tt).reshape(1,2)
    return predict_y
####################################
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_test[:,1], y_test, s=10, c='b', marker="s", label='real')
ax1.scatter(X_test[:,1],regress(), s=10, c='r', marker="o", label='NN Prediction')
plt.show()
#print (X_test[:,1].tolist())
#print (regress())
