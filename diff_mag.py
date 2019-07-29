import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

######################################
def perodic_table(molecule):
    return_list = []
    metal_atom = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    halo_atom = ['F','Cl','Br','I']
    for i in metal_atom:
        if i in molecule[-23:-18]:
            return_list.append(metal_atom.index(i))
    for j in halo_atom:
        if j in molecule[-20:-17]:
            return_list.append(halo_atom.index(j))
    return return_list
######################################
def get_all_file(directory):
    return_list = []
    dire = []
    for x in os.listdir(directory):
        if x[:13] == 'MX2_H_LDA_mag':
            dire.append(x)
    for a in dire:
        os.chdir(directory+"/"+a)
        for i in os.listdir():
            return_list.append(os.getcwd()+"/"+i+"/"+"vector_vs_energy")
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
all_vec_eng = get_all_file(os.getcwd())
######################################
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
        try:
            with open(c, "r") as p:
                for i, line in enumerate(p):
                    coord_1 = []
                    if "#" not in line:
                        length = float(line.split()[0])
                        bond = float(line.split()[3])
                        mag = float(line.split()[2])
                        f = float(line.split()[1]) 
                        para_list.append(length)
                        para_list.append(bond)
                        para_list.append(mag)
                        if len(perodic_table(c)) == 2:
                            para_list += perodic_table(c)
                        bond_list.append(bond)
                        energy_list.append(f)
        except PermissionError:
            pass
    a = len(energy_list)
    parameter = np.array(para_list).reshape(a,5)
    energy = np.array(energy_list)
    bond = np.array(bond_list)
    return parameter, energy,bond
######################################
parameter, y, bond = get_coord(all_vec_eng)


X_train, X_test, y_train, y_test = train_test_split(parameter, y, random_state = 3)
######################################
def regress(num1,num2):

    score_list = []
    predict_list = []
    #solver_list = ['lbfgs','sgd','adam']
    unit_list = [5,10,15,20,25,30,35,40,45,50]
    unit = [15,30,45,60,75,100]
#    for a in unit:
    product_model = MLPRegressor(hidden_layer_sizes=(num1,),
                                 activation='relu',
                                 solver='lbfgs',
                                 learning_rate='constant',
                                 max_iter=2000,
                                 learning_rate_init=0.01,
                                 alpha=num2)
    product_model.fit(X_train, y_train)
    score = product_model.score(X_test, y_test)
    score_list.append(score)
    predict_y =product_model.predict(X_test)
    return predict_y
####################################
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X_test[:50,1], y_test[:50], s=10, c='b', marker="s", label='real')
#ax1.scatter(X_test[100:150,1],regress(100,0.0008)[100:150], s=10, c='r', marker="o", label='NN Prediction')
ax1.scatter(X_test[:50,1],regress(100,0.0003)[:50], s=10, c='r', marker="o", label='NN Prediction')
plt.show()



#print (X_test[:,1].tolist())
#print (regress())
