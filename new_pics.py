import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

import matplotlib.pyplot as plt
min_max_scaler = preprocessing.MinMaxScaler()
######################################
def perodic_table(molecule):
    return_list = []
    metal_atom = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    halo_atom = ['F','Cl','Br','I']
    for i in metal_atom:
        if i in molecule[-31:-26]:
            return_list.append(metal_atom.index(i))
    for j in halo_atom:
        if j in molecule[-28:-25]:
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
            return_list.append(os.getcwd()+"/"+i+"/"+"vector_vs_energy_shifted")
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
def get_certain_testset(directory,molecule):
    return_list = []
    dire = []
    for x in os.listdir(directory):
        if x[:13] == 'MX2_H_LDA_mag':
            dire.append(x)
    for a in dire:
        os.chdir(directory+"/"+a)
        for i in os.listdir():
            if i == molecule:
                return_list.append(os.getcwd()+"/"+i+"/"+"vector_vs_energy_shifted")
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

def get_molecule_dataset(molecule):
    path_list = get_certain_testset(os.getcwd(),molecule)
    num = len(path_list)
    return_list = [None] * 4 *num
    for i in range(0,num):
        the_list= [None]
        the_list[0] = path_list[i]
        a,b,c = get_coord(the_list)
        return_list[4*i] = min_max_scaler.transform(a)
        return_list[4*i+1] = b
        return_list[4*i+2] = c
        return_list[4*i+3] = a
    return return_list

#####################################
#parameter_certain_testset,y_certain_testset,bond_certain_testset = get_coord(get_certain_testset(os.getcwd(),0,'Sc'))
#scaled_parameter_certain_testset=min_max_scaler.fit_transform(parameter_certain_testset)

######################################
def regress(unit,alpha,X_train,y_train):

    score_list = []
    predict_y_list = [None] * 4
    error_list = []
    #solver_list = ['lbfgs','sgd','adam']
    product_model = MLPRegressor(hidden_layer_sizes=(unit,unit,unit,),
                                 activation='relu',
                                 solver='lbfgs',
                                 learning_rate='constant',
                                 max_iter=2000,
                                 learning_rate_init=0.01,
                                 alpha=alpha)
    product_model.fit(X_train, y_train)
#    for i in range(0,4):
#        predict_y_list[i] = product_model.predict(certain_test_list[4*i]) 
  #  predict_y_train =product_model.predict(X_train)
  #  predict_y_test =product_model.predict(scaled_parameter_certain_testset)

  #  error_train = sum(abs(predict_y_train-y_train))/len(y_train)
  #  error_test = sum(abs(predict_y_test-y_test))/len(y_test)

    return product_model
####################################
def draw_pic(a,molecule):
    mole_dataset = get_molecule_dataset(molecule)
    num = int(len(mole_dataset)/4)
    color_list = ['yellow','red','blue','green','purple','orange']
   
    fig = plt.figure(molecule,figsize=[8,6])
    fig.suptitle(molecule)
    ax1 = fig.add_subplot(111)
    for i in range(0,num):
        ax1.scatter(mole_dataset[4*i+3][:,0],mole_dataset[4*i+1], s=10, c=color_list[i], marker="s", label='real')
        ax1.scatter(mole_dataset[4*i+3][:,0],a.predict(mole_dataset[4*i]), s=10, c=color_list[i], marker="o", label='real')

    plt.show()
#####################################
def draw_all_pics(a):
    metal_atom = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    halo_atom = ['F2','Cl2','Br2','I2']
    all_mole_list = []
    for i in metal_atom:
        for j in halo_atom:
            all_mole_list.append(i+j)
    for molecule in all_mole_list:
        draw_pic(a,molecule)
####################################



#fig = plt.figure('ScF2')
#ax1 = fig.add_subplot(111)
#ax1.scatter((get_all_certain_testset()[3])[:,0],get_all_certain_testset()[1], s=10, c='b', marker="s", label='real')
#ax1.scatter((get_all_certain_testset()[3])[:,0],regress(100,0.0001,get_all_certain_testset())[0], s=10, c='r', marker="o", label='real')
#
#fig2 = plt.figure('ScCl2')
#ax1_2 = fig2.add_subplot(111)
#ax1_2.scatter((get_all_certain_testset()[7])[:,0],get_all_certain_testset()[5], s=10, c='b', marker="s", label='real')
#ax1_2.scatter((get_all_certain_testset()[7])[:,0],regress(100,0.0001,get_all_certain_testset())[1], s=10, c='r', marker="o", label='real')
#
##ax1.scatter(X_test[100:150,1],regress(100,0.0008)[100:150], s=10, c='r', marker="o", label='NN Prediction')
##ax1.scatter(X_test[:50,1],regress(100,0.0003)[:50], s=10, c='r', marker="o", label='NN Prediction')
#plt.show()
#
#print (X_test[:,1].tolist())
#print (regress())
#unit_list = [20,40,80,100]
#alpha_list = [0.0001,0.0003]
#for alpha in alpha_list:
#    for unit in unit_list:
#        train, test = regress(unit,alpha)
#        with open("NN_result_train_test", "a") as p:
#            p.write('the error of ' + str(unit) + ' unit in each of 1 layer with alpha = '+ str(alpha) +' is '+ 'train: ' + str(train) + ' test: ' + str(test)+"\n")
#

#regress(100,0.0001,get_all_certain_testset())
all_vec_eng = get_all_file(os.getcwd())
parameter, y, bond = get_coord(all_vec_eng)
scaled_para = min_max_scaler.fit_transform(parameter)
X_train, X_test, y_train, y_test = train_test_split(scaled_para, y, random_state = 3, test_size=0.4)
model = regress(250,0.0001,X_train,y_train)
draw_all_pics(model)
#mole_dataset = get_molecule_dataset('ScCl2')
#print (mole_dataset[3][:,0])
#print (mole_dataset[1])
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(mole_dataset[3][:,0],mole_dataset[1], s=10, c='r', marker="s", label='real')
#plt.show()



#draw_pic(model,'MnCl2')
