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
    metal_row = [4,4,4,4,4,4,4,4,4,4]
    metal_column = [3,4,5,6,7,8,9,10,11,12]
    metal_electron = [21,22,23,24,25,26,27,28,29,30]
    metal_en = [1.36,1.54,1.63,1.66,1.55,1.83,1.88,1.91,1.90,1.65]
    metal_zval = [11,10,11,12,13,8,9,10,0,0]
    metal_ea = [-18,-8,-51,-65.2,0,-15,-64.0,-111.7,-119.2,0]
    metal_ionized = [633.1,658.8,650.9,652.9,717.3,762.5,760.4,737.1,745.5,906.4]
    metal_radius = [162,147,134,128,127,126,125,124,128,134]
    halo_atom = ['O','ZZZZZ','ZZZZZ','Te']
    halo_row = [2,3,4,5]
    halo_column = [16,16,16,16]
    halo_electron = [9,17,35,53]
    halo_en = [3.44,2.58,2.55,2.1]
    halo_zval = [6,6,6,6]
    halo_ea = [-328.2,-348.6,-324.5,-295.2]
    halo_ionized = [1313.9,999.6,941.0,869.3]
    halo_radius = [152,180,190,206]
    for i in metal_atom:
        if i in molecule[-31:-26]:
            c = metal_atom.index(i)
            return_list.append(metal_row[c])
            return_list.append(metal_column[c])
            return_list.append(metal_en[c])
            return_list.append(metal_zval[c])
            return_list.append(metal_radius[c])
    for j in halo_atom:
        if j in molecule[-28:-25]:
            c = halo_atom.index(j)
            return_list.append(halo_row[c])
            return_list.append(halo_column[c])
            return_list.append(halo_en[c])
            return_list.append(halo_zval[c])
            return_list.append(halo_radius[c])

    if ('S' in molecule[-28:-25]) and ('Se' not in molecule[-28:-25]):
        return_list.append(halo_row[1])
        return_list.append(halo_column[1])
        return_list.append(halo_en[1])
        return_list.append(halo_zval[1])
        return_list.append(halo_radius[1])

    if 'Se' in molecule[-28:-25]:
        return_list.append(halo_row[2])
        return_list.append(halo_column[2])
        return_list.append(halo_en[2])
        return_list.append(halo_zval[2])
        return_list.append(halo_radius[2])
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
all_vec_eng = get_all_file('/home/mx2/work/MChalc2/fix_spin')
######################################
def str_list2float_list(str_l):
    return_list = []
    for x in str_l:
        return_list.append(float(x))
    return return_list
#######################################
def get_coord(the_list):
   
    para_list_f = []
    energy_list_f = []
    bond_list_f = []
    geo_list_f = []
    for c in the_list:
        para_list=[] 
        energy_list=[]
        bond_list=[]
        try:
            with open(c, "r") as p:
                for i, line in enumerate(p):
                    coord_1 = []
                    if "#" not in line:
                        length = float(line.split()[0])
#                        bond = float(line.split()[3])
                        mag = float(line.split()[2])
                        f = float(line.split()[1]) 
                        para_list.append(length)
                        #para_list.append(bond)
                        para_list.append(mag)
                        para_list.append(length**2)
                        if len(perodic_table(c)) == 10:
                            para_list += perodic_table(c)
#                        bond_list.append(bond)
                        energy_list.append(f)
            with open(c[:-7]+'add_geo', "r") as p:
                for i, line in enumerate(p):
                    coord_1 = []
                    if ("#" not in line) and (len(line.split())==6):
                        bond = float(line.split()[3])
                        bond_list.append(bond)
            if len(energy_list) == len(bond_list):
                c = np.array(para_list).reshape(len(energy_list),13)
                c=np.insert(c,1,bond_list,axis=1)
                para_list_f+=c.tolist()
                energy_list_f+=energy_list
        except PermissionError:
            pass
        except FileNotFoundError:
            pass
#    for c in geo_list:
#        try:
#            with open(c, "r") as p:
#                for i, line in enumerate(p):
#                    coord_1 = []
#                    if ("#" not in line) and (len(line.split())==6):
#                        bond = float(line.split()[3])
#                        bond_list.append(bond)
#        except PermissionError:
#            pass
#        except FileNotFoundError:
#            pass
    a = len(energy_list_f)
#    b = len(bond_list)
    parameter = np.array(para_list_f).reshape(a,14)
    energy = np.array(energy_list_f)
#    if a == b:
#        parameter=np.insert(parameter,1,bond_list,axis=1)
#    bond = np.array(bond_list)
    return parameter, energy
######################################
parameter, y= get_coord(all_vec_eng)
scaled_para = min_max_scaler.fit_transform(parameter)
X_train, X_test, y_train, y_test = train_test_split(scaled_para, y,test_size=0.6, random_state = 3)

######################################
def regress(unit,alpha):

    score_list = []
    predict_list = []
    error_train_list = []
    error_test_list = []
    #solver_list = ['lbfgs','sgd','adam']
    for i in range(0,3):
        product_model = MLPRegressor(hidden_layer_sizes=(unit,unit,unit,),
                                     activation='relu',
                                     solver='lbfgs',
                                     learning_rate='constant',
                                     max_iter=2000,
                                     learning_rate_init=0.01,
                                     alpha=alpha)
        product_model.fit(X_train, y_train)
        #score = product_model.score(X_test, y_test)
        #score_list.append(score)
        predict_y_train =product_model.predict(X_train)
        predict_y_test =product_model.predict(X_test)

        error_train = sum(abs(predict_y_train-y_train))/len(y_train)
        error_test = sum(abs(predict_y_test-y_test))/len(y_test)
        error_train_list.append(error_train)
        error_test_list.append(error_test)

    error_train = sum(error_train_list)/3
    error_test = sum(error_test_list)/3
    return error_train,error_test
####################################
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(X_test[:200,1],abs(regress(100,0.0003)[:200]-y_test[:200]), s=10, c='r', marker="s", label='real')
#ax1.scatter(X_test[100:150,1],regress(100,0.0008)[100:150], s=10, c='r', marker="o", label='NN Prediction')
#ax1.scatter(X_test[:50,1],regress(100,0.0003)[:50], s=10, c='r', marker="o", label='NN Prediction')
#plt.show()

#print (X_test[:,1].tolist())
#print (regress())

os.chdir('/lustre/lwork/yclee/python_practice/machine_learning/chalcogenide')
unit_list = [150,200,250,300]
alpha_list = [0.0001,0.001]
for alpha in alpha_list:
    for unit in unit_list:
        train, test = regress(unit,alpha)
        with open("NN_result_3layer_0.4", "a") as p:
            p.write('bond_radius_periodic_minus the error of ' + str(unit) + ' unit in each of 3 layers with alpha = '+ str(alpha) +' is '+ 'train: ' + str(train) + ' test: ' + str(test)+"\n")

