import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
min_max_scaler = preprocessing.MinMaxScaler()
name = sys.argv[1]
######################################
def perodic_table(molecule):
    return_list = []
    metal_atom = ['Sc','Ti','V','Cr','XXXXX','XXXX','XXXX','XXXX','XXXXX','XXXX']
    metal_row = [4,4,4,4,4,4,4,4,4,4]
    metal_column = [3,4,5,6,7,8,9,10,11,12]
    metal_electron = [21,22,23,24,25,26,27,28,29,30]
    metal_en = [1.36,1.54,1.63,1.66,1.55,1.83,1.88,1.91,1.90,1.65]
    metal_zval = [11,10,11,12,13,8,9,10,0,0]
    metal_ea = [-18,-8,-51,-65.2,0,-15,-64.0,-111.7,-119.2,0]
    metal_ionized = [633.1,658.8,650.9,652.9,717.3,762.5,760.4,737.1,745.5,906.4]
    metal_radius = [162,147,134,128,127,126,125,124,128,134]
    halo_atom = ['F','Cl','Br','I']
    halo_row = [2,3,4,5]
    halo_column = [17,17,17,17]
    halo_electron = [9,17,35,53]
    halo_en = [3.98,3.16,2.96,2.66]
    halo_zval = [7,7,7,7]
    halo_ea = [-328.2,-348.6,-324.5,-295.2]
    halo_ionized = [1681.0,1251.2,1139.9,1008.4]
    halo_radius = [147,175,185,198]
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
            if (i != name):
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
all_vec_eng = get_all_file('/home/mx2/work/MH2/fix_spin')
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
                        dis = float(line.split()[4])
                        if len(perodic_table(c)) == 10:
                            para_list.append(length)
                            para_list.append(bond)
                            para_list.append(mag)
                            para_list.append(length**2)
                            para_list.append(dis)
                            para_list += perodic_table(c)
                            bond_list.append(bond)
                            energy_list.append(f)
        except PermissionError:
            pass
    a = len(energy_list)
    parameter = np.array(para_list).reshape(a,15)
    energy = np.array(energy_list)
    bond = np.array(bond_list)
    return parameter, energy,bond
######################################
parameter, y, bond = get_coord(all_vec_eng)

scaled_para = min_max_scaler.fit_transform(parameter)

X_train, X_test, y_train, Y_test = train_test_split(scaled_para, y,test_size=0.6, random_state = 3)
#####################################
def get_molecule_dataset(molecule):
    path_list = get_certain_testset(os.getcwd(),molecule)
    num = len(path_list)
    return_list = [None] * 4 *num
    new_para = []
    new_y = []
    for i in range(0,num):
        the_list= [None]
        the_list[0] = path_list[i]
        a,b,c = get_coord(the_list)
        return_list[4*i] = a#min_max_scaler.transform(a)
        return_list[4*i+1] = b
        return_list[4*i+2] = c
        return_list[4*i+3] = a
       # rad_x_list = return_list[4*i].tolist()
       # the_num = random.randrange(len(rad_x_list))
       # new_para += rad_x_list[the_num]
       # rad_y = return_list[4*i+1]
       # new_y.append(rad_y[the_num])
        para , no1, y, no2 = train_test_split(return_list[4*i], return_list[4*i+1],test_size=0.2, random_state = 3)
        for i in range(0,len(para.tolist())):
            new_para+=para.tolist()[i]
        for a in range(0,len(y)):
            new_y.append(y[a])
    return return_list,np.array(new_para).reshape(int(len(new_para)/15),15),new_y

######################################
mole_test,new_para,new_y = get_molecule_dataset(name)

#X_train = np.insert(X_train,-1,new_para,axis = 0)
#y_train = np.insert(y_train,-1,new_y,axis = 0)

#####################################
def regress(unit,alpha):

    score_list = []
    predict_list = []
    error_train_list = []
    error_test_list = []
    #solver_list = ['lbfgs','sgd','adam']
    product_model = MLPRegressor(hidden_layer_sizes=(unit,unit,unit,),
                                 activation='relu',
                                 solver='lbfgs',
                                 learning_rate='constant',
                                 max_iter=2000,
                                 learning_rate_init=0.01,
                                 alpha=alpha)
    product_model.fit(X_train, y_train)

    predict_y_test = []
    y_test = []
    for i in range(0,len(mole_test)):
        if i % 4 == 0:
            predict = product_model.predict(mole_test[i])
            predict_y_test +=predict.tolist()
        if i % 4 == 1:
            y_test += mole_test[i].tolist()
    error_test = sum(abs(np.array(predict_y_test)-np.array(y_test)))/len(y_test)

    return error_test
####################################
os.chdir('/lustre/lwork/yclee/python_practice/machine_learning/test_halo')
unit_list = [150,200,250,300]
alpha_list = [0.0001]
for alpha in alpha_list:
    for unit in unit_list:
        test = regress(unit,alpha)
        with open("NN_result_3layer_0.4", "a") as p:
            p.write(name+'ttt_dis_radius_periodic_minus the error of ' + str(unit) + ' unit in each of 3 layers with alpha = '+ str(alpha) +' is '+ 'test: ' + str(test)+"\n")
