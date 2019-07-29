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
    metal_zval = [11,10,11,12,13,8,9,10,11,12]
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
def get_all_file(directory,atom):
    return_list = []
    dire = []
    for x in os.listdir(directory):
        if x[:13] == 'MX2_H_LDA_mag':
            dire.append(x)
    for a in dire:
        os.chdir(directory+"/"+a)
        for i in os.listdir():
            if atom not in i:
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
                        dis = float(line.split()[4])
                        if len(perodic_table(c)) == 10:
                            para_list.append(length)
             #           para_list.append(bond)
                            para_list.append(mag)
                            para_list.append(length**2)
              #          para_list.append(dis)
                            para_list += perodic_table(c)
                            bond_list.append(bond)
                            energy_list.append(f)
        except PermissionError:
            pass
    a = len(energy_list)
    parameter = np.array(para_list).reshape(a,13)
    energy = np.array(energy_list)
    bond = np.array(bond_list)
    return parameter, energy,bond
######################################

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
        return_list[4*i] = min_max_scaler.transform(a)
        return_list[4*i+1] = b
        return_list[4*i+2] = c
        return_list[4*i+3] = a
#        para , no1, y, no2 = train_test_split(return_list[4*i], return_list[4*i+1],test_size=0.8, random_state = 3)
#        for i in range(0,len(para.tolist())):
#            new_para+=para.tolist()[i]
#        for a in range(0,len(y)):
#            new_y.append(y[a])
#    return return_list,np.array(new_para).reshape(int(len(new_para)/15),15),new_y

    return return_list

#####################################
#parameter_certain_testset,y_certain_testset,bond_certain_testset = get_coord(get_certain_testset(os.getcwd(),0,'Sc'))
#scaled_parameter_certain_testset=min_max_scaler.fit_transform(parameter_certain_testset)

######################################
def regress(unit,alpha,X_train,y_train,X_test,y_test):

    score_list = []
    predict_y_list = [None] * 4
    error_list = []
    #solver_list = ['lbfgs','sgd','adam']
    error = 1
    for i in range(0,5):
        product_model = MLPRegressor(hidden_layer_sizes=(unit,unit,unit,),
                                     activation='relu',
                                     solver='lbfgs',
                                     learning_rate='constant',
                                     max_iter=3000,
                                     learning_rate_init=0.01,
                                     alpha=alpha)
        product_model.fit(X_train, y_train)
        predict_y_test =product_model.predict(X_test)
    
        error_test = sum(abs(predict_y_test-y_test))/len(y_test)
        if error_test < error:
            error = error_test
            final_model = product_model
    return final_model
####################################
def draw_pic(a,molecule,train):
    mole_dataset = get_molecule_dataset(molecule)
    num = int(len(mole_dataset)/4)
    color_list = ['red','blue','green','purple','orange','brown']
   
    fig = plt.figure(molecule,figsize=[12,9])
    fig.suptitle(molecule)
    ax1 = fig.add_subplot(111)
    min_list = []
    error_list = []
    for i in range(0,num):
        predict_list = a.predict(mole_dataset[4*i])
        the_min = min(predict_list)
        min_list.append(the_min)
    the_min = min(min_list)
    for i in range(0,num):
        yes_list,other_list,yes_list_o,other_list_o = in_train_or_not(train,mole_dataset[4*i],mole_dataset[4*i+3])
        mole_predict = a.predict(other_list)-the_min
        error =sum(abs(mole_dataset[4*i+1]-mole_predict))/len(mole_predict)
        error_list.append(error)
        ax1.scatter(mole_dataset[4*i+3][:,0],mole_dataset[4*i+1], s=20, c=color_list[i], marker="o", label='real')        
        
#        ax1.scatter(yes_list_o[:,0],a.predict(yes_list), s=120, c=color_list[i], marker="+", label='NN')
        ax1.scatter(other_list_o[:,0],mole_predict, s=90, c=color_list[i], marker="x", label='NN')
    ax1.set_xlabel('average error = '+str(sum(error_list)/num)[:5]+' eV')
 
    fig_actual = plt.figure(molecule+'_actual',figsize=[12,9])
    fig_actual.suptitle(molecule+'_actual')
    ax2 = fig_actual.add_subplot(111)
    for i in range(0,num):
        ax2.scatter(mole_dataset[4*i+3][:,0],mole_dataset[4*i+1], s=20, c=color_list[i], marker="o", label='real')

    fig_predict = plt.figure(molecule+'_predict',figsize=[12,9])
    fig_predict.suptitle(molecule+'_predict')
    ax3 = fig_predict.add_subplot(111)
    for i in range(0,num):
        yes_list,other_list,yes_list_o,other_list_o = in_train_or_not(train,mole_dataset[4*i],mole_dataset[4*i+3])
        mole_predict = a.predict(other_list)-the_min
        ax3.scatter(other_list_o[:,0],mole_predict, s=90, c=color_list[i], marker="x", label='NN')

    plt.show()


#####################################
def draw_all_pics(a,train):
    metal_atom = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
    halo_atom = ['F2','Cl2','Br2','I2']
    all_mole_list = []
    for i in metal_atom:
        for j in halo_atom:
            all_mole_list.append(i+j)
    for molecule in all_mole_list:
        draw_pic(a,molecule,train)
####################################
def in_train_or_not(train,mole_list,mole_list_o):
    index_list = []
    yes_list = []
    other_list = []
    yes_list_o = []
    other_list_o = []
    for x in range(0,len(mole_list)):
        read = mole_list[x]
        for i in range(0,len(train)):
            counter = 0
            for z in range(0,len(read)):
                if (read)[z] == (train[i])[z]:
    
                    counter +=1
            if counter == len(read):
                yes_list.append(read)
                index_list.append(x)
    for x in range(0,len(mole_list)):
        if x not in index_list:
            other_list.append(mole_list[x])
            other_list_o.append(mole_list_o[x])
        if x in index_list:
            yes_list_o.append(mole_list_o[x])
    return np.array(yes_list), np.array(other_list),np.array(yes_list_o),np.array(other_list_o)
#######################################

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
#for atom in ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']: 
atom = sys.argv[1]
all_vec_eng = get_all_file('/home/mx2/work/MH2/fix_spin',atom)
parameter, y, bond = get_coord(all_vec_eng)
scaled_para = min_max_scaler.fit_transform(parameter)
X_train, X_test, y_train, y_test = train_test_split(scaled_para, y, random_state = 3, test_size=0.6)
model = regress(300,0.001,X_train,y_train,X_test,y_test)
halo_atom = ['F2','Cl2','Br2','I2']
all_mole_list = []
for j in halo_atom:
    all_mole_list.append(atom+j)
for molecule in all_mole_list:
    draw_pic(model,molecule,X_train)
#draw_all_pics(model,X_train)


#mole_dataset = get_molecule_dataset('CoCl2')
#a,b,c,d = in_train_or_not(X_train,mole_dataset[0],mole_dataset[3])
#print (mole_dataset[3][:,0])
#print (mole_dataset[1])
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(mole_dataset[3][:,0],mole_dataset[1], s=10, c='r', marker="s", label='real')
#plt.show()



#draw_pic(model,'MnCl2')
