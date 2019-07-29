import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
#######################################
def str_list2float_list(str_l):
    return_list = []
    for x in str_l:
        return_list.append(float(x))
    return return_list
#######################################
def get_coord(file_target):
   
    coord_2 = []
    coord_3 = []
    with open(file_target, "r") as p:
        for i, line in enumerate(p):
            coord_1 = []
            if "#" not in line:
                #d = str_list2float_list(line.split())[0:3]
                b = float(line.split()[0])
                e = float(line.split()[3])
                f = float(line.split()[1]) 
                coord_1.append(b)
                coord_1.append(e)
                coord_3.append(f)
                coord_2 += coord_1
                a = int(len(coord_2)/2)
    all_coord_1 = np.array(coord_2).reshape(a,2)
    all_coord_2 = np.array(coord_3)
    return all_coord_1, all_coord_2
######################################
inp, y = get_coord("vector_vs_energy")

X_train, X_test, y_train, y_test = train_test_split(inp, y, random_state = 0)

def regress():
    tt = [3.38438,2.58750952154]
    j= np.array(tt).reshape(1,2)

    score_list = []
    predict_list = []
    #solver_list = ['lbfgs','sgd','adam']
    unit_list = [5,10,15,20,25,30,35,40,45,50]
    for a in unit_list:
        product_model = MLPRegressor(hidden_layer_sizes=(a,),
                                     activation='relu',
                                     solver='lbfgs',
                                     learning_rate='constant',
                                     max_iter=1000,
                                     learning_rate_init=0.01,
                                     alpha=0.0001)
        product_model.fit(X_train, y_train)
        score = product_model.score(X_test, y_test)
        score_list.append(score)
        predict =product_model.predict(j)
        predict_list.append(predict)
#tt = [3.22236,2.55410784594]
#j= np.array(tt).reshape(1,2)
    return score_list,predict_list
####################################

print (regress())
