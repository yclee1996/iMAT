import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
#from adspy_shared_utilities import load_crime_dataset


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

print (X_cancer[0])
print (len(y_cancer))
print (type(y_cancer))
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
print (type(X_train))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)


zz = X_train_scaled.tolist()
print (type(y_train))
print (len(zz[0]))
print (len(zz))

#print('Breast cancer dataset')
#print('Accuracy of NN classifier on training set: {:.2f}'
#     .format(clf.score(X_train_scaled, y_train)))
#print('Accuracy of NN classifier on test set: {:.2f}'
#     .format(clf.score(X_test_scaled, y_test)))
