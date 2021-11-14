import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import sys
import pickle

dt = pd.read_csv("liver.csv")
Y=np.array(dt["Dataset"])
dt.drop(["Dataset"],axis="columns",inplace=True)
X=np.array(dt)

for i in X:
    if i[1]=='Male':
        i[1] = 0
    else:
        i[1] = 1


X_train = []
Y_train = []
X_test = []
Y_test = []
n = len(X)

for i in range(0,int(0.7*n)):
    X_train.append(X[i])
    Y_train.append(Y[i])

for i in range(int(0.7*n),n):
    X_test.append(X[i])
    Y_test.append(Y[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

clf = LogisticRegression(max_iter=5000)
clf.fit(X_train,Y_train.reshape(-1,))

Y_pred = clf.predict(X_test)
Y_prob = clf.predict_proba(X_test)
Y_test = Y_test.reshape(-1,)

m = len(Y_test)
correct = 0
for i in range(len(Y_test)):
    if Y_test[i]==Y_pred[i]:
        correct += 1

accuracy = correct*100/m

print('Accuracy: ',accuracy)



