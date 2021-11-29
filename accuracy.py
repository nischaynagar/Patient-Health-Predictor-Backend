import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

# Cancer model
dt = pd.read_csv("cancer.csv")
dt.drop(["id"],axis="columns",inplace=True)
Y=np.array(dt["diagnosis"])
dt.drop(["diagnosis"],axis="columns",inplace=True)
X=np.array(dt)
n = len(X)
x_train = X[:int(n*0.8)]
y_train = Y[:int(n*0.8)]
x_test = X[int(n*0.8)+1:]
y_test = Y[int(n*0.8)+1:]
clf = LogisticRegression(max_iter=10000)
clf.fit(x_train,y_train.reshape(-1,))
y_pred = clf.predict(x_test)
print("Accuracy of Cancer Prediction Model: ",accuracy_score(y_test,y_pred)*100)

# Diabetes model
dt=pd.read_csv("diabetes.csv")
Y=np.array(dt["Outcome"])
dt.drop(["Outcome"],axis="columns",inplace=True)
X=np.array(dt)
n = len(X)
x_train = X[:int(n*0.8)]
y_train = Y[:int(n*0.8)]
x_test = X[int(n*0.8)+1:]
y_test = Y[int(n*0.8)+1:]

clf = LogisticRegression(max_iter=10000)
clf.fit(x_train,y_train.reshape(-1,))
y_pred = clf.predict(x_test)
print("Accuracy of Diabetes Prediction Model: ",accuracy_score(y_test,y_pred)*100)

# Heart model
dt = pd.read_csv("heart.csv")
Y=np.array(dt["target"])
dt.drop(["target"],axis="columns",inplace=True)
X=np.array(dt)
n = len(X)
x_train = X[:int(n*0.8)]
y_train = Y[:int(n*0.8)]
x_test = X[int(n*0.8)+1:]
y_test = Y[int(n*0.8)+1:]

clf = LogisticRegression(max_iter=10000)
clf.fit(x_train,y_train.reshape(-1,))
y_pred = clf.predict(x_test)
print("Accuracy of Heart Prediction Model: ",accuracy_score(y_test,y_pred)*100)

# Liver model
dt = pd.read_csv("liver.csv")
Y=np.array(dt["Dataset"])
dt.drop(["Dataset"],axis="columns",inplace=True)
X=np.array(dt)

for i in X:
    if i[1]=='Male':
        i[1] = 0
    else:
        i[1] = 1

n = len(X)
x_train = X[:int(n*0.8)]
y_train = Y[:int(n*0.8)]
x_test = X[int(n*0.8)+1:]
y_test = Y[int(n*0.8)+1:]

clf = LogisticRegression(max_iter=10000)
clf.fit(x_train,y_train.reshape(-1,))
y_pred = clf.predict(x_test)
print("Accuracy of Liver Prediction Model: ",accuracy_score(y_test,y_pred)*100)



