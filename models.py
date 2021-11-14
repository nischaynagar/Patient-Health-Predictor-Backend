import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# Cancer model
dt = pd.read_csv("cancer.csv")
dt.drop(["id"],axis="columns",inplace=True)
Y=np.array(dt["diagnosis"])
dt.drop(["diagnosis"],axis="columns",inplace=True)
X=np.array(dt)
print(len(X[0]))
clf = LogisticRegression(max_iter=10000)
clf.fit(X,Y.reshape(-1,))

filename = 'cancer_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# Diabetes model
dt=pd.read_csv("diabetes.csv")
Y=np.array(dt["Outcome"])
dt.drop(["Outcome"],axis="columns",inplace=True)
X=np.array(dt)
print(len(X[0]))

clf = LogisticRegression(max_iter=10000)
clf.fit(X,Y.reshape(-1,))

filename = 'diabetes_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# Heart model
dt = pd.read_csv("heart.csv")
Y=np.array(dt["target"])
dt.drop(["target"],axis="columns",inplace=True)
X=np.array(dt)
print(len(X[0]))

clf = LogisticRegression(max_iter=10000)
clf.fit(X,Y.reshape(-1,))

filename = 'heart_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# Liver model
dt = pd.read_csv("liver.csv")
Y=np.array(dt["Dataset"])
dt.drop(["Dataset"],axis="columns",inplace=True)
X=np.array(dt)
print(len(X[0]))

for i in X:
    if i[1]=='Male':
        i[1] = 0
    else:
        i[1] = 1

clf = LogisticRegression(max_iter=10000)
clf.fit(X,Y.reshape(-1,))

filename = 'liver_model.sav'
pickle.dump(clf, open(filename, 'wb'))




