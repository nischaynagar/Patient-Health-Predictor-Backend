import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import sys
import pickle


CANCER_MODEL = int(1)
HEART_MODEL = int(2)
DIABETES_MODEL = int(3)
LIVER_MODEL = int(4)

number_of_features_map = {
    CANCER_MODEL: int(30),
    HEART_MODEL: int(13),
    DIABETES_MODEL: int(8),
    LIVER_MODEL: int(10)
}

model_name_map = {
    CANCER_MODEL: "cancer_model.sav",
    HEART_MODEL: "heart_model.sav",
    DIABETES_MODEL: "diabetes_model.sav",
    LIVER_MODEL: "liver_model.sav"
}

requested_model = int(sys.argv[1])

model_name = model_name_map[requested_model]
print(model_name)

clf = pickle.load(open(model_name, 'rb'))

user_entered = []

for i in range(2,2+number_of_features_map[requested_model]):
    user_entered.append(sys.argv[i])

print(user_entered)
user_entered = [user_entered]
print('Prediction: ',clf.predict(user_entered))