import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import joblib

dt = pd.read_csv("save_model/homeprices.csv")

model = linear_model.LinearRegression()
model.fit(dt[["area"]], dt["price"])

with open('save_model/model_pickle', 'wb') as f:
    pickle.dump(model,f)

joblib.dump(model,'save_model/model_joblib')