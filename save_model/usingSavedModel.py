import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import joblib

model = linear_model.LinearRegression()

# with open("save_model/model_pickle", "rb") as f:
#     model = pickle.load(f)

model = joblib.load("save_model/model_joblib")

print(model.predict([[5000]]))