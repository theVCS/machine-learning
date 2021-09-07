import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

# data processing
dt = pd.read_csv("multivariateRegression/homeprices.csv")
medianBed = math.floor(dt["bedrooms"].median())
dt["bedrooms"] = dt["bedrooms"].fillna(medianBed)

# linear model
reg = linear_model.LinearRegression()
reg.fit(dt[["area", "bedrooms", "age"]], dt["price"])

print(reg.predict([[400,50,10]]))
