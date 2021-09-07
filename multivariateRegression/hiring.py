import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n
import math

dt = pd.read_csv("multivariateRegression/hiring.csv")

# data processing
dt["experience"] = dt["experience"].fillna("zero")
for i in range(0,len(dt["experience"])):
    dt["experience"][i]=w2n.word_to_num(dt["experience"][i])

medianTS = dt["test_score(out of 10)"].median()
medianTS = math.floor(medianTS)
dt["test_score(out of 10)"] = dt["test_score(out of 10)"].fillna(medianTS)

# linear regression model
reg = linear_model.LinearRegression()
reg.fit(dt[["experience", "test_score(out of 10)", "interview_score(out of 10)"]], dt["salary($)"])

print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))
