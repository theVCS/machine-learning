import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


def gradientDescent(x, y):
    x=pd.array(x)
    y=pd.array(y)
    m_curr = c_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002

    for i in range(iterations):
        predicted = m_curr*x + c_curr
        del_m = -(2/n)*sum((y-predicted)*x)
        del_c = -(2/n)*sum(y-predicted)
        m_curr = m_curr - learning_rate*del_m
        c_curr = c_curr - learning_rate*del_c
        
    cost = 0
    predicted = m_curr*x+c_curr
    for val in (y-predicted):
        cost = cost + val*val
    print("m={}, c={}, cost={}".format(m_curr, c_curr, cost))


dt = pd.read_csv("gradientDescent/test_scores.csv")
gradientDescent(dt.math, dt.cs)

reg = linear_model.LinearRegression()
reg.fit(dt[["math"]],dt["cs"])
print(reg.coef_[0])
print(reg.intercept_)
