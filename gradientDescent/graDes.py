# maths behind linear regression
# err = sum(square(yi - (mxi + c)))/n
# here we need to minimize the err so what we do is partially differentiate with respect to m and c
# new_m = old_m - learning_rate * d(err)/dm
# new_c = old_c - learning_rate * d(err)/dc

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math


def gradientDescent(x, y):
    m_curr = c_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        predicted = m_curr*x + c_curr
        cost = 0
        for val in (y-predicted):
            cost = cost + val*val
        plt.plot(x,predicted,color="green")
        del_m = -(2/n)*sum((y-predicted)*x)
        del_c = -(2/n)*sum(y-predicted)
        m_curr = m_curr - learning_rate*del_m
        c_curr = c_curr - learning_rate*del_c
        print("m={}, c={}, cost={}".format(m_curr,c_curr,cost/n))

    plt.show()

x = [1,2,3,4,5]
y = [5,6,7,8,9]
x = np.array(x)
y = np.array(y)

gradientDescent(x, y)
# m = 1, c = 4
