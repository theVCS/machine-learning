import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dp = pd.read_csv("linearRegression/homeprices.csv")
# print(dp)

# plotting graph on points
# plt.xlabel("area")
# plt.ylabel("price")
# plt.scatter(dp["area"], dp["price"])
# plt.show()

# print(dp.area)
reg = linear_model.LinearRegression()
reg.fit(dp[["area"]],dp["price"])

#y = mx + c

#=> m
# print(reg.coef_)

#=> c
# print(reg.intercept_)

areas = pd.read_csv("linearRegression/areas.csv")
areas["price"] = reg.predict(areas)

plt.plot(areas[["area"]], areas['price'])
plt.show()

areas.to_csv("linearRegression/predictions.csv")
