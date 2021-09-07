import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

dt = pd.read_csv("linearRegression/exercise.csv")

# plt.scatter(dt["year"], dt["pci"])
# plt.show()

reg = linear_model.LinearRegression()
reg.fit(dt[["year"]],dt["pci"])

print(reg.predict([[2020]]))