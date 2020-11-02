import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.linear_model
temperatures = np.array([29, 28, 34, 31,25, 29, 32, 31,24, 33, 25, 31,26, 30])
drink_sales = np.array([7.7, 6.2, 9.3, 8.4,5.9, 6.4, 8.0, 7.5,5.8, 9.1, 5.1, 7.3,6.5, 8.4])
X = pd.DataFrame(temperatures, columns=["Temperature"])       # 飲料溫度
target = pd.DataFrame(drink_sales, columns=["Drink_Sales"])   # 飲料平均銷售量
y = target["Drink_Sales"]
lm = sklearn.linear_model.LinearRegression()         #
lm.fit(X, y)                                         # X代表訓練的資料   y代表測試資料
print("迴歸係數:", lm.coef_)
print("截距:", lm.intercept_ )


