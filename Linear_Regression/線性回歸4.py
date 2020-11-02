import 線性回歸3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
linreg=linear_model.LinearRegression()
diabetes=線性回歸3.datasets
x_train=線性回歸3.x_train
y_train=線性回歸3.y_train
x_test=線性回歸3.x_test
y_test=線性回歸3.y_test
x0_test=x_test[:,0]
print(x0_test)
x0_train=x_train[:,0]
x0_test=x0_test[:,np.newaxis]      # 每個數據都當一筆資料
print(x0_test)
x0_train=x0_train[:,np.newaxis]
print('對十個生理因素進行迴歸分析')
plt.figure(figsize=(8,15))
for col in range(0,10):
    xi_test=x_test[:,col]
    xi_train=x_train[:,col]
    xi_test=xi_test[:,np.newaxis]
    xi_train=xi_train[:,np.newaxis]
    linreg.fit(xi_train,y_train)
    y=linreg.predict(xi_test)
    plt.subplot(5,2,col+1)
    plt.scatter(xi_test,y_test,color='k')
    plt.plot(xi_test,y,color='b',linewidth=3)
plt.show()

