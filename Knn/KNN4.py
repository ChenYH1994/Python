import KNN2
import numpy as np
from sklearn.model_selection import train_test_split    # 把訓練的資料進行分類
iris2=KNN2.iris2
num_cols=['sepal length','sepal width','petal length','petal width']
iris_split=train_test_split(iris2,test_size=75)     # 資料分類 : 訓練集 75筆、測試集 75筆
print(iris_split[0])
print('訓練的特徵')
X_train=iris_split[0].iloc[:,:4]                    # 選 前 4 筆資料來訓練
print(X_train)
print('訓練的標籤')
y_train=iris_split[0].iloc[:,4]                     # 第 5 筆資料作為訓練的答案
print(y_train)
print('標籤還得做一個動作')
y_train=np.ravel(iris_split[0].iloc[:,4])           # 把答案轉成一維
print(y_train)
print('測試的特徵')
X_test=iris_split[1].iloc[:,:4]                     # 剩下 75 筆作為模型的測試用
print(X_test)
print('測試的標籤')
y_test=np.ravel(iris_split[1].iloc[:,4])            # 測試集的答案
print(y_test)