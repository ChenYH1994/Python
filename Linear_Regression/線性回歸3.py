from sklearn import datasets
import pandas as pd
import numpy as np
diabetes=datasets.load_diabetes()    # 糖尿病數據庫，目標 : 迴歸預測一年後病情
print('查看數據')
print(diabetes)
print()
print('第一位病人的數據')
print(diabetes.data[0])  # 第一位病人的數據
print()
'''
分別代表: 年齡、性別、體質指數、血壓、S1~S6(六種血清化驗的數據)，共十個數。
這些數都做了均值中心化處理，然後再用標準差乘以個體數。任何一行(比如:年齡)的數值和都將近 1。
'''
print('取一行(如:年齡)的總和:')
sum_=np.sum(diabetes.data[:,0]**2)
print(sum_)
print()

print('將數據轉成 Dataframe')
diabetes1=pd.DataFrame(diabetes.data,columns=['age','sex','body_mass_index',
                                              'average_blood_pressure','S1',
                                              'S2','S3','S4','S5','S6'
                                              ])
print(diabetes1)
print()

print('加入病情發展指數: 介於 [25,346] 之間的traget ')
diabetes1['level']=diabetes.target
print(diabetes1)
print()
print('存進 html，用瀏覽器觀看')
diabetes1.to_html('diabetes1.html')
print()

from sklearn import linear_model
from sklearn import datasets
print('取前422位病人資料做訓練')
linreg=linear_model.LinearRegression()
x_train=diabetes.data[:-20]
y_train=diabetes.target[:-20]
x_test=diabetes.data[-20:]
y_test=diabetes.target[-20:]
linreg.fit(x_train,y_train)
print('看一下迴歸係數')
print(linreg.coef_)
print()
print('做測試')
d=linreg.predict(x_test)
print(d)
print()
print('跟答案做比較')
print(y_test)
print()
print('做方差，值越接近 1，說明預測愈準確')
print(linreg.score(x_test,y_test))

