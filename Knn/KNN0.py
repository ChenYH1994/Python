import pandas as pd
from sklearn import datasets
iris=datasets.load_iris()
print(iris)
print('1-資料探索')
print(iris.keys())
print('代表花萼的長度寬度、花瓣的長度寬度')
print(iris.feature_names)
print('代表鳶尾花的三種分類')
print(iris.target_names)
print('0 1 2 對應到鳶尾花的三個分類')
print(iris.target)
print('資料有多少筆')
print(len(iris.data))