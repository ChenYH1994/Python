import pandas as pd
import numpy as np         # 主題 : 分析並找出預測德國信貸的客戶的信用度的特徵
import bank_note           # 德國信貸的替代碼資料
credit=pd.read_csv('German_Credit.csv')     # 讀取德國信貸的用戶資料
print('1.先將資料轉為 html，用網頁瀏覽')
credit.to_html('credit1.html')
print('2.請加入欄位名稱，並且將"-"換成底線"_" ')  # 此資料欄位名稱都是'_'
credit=pd.read_csv('German_Credit.csv',names=bank_note.columns)
credit.to_html('credit2.html')
print('3.查看欄位的資料型態，並且留意類型是否是 object')
print(credit.dtypes)
print('4.查看是否有遺失值')
print(credit.isnull().any())
print('5.把有遺失的資料換成 Nan，這筆資料沒有')


