import pandas as pd
import numpy as np
import mobile_1
print('針對遺失值進行處理')
auto_prices=mobile_1.auto_prices
print(auto_prices.isnull().sum())
print('查看有多少筆資料')
print(auto_prices.count())
print('有遺失值欄位的那一個row刪除')
auto_prices.dropna(axis=0,inplace=True)
print('查看有多少筆資料')
print(auto_prices.count())
print('改變欄位資料型態')
cols2=['price','bore','stroke','horsepower','peak_rpm','normalized_losses']
for column in cols2:
    auto_prices[column]=auto_prices[column].astype('float')
auto_prices.to_html('auto4.html')
print('查看欄位的資料型態')
print(auto_prices.dtypes)
print(np.mean(auto_prices['price']))