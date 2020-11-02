import pandas as pd
import numpy as np
auto_prices=pd.read_csv('Automobile price data _Raw_.csv')   # 汽車價格數據
auto_prices.to_html('auto1.html')       # 轉 html 檔，可在網頁上看清楚
print('欄位名稱的-改為_')
cols=auto_prices.columns
auto_prices.columns=[str.replace('-','_') for str in cols]
auto_prices.to_html('auto2.html')       # 欄位符號已經更改
print('查看欄位的資料型態')
print(auto_prices.dtypes)
print('若欄位型態為object,但是它應該是float或int,代表有其他文字')
# print(np.mean(auto_prices['price']))  # 若把帶有文字的資料進行運算會出錯
print('查看是否有遺失值')
print(auto_prices.isnull().sum())       # 統計各行的遺失值數量，但是要有文字 Nan 才會算進去
print('以上發現都是 0 ，因為數據必須是 Nan 才會計算進去')
print('所以挑選有遺失資料的欄位,準備將"?"轉換為 Nan')
'''方法1'''
# auto_prices.replace('?',np.NAN)       # 這是直接將全部 "?" 改成 NaN 的方法
'''方法2'''
# cols3=[]
# print(auto_prices.dtypes)
# for y in auto_prices.columns:           # 叫出每個欄位
#     if(auto_prices[y].dtype == np.object):     # 若有欄位內容有文字
#         auto_prices.loc[auto_prices[y] == '?', y] = np.nan    # 轉換成 Nan
#         print('嘗試轉換型態欄位:',y)
#         try:
#             pd.to_numeric(auto_prices[y])
#             auto_prices[y]=auto_prices[y].astype(float)
#             print('轉換成功')
#             cols3.append(y)    # 將有轉換的欄位名稱存入 cols3
#         except:
#             print('轉換失敗')
# print(cols3)                   # cols3 就是下面的 cols2
cols2=['price','bore','stroke','horsepower','peak_rpm','normalized_losses']
for column in cols2:
    auto_prices.loc[auto_prices[column]=='?',column]=np.nan  # loc(row,column)找出這些位置的資料
auto_prices.to_html('auto3.html')
print('再查看一次各有多少遺失值?')
print(auto_prices.isnull().sum())
print('查看欄位的資料型態')
print(auto_prices.dtypes)
#print(np.mean(auto_prices['price']))
