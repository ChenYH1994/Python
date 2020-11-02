import pandas as pd
import numpy as np
import mobile_2
print('3-探索資料')
print('查看數值型態')
auto_prices=mobile_2.auto_prices
print(auto_prices.describe())
auto5=auto_prices.describe()
auto5.to_html('auto5.html')
print('查看字串型態')
print(auto_prices.describe(include=['object']))
auto6=auto_prices.describe(include=['object'])
auto6.to_html('auto6.html')
cols3=['make','fuel_type','aspiration','num_of_doors',
       'body_style','drive_wheels','engine_location','engine_type',
       'num_of_cylinders','fuel_system']
for column in cols3:
    print('這是:',column)
    print(auto_prices[column].value_counts())  # 每一欄位的資料統計