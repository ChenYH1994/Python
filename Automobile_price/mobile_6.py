import matplotlib.pyplot as plt
import seaborn as sns
import mobile_2
print('6-了解數值資料與價格的關係')
auto_prices=mobile_2.auto_prices
cols4=['curb_weight','engine_size','horsepower','city_mpg']
cols2=mobile_2.cols2
print('數值型態欄位')
print('我們可以先一個一個呈現')
#plt.figure(figsize=(12,8))
for col in cols4:
    auto_prices.plot.scatter(x=col,y='price')
    plt.show()

print('7-了解數值資料與價格的關係')
auto_prices=mobile_2.auto_prices
cols4=['curb_weight','engine_size','horsepower','city_mpg']    # 數值資料
cols5=auto_prices[cols2]
cols5.corr().to_html('cols5.html')      # 共變異係數
print(cols5.corr())

print('8-了解馬力與價錢的線性迴歸')
auto_prices=mobile_2.auto_prices
cols2=['price','horsepower']
cols6=auto_prices[cols2]
cols6.corr().to_html('cols6.html')
print(cols6.corr())
import statsmodels.api as sm
x1=sm.add_constant(cols6.horsepower,prepend=False)
lm_mod=sm.OLS(cols6.price,x1)
res=lm_mod.fit()
print(res.summary())
print('得到一張表格,請查詢coef')
print('const代表常數,horsepower代表係數')
print('price=145.2429*horsepower-2497.4812')
x0=auto_prices['horsepower']
y0=auto_prices['price']
y1=145.2429*x0-2497.4812
plt.scatter(x0,y0)
plt.plot(x0,y1,'r--')
plt.show()
