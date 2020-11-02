import matplotlib.pyplot as plt
import seaborn as sns
import mobile_2
print('5-資料密度顯示')
auto_prices=mobile_2.auto_prices
cols2=mobile_2.cols2
print('數值型態欄位')
print('我們可以先一個一個呈現')
#plt.figure(figsize=(12,8))
for col in cols2:
    plt.title(col)
    sns.set_style('whitegrid')
    sns.distplot(auto_prices[col],bins=10,rug=True,hist=False)
    plt.show()